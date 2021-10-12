import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# TODO: remove this dependency
# from torchsearchsorted import searchsorted
from torch import searchsorted


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(device=x.device))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['random_feature']:
            # np.random.seed(5)
            # freq_bands = 2 ** 15 * np.random.randn(N_freqs)
            freq_bands = 2.0**torch.linspace(0., max_freq, steps=N_freqs)
        elif self.kwargs['log_sampling']:
            freq_bands = 2.0**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3, i=0):
    if i == -1:
        return nn.Identity(), input_dims

    embed_kwargs = {
                'include_input' : True,
                'random_feature' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


class Velocity(nn.Module):
    def __init__(self, base, embed_fn, sin_init, W, skips, zero_vel):
        super(Velocity, self).__init__()
        self.pts_linears = nn.ModuleList(
            [nn.Linear(4, W)] + [nn.Linear(W, W) for i in range(4)])
        # self.pts_linears = pts_linears
        # self.pts_linears = base
        self.sin_init = sin_init
        self.zero_vel = zero_vel

        # self.velocity_linear_1 = nn.Linear(W, W)
        self.velocity_linear_2 = nn.Linear(W, 3)

        self.embed_fn = embed_fn
        self.skips = skips
        self.velocity_linear_2.weight.data.fill_(0.0)

    def concat_pts(self, t, inp, T_batch):
        if T_batch is None:
            T_batch = t[:, None].repeat(batch_size, 1)
            sign = torch.ones(batch_size).to(inp.device)
        else:
            # sign = ((T_batch[:, 1] > T_batch[:, 0]).float() - 0.5) * 2
            sign = torch.ones(T_batch.size(0)).to(inp.device)
            T_batch = T_batch[:, :1] + t

        inp = torch.cat([inp, T_batch], dim=1)
        return inp, sign

    def forward(self, t, inp, T_batch=None, reverse=False):

        input_pts, sign = self.concat_pts(t, inp, T_batch)

        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)

        if self.sin_init:
            h = input_pts * 30
        else:
            h = input_pts

        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)

            if self.sin_init:
                h = torch.sin(h)
            else:
                h = F.relu(h)

            # if i in self.skips:
            #     h = torch.cat([input_pts, h], -1)

        # velocity = self.velocity_linear_2(F.relu(self.velocity_linear_1(h)))
        velocity = self.velocity_linear_2(h)

        if self.zero_vel:
            velocity = F.relu(velocity)
            velocity = F.relu(-velocity)

        if reverse:
            velocity = -1 * velocity

        return velocity

    def forward_velocity(self, input_pts):
        if self.zero_vel:
            return torch.zeros_like(input_pts)[..., :1]

        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)

        if self.sin_init:
            h = input_pts * 30
        else:
            h = input_pts

        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)

            if self.sin_init:
                h = torch.sin(h)
            else:
                h = F.relu(h)

            # if i in self.skips:
            #     h = torch.cat([input_pts, h], -1)

        # velocity = self.velocity_linear(h)
        # velocity = self.velocity_linear_2(F.relu(self.velocity_linear_1(h)))
        velocity = self.velocity_linear_2(h)

        if self.zero_vel:
            velocity = F.relu(velocity)
            velocity = F.relu(-velocity)

        return velocity

# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False, sin_init=False, velocity=False, embed_fn=None, no_decomp=False, zero_vel=False):
        """
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.sin_init = sin_init
        self.velocity = velocity
        self.embed_fn = embed_fn
        self.no_decomp = no_decomp

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.view_linear = nn.Linear(input_ch_views, W)
        # self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        self.views_linears = nn.ModuleList(
            [nn.Linear(2 * W, W//2)] + [nn.Linear(W//2, W//2) for i in range(1)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, output_ch)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

        if self.sin_init:
            self.initialize()
            torch.nn.init.xavier_uniform_(self.alpha_linear.weight)
            torch.nn.init.xavier_uniform_(self.rgb_linear.weight)
            self.first_initialize()

        self.velocity_module = Velocity(W, None, sin_init, W, skips, zero_vel)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                inp_dim = m.weight.data.size(0)
                range_val = (6 / inp_dim) ** 0.5 / 30.
                m.weight.data.uniform_(-range_val, range_val)
                m.bias.data.fill_(0.0)

    def first_initialize(self):
        for m in [self.view_linear, self.pts_linears[0]]:
            inp_dim = m.weight.data.size(0)
            range_val = (1 / inp_dim)
            m.weight.data.uniform_(-range_val, range_val)
            m.bias.data.fill_(0.0)

    def forward_pts(self, x, render=False):
        input_pts = x

        if self.sin_init:
            h = input_pts * 30
        else:
            h = input_pts

        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)

            if self.sin_init:
                h = torch.sin(h)
            else:
                h = F.relu(h)

        if render:
            return self.alpha_linear(h)
        else:
            return h

    def forward_velocity(self, x):
        input_pts = x

        if self.sin_init:
            h = input_pts * 30
        else:
            h = input_pts

        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)

            if self.sin_init:
                h = torch.sin(h)
            else:
                h = F.relu(h)

            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        velocity = self.velocity_linear(h)

        return velocity

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = self.forward_pts(input_pts)

        if self.use_viewdirs:
            outputs = self.alpha_linear(h)
            rgb, alpha = outputs[:, :3], outputs[:, 3:]
            feature = self.feature_linear(h)

            if self.sin_init:
                feature = torch.sin(feature)
                view_feature = torch.sin(self.view_linear(30 * input_views))
            else:
                view_feature = self.view_linear(input_views)

            h = torch.cat([feature, view_feature], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)

                if self.sin_init:
                    h = torch.sin(h)
                else:
                    h = F.relu(h)

#             rgb_res = self.rgb_linear(h)
#             rgb = rgb_res + rgb
#             outputs = torch.cat([rgb, alpha, rgb_res], -1)
            if self.no_decomp:
                rgb = self.rgb_linear(h)
                rgb_res = torch.zeros_like(rgb)
                outputs = torch.cat([rgb, alpha, rgb_res], -1)
            else:
                rgb_res = self.rgb_linear(h)
                rgb = rgb_res + rgb
                outputs = torch.cat([rgb, alpha, rgb_res], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



# Ray helpers
def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1).to(c2w.device)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def get_rays_coord(h, w, H, W, focal, c2w):
    dirs = np.array([(w-W*.5)/focal, -(h-H*.5)/focal, -1])
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)

    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    u = u.to(weights.device)

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = searchsorted(cdf.detach(), u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min(cdf.shape[-1]-1 * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples
