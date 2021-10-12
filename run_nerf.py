import os, sys
import cv2
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ssim import ssim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_blender import load_blender_data
from load_gibson import load_gibson_data

from skimage.io import imread
from skimage.transform import resize
from lpip import models
from torchdiffeq import odeint, odeint_adjoint
from tqdm import tqdm, trange



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_device = torch.device("cpu")
np.random.seed(0)
DEBUG = False
percept_model = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True)


def batchify(fn, chunk):
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def model_copy(model, model_copy):
    for p, p_copy in zip(model.parameters(), model_copy.parameters()):
        p_copy.data[:] = p.data[:]

def generate_data(loc_t_before, loc_t_after):
    diff = 0

    while True:
        ix = np.random.randint(loc_t_after.size(0))
        diff = (loc_t_after[ix] - loc_t_before[ix]).mean().item()
        t_step = torch.Tensor([0, diff]).to(loc_t_before.device)

        if t_step[1] < 0:
            t_step = -1 * t_step
            reverse = True
        else:
            reverse = False

        if diff != 0:
            return diff, t_step, ix, reverse

def batchify_point(fn, chunk):
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn.forward_pts(inputs[i:i+chunk], render=True) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

def flatten(arr):
    s = arr.shape
    return arr.reshape((s[0]*s[1], *s[2:]))


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):

    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs[:, :, :viewdirs.size(1)].shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def run_network_point(inputs, fn, embed_fn, netchunk=1024*64, velocity=False):

    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    outputs_flat = batchify_point(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])

    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):

    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    for k in all_ret:
        try:
            all_ret[k] = torch.cat(all_ret[k], 0)
        except:
            continue

    return all_ret


def render(H, W, focal, chunk=1024*64, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None, timestep=None,
                  **kwargs):

    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
        kwargs['render_image'] = True
    else:
        # use provided ray batch
        rays_o, rays_d = rays
        kwargs['render_image'] = False


    if timestep is not None:
        rays_o = torch.cat([rays_o, torch.ones_like(rays_o)[:, :, :1] * timestep], dim=-1)


    if len(rays_d.shape) == 2:
        rays_d = rays_d[:, :3]

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

        if kwargs['use_time']:
            s = rays_o.size()
            rays_o_flat = rays_o.view(-1, s[-1])
            viewdirs = torch.cat([viewdirs, torch.Tensor(rays_o_flat[..., -3:]).to(viewdirs.device)], dim=-1)

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,4]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        if k in ["max_pt", "min_pt"]:
            all_ret[k] = all_ret[k]
        else:
            try:
                k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
                all_ret[k] = torch.reshape(all_ret[k], k_sh)
            except:
                pass

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, render_timesteps, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, velocity=False, render_res=False):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    # compute statistics for rendering
    psnrs = []
    mses = []
    lpips = []
    ssims = []
    accs = []

    for i, (c2w, timestep) in enumerate(tqdm(zip(render_poses, render_timesteps))):
        print(i, time.time() - t)
        t = time.time()
        if type(focal) == np.ndarray:
            focal = float(focal.mean())

        if render_res:
            rgb, disp, acc, _ = render(2*H, 2*W, focal * 2, chunk=8*1024, c2w=c2w[:3,:4], timestep=timestep, velocity=velocity, **render_kwargs)
        else:
            rgb, disp, acc, _ = render(H, W, focal, chunk=8*1024, c2w=c2w[:3,:4], timestep=timestep, velocity=velocity, **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy() / disp.cpu().numpy().max())
        accs.append(acc.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            psnrs.append(p)

            mse = np.square(rgb.cpu().numpy() - gt_imgs[i]).mean()
            mses.append(mse)

            gt_img = torch.Tensor(gt_imgs[i]).to(rgb.device)
            ssim_val = ssim(rgb[None, :, :, :], gt_img[None, :, :, :])
            ssims.append(ssim_val.item())

            # Center images for LPIP
            rgb = (rgb - 0.5) * 2.0
            gt_img = (gt_img - 0.5) * 2.0

            with torch.no_grad():
                d = percept_model.forward(rgb[None, :, :, :].permute(0, 3, 1, 2), gt_img[None, :, :, :].permute(0, 3, 1, 2))

            lpips.append(d.item())

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    print("PSNR ", np.mean(psnrs))
    print("MSE ", np.mean(mses))
    print("LPIPS ", np.mean(lpips))
    print("SSIMS ", np.mean(ssims))
    print("here")

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)
    accs = np.stack(accs, 0)

    if velocity:
        # min_val = accs.min(axis=0).min(axis=0).min(axis=0)
        # max_val = accs.max(axis=0).max(axis=0).min(axis=0)
        min_val = accs.min()
        max_val = accs.max()
        accs = np.clip((accs - min_val) / (max_val - min_val + 1e-3), 0, 1)

    return rgbs, disps, accs, mses

def compute_contrast(f1, f2, f3=None):
    f1 = F.normalize(f1, p=2, dim=-1)
    f2 = F.normalize(f2, p=2, dim=-1)

    dot_sim = (f1 * f2).sum(dim=-1) / 0.08

    f1_expand = f1[:, None, :]

    if f3 is not None:
        f2_expand = torch.cat([f3[None, :, :], f2[None, :, :]], dim=1)
    else:
        f2_expand = f2[None, :, :]

    partition_func = (f1_expand * f2_expand).sum(dim=-1) / 0.08
    partition_func = torch.logsumexp(partition_func, dim=-1)

    loss_contrast = -dot_sim + partition_func
    loss_contrast = loss_contrast.mean()
    return loss_contrast


def create_nerf(args):
    embed_fn, input_ch = get_embedder(args.multires, input_dims=4, i=args.i_embed)
    # embed_fn, input_ch = get_embedder(args.multires, input_dims=3, i=args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        if args.use_time:
            embeddirs_fn, input_ch_views = get_embedder(args.multires_views, input_dims=4, i=args.i_embed)
        else:
            embeddirs_fn, input_ch_views = get_embedder(args.multires_views, input_dims=3, i=args.i_embed)

    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, no_decomp=args.no_decomp, use_viewdirs=args.use_viewdirs, sin_init=args.sin_init, velocity=args.velocity, embed_fn=embed_fn).to(device)
    model_copy = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, no_decomp=args.no_decomp, use_viewdirs=args.use_viewdirs, sin_init=args.sin_init, velocity=args.velocity, embed_fn=embed_fn).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    model_fine_copy = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, no_decomp=args.no_decomp, use_viewdirs=args.use_viewdirs, sin_init=args.sin_init, velocity=args.velocity, embed_fn=embed_fn).to(device)
        model_fine_copy = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, no_decomp=args.no_decomp, use_viewdirs=args.use_viewdirs, sin_init=args.sin_init, velocity=args.velocity, embed_fn=embed_fn).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)


    network_query_fn_pt = lambda inputs, network_fn : run_network_point(inputs, network_fn,
                                                                embed_fn=embed_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if '.tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step'] + 1
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        model_copy.load_state_dict(ckpt['network_fn_state_dict'])

        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
            model_fine_copy.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'network_query_fn_pt' : network_query_fn_pt,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'network_fine_copy' : model_fine_copy,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'network_fn_copy' : model_copy,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        'use_time' : args.use_time,
    }

    # NDC only good for LLFF-style forward facing data
    # if args.dataset_type != 'llff' or args.no_ndc:
    print('Not ndc!')
    render_kwargs_train['ndc'] = False
    render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False, v_val=None, render_image=False):
    """ A helper function for `render_rays`.
    """
    white_bkgd = False
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw) * dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).to(dists.device).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn_like(raw[...,3]) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    # import pdb
    # pdb.set_trace()
    # print(alpha)
    # print(rgb)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha.device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    # import pdb
    # pdb.set_trace()
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    weights_depth = torch.sum(weights, dim=-1, keepdim=True)
    final_sum = 1 - weights_depth
    weights_depth = torch.cat([weights, final_sum], dim=-1)

    z_vals_depth = torch.cat([z_vals, torch.ones((z_vals.shape[0], 1)).to(z_vals.device) * 1e5], dim=-1)


    depth_map = torch.sum(weights_depth[..., :-1] * z_vals_depth[..., :-1], -1)

    disp_map = 1./(1 + depth_map)

    if v_val is not None:
        # idx = weights.max(dim=1)[1]
        # idx = idx[:, None, None].repeat(1, 1, 3)
        # acc_map = torch.gather(v_val, 1, idx)[:, 0, :]
        acc_map = (weights[:, :, None] * v_val[:, :, :]).sum(dim=1)
    else:
        acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map, alpha


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                velocity=False,
                **kwargs):
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:4], ray_batch[:,4:7] # [N_rays, 3] each

    if kwargs['use_time']:
        viewdirs = ray_batch[:,-4:] if ray_batch.shape[-1] > 8 else None
    else:
        viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None

    # viewdirs = None
    bounds = torch.reshape(ray_batch[...,7:9], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    # N_samples = N_samples + random.randint(-20, 20)

    t_vals = torch.linspace(0., 1., steps=N_samples).to(bounds.device)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    model_fine = network_fine
    velocity_module = model_fine.velocity_module

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand_like(z_vals)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = torch.cat([rays_o[...,None,:3] + rays_d[...,None,:] * z_vals[...,:,None], rays_o[...,None, 3:].repeat(1, z_vals.size(-1), 1)], dim=-1) # [N_rays, N_samples, 3]

#     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn)
    pts_orig = pts
    raw_orig = raw

    rgb_map, disp_map, acc_map, weights, depth_map_coarse, alpha = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest, render_image=kwargs['render_image'])
    weights_orig = weights

    max_idx = weights.max(dim=1)[1]
    max_idx = max_idx[:, None, None].repeat(1, 1, 4)
    max_depth_pts = torch.gather(pts, 1, max_idx)
    max_pt = pts[:, :, :3].max(dim=0)[0].max(dim=0)[0]
    min_pt = pts[:, :, :3].min(dim=0)[0].min(dim=0)[0]


    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = torch.cat([rays_o[...,None,:3] + rays_d[...,None,:] * z_vals[...,:,None], rays_o[...,None, 3:].repeat(1, z_vals.size(-1), 1)], dim=-1) # [N_rays, N_samples, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)
        rgb_res = raw[..., -3:]
        rgb_pred = raw[..., :3]
        if velocity:
            pts_list = torch.chunk(pts, 10, dim=0)
            v_vals = []

            for pts_i in pts_list:
                v_vel = velocity_module.forward_velocity(pts_i)
                v_vals.append(v_vel)

            v_val = torch.cat(v_vals, dim=0)
        else:
            v_val= None

        rgb_map, disp_map, acc_map, weights, depth_map, alpha = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest, v_val=v_val, render_image=kwargs['render_image'])

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'depth_map': depth_map, 'depth_map_coarse': depth_map_coarse, 'acc_map' : acc_map, 'min_pt': min_pt, 'max_pt': max_pt, 'rgb_res': rgb_res, 'weights': weights, 'pts': pts, 'alpha': alpha, 'raw_pts': raw, 'z_vals': z_vals, 'rays_d': rays_d, 'rays_o': rays_o}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print("! [Numerical Error] {} contains nan or inf.".format(k))

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=4, help='layers in network')
    parser.add_argument("--netwidth", type=int, default=512, help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=4, help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=512, help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250000, help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', help='do not reload weights from saved ckpt')
    parser.add_argument("--use_time", action='store_true', help='add time to pose regression')
    parser.add_argument("--ft_path", type=str, default=None, help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0, help='number of additional fine samples per ray')
    parser.add_argument("--frames", type=int, default=1000, help='maximum number of frames to load')
    parser.add_argument("--perturb", type=float, default=1., help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=1e0, help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--ood_render", action='store_true', help='render ood')
    parser.add_argument("--rotate_render", action='store_true', help='render rotate')
    parser.add_argument("--camera_render", action='store_true', help='render the start of the camera')
    parser.add_argument("--camera_render_after", action='store_true', help='render the end of the camera')

    parser.add_argument("--render_only", action='store_true', help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', help='render the test set instead of render_poses path')
    parser.add_argument("--noise", action='store_true', help='add noise to images (for video processing)')
    parser.add_argument("--pouring", action='store_true', help='pouring fluid')
    parser.add_argument("--fern", action='store_true', help='use the fern version of the LLFF instead')
    parser.add_argument("--debug", action='store_true', help='debug the model')
    parser.add_argument("--render_factor", type=int, default=0, help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--no_decomp", action='store_true', help='no decomposition of the view direction')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=1, help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', help='options : armchair / cube / greek / vase')
    parser.add_argument("--velocity", action='store_true', help='supervise velocity')
    parser.add_argument("--surface_loss", action='store_true', help='supervise surface revoery')
    parser.add_argument("--vel_loss", action='store_true', help='supervise veloicty_loss')
    parser.add_argument("--rgb_loss", action='store_true', help='supervise rgb loss')
    parser.add_argument("--grad_penalty", action='store_true', help='penalty of the gradient')
    parser.add_argument("--depth_loss", action='store_true', help='supervise rgb loss')
    parser.add_argument("--bkg_loss", action='store_true', help='supervise background velocity loss')
    parser.add_argument("--uniform_vel_loss", action='store_true', help='the robot has the same velocity')
    parser.add_argument("--bkg_no_rgb_loss", action='store_true', help='do not calculate rgb loss on small position change points')
    parser.add_argument("--unsup_velocity", action='store_true', help='unsupervised discovery of velocity')
    parser.add_argument("--use_past_rays", action='store_true', help='used past rays')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--contrast_random", action='store_true', help='contrast with random files as opposed ')
    parser.add_argument("--half_res", action='store_true', help='load blender synthetic data at 400x400 instead of 800x800')
    parser.add_argument("--camera_depth", action='store_true', help='return the depth of each rays as additional supervision for NeRF')
    parser.add_argument("--no_optical_flow", action='store_true', help='don"t enforce optical flow')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--sin_init", action='store_true', help='initialize using the sin function')
    parser.add_argument("--lindisp", action='store_true', help='sampling linearly in disparity rather than depth')
    parser.add_argument("--optical_flow", action='store_true', help='sampling linearly in disparity rather than depth')
    parser.add_argument("--scene_flow", action='store_true', help='utilize scene flow to train the model')
    parser.add_argument("--scene_flow_unsup", action='store_true', help='utilize scene flow to train the model')
    parser.add_argument("--unsup_vel", action='store_true', help='unsupervised discovery of velocity')
    parser.add_argument("--spherify", action='store_true', help='set for spherical 360 scenes')
    parser.add_argument("--render_res", action='store_true', help='test higher resolution rendering with flow')
    parser.add_argument("--llffhold", type=int, default=8, help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=10000, help='frequency of render_poses video saving')

    return parser



def train():

    parser = config_parser()
    args = parser.parse_args()


    # Load data

    if args.dataset_type == 'llff':
        if args.fern:
            dataset = FernDataset(args.datadir, args)
            train_dataloader = DataLoader(dataset, num_workers=4, batch_size=4, shuffle=True, pin_memory=False, collate_fn=dataset.collate_fn, drop_last=False)
        else:
            dataset = SFMDataset(args.datadir, args)
            train_dataloader = DataLoader(dataset, num_workers=8, batch_size=16, shuffle=True, pin_memory=False, collate_fn=dataset.collate_fn, drop_last=False)
        hwf = dataset.hwf

        render_poses = dataset.render_poses
        render_timesteps = dataset.render_timesteps

        print('DEFINING BOUNDS')
        near = dataset.bound_min.min()
        far = dataset.bound_max.max()
        print('NEAR FAR', near, far)


    elif args.dataset_type == 'blender':

        if args.optical_flow:
            images, poses, render_poses, render_timesteps, hwf, i_split, timesteps, keypoints, keypoints_timestep, keypoints_pose, depths = load_blender_data(args.datadir, args, args.half_res, args.testskip)
            keypoints = np.array(keypoints)
            keypoints_timestep = np.array(keypoints_timestep)
            keypoints_pose = np.array(keypoints_pose)
        elif args.scene_flow or args.velocity:
            images, poses, render_poses, render_timesteps, hwf, i_split, timesteps, locations, locations_timestep, bounds, depths = load_blender_data(args.datadir, args, args.half_res, args.testskip)
            locations = np.array(locations)
            locations_timestep = np.array(locations_timestep)
        else:
            images, poses, render_poses, render_timesteps, hwf, i_split, timesteps, depths = load_blender_data(args.datadir, args, args.half_res, args.testskip)
        # print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        if args.pouring:
            near = 0.
            far = 20.
        else:
            near = 0.
            far = 8.
        args.white_bkgd = False

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'gibson':
        if args.optical_flow:
            images, poses, render_poses, render_timesteps, hwf, i_split, timesteps, keypoints, keypoints_timestep, keypoints_pose = load_gibson_data(args.datadir, args, args.half_res, args.testskip)
            keypoints = np.array(keypoints)
            keypoints_timestep = np.array(keypoints_timestep)
            keypoints_pose = np.array(keypoints_pose)
        elif args.scene_flow or args.velocity:
            images, poses, render_poses, render_timesteps, hwf, i_split, timesteps, locations, locations_timestep, bounds = load_gibson_data(args.datadir, args, args.half_res, args.testskip)
            locations = np.array(locations)
            locations_timestep = np.array(locations_timestep)
        else:
            images, poses, render_poses, render_timesteps, hwf, i_split, timesteps, = load_gibson_data(args.datadir, args, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
#         i_train, i_val, i_test = i_split
        i_train = i_split[0]

        near = 0.0
        far = 8.
        args.white_bkgd = False

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # print(poses)


    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if args.render_test:
        render_poses = np.array(poses[i_test])
        render_timesteps = np.array(timesteps[i_test])


    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())


    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    if args.velocity:
        model_fine = render_kwargs_train['network_fine']
        velocity_module = model_fine.velocity_module
        rtol = 0.001
        atol = 0.0001
        # rtol = 1e-3
        # atol = 1e-4
        # rtol = 1e-4
        # atol = 1e-5
        ode_solver = "dopri5"
        # odeint = odeint_adjoint

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            # render_test switches to test poses
            images = images
            poses = torch.Tensor(poses).to(device)

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _, _, _ = render_path(poses, timesteps, hwf, args.chunk / 2, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        # if args.dataset_type in ["dynamic"]:
        #     rays = np.stack([get_rays_np(H, W, focal[i], p) for i, p in enumerate(poses[:,:3,:4])], 0) # [N, ro+rd, H, W, 3]
        # else:
        rays = np.stack([get_rays_np(H, W, focal, p) for i, p in enumerate(poses[:,:3,:4])], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')

        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only

        if args.camera_depth:
            depths = np.stack(depths, 0)
            depths = np.reshape(depths, [-1, 1])

        s = rays_rgb.shape
        timesteps_tile = timesteps[i_train]
        timesteps_tile = np.tile(timesteps_tile[:, None, None, None, None], (1, s[1], s[2], s[3], 1))

        rays_rgb = np.concatenate([rays_rgb, timesteps_tile], axis=4)
        rays_rgb = np.reshape(rays_rgb, [-1,3,4]) # [(N-1)*H*W, ro+rd+rgb, 3]

        if args.scene_flow or args.velocity:
            loc_before, loc_after = locations[:, 0], locations[:, 1]
            loc_t_before, loc_t_after = locations_timestep[:, None, :1], locations_timestep[:, None, 1:]
            bounds = np.array(bounds)



        # Generate constraints for timestep flow
        rays_rgb = rays_rgb.astype(np.float32)
        rix = np.random.permutation(rays_rgb.shape[0])
        rays_rgb = rays_rgb[rix]

        if args.camera_depth:
            depths = depths[rix]
            depths = torch.Tensor(depths).to(device)
        # print('shuffle rays')
        # np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0
        ray_batch = 0
    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb)
        if args.optical_flow:
            rays_before = torch.Tensor(rays_before).to(device)
            rays_after = torch.Tensor(rays_after).to(device)
        elif args.scene_flow or args.velocity:
            loc_before, loc_after = torch.Tensor(loc_before), torch.Tensor(loc_after)
            loc_t_before, loc_t_after = torch.Tensor(loc_t_before).to(device), torch.Tensor(loc_t_after).to(device)
            if len(bounds.shape) == 4:
                bounds = torch.Tensor(bounds[:, 0, :])
            else:
                bounds = torch.Tensor(bounds)

            bounds = bounds.cuda()
            loc_before = loc_before.cuda()
            loc_after = loc_after.cuda()
            loc_t_before = loc_t_before.repeat(1, bounds.size(1), 1)
            loc_t_after = loc_t_after.repeat(1, bounds.size(1), 1)
            loc_t_before = loc_t_before.cuda()
            loc_t_after = loc_t_after.cuda()

    N_iters = 1000000
    print('Begin')
    print('TRAIN views are', i_train)
    #print('TEST views are', i_test)
    #print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    i = global_step

    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand].to(device) # [B, 2+1, 3*?]

            if args.camera_depth:
                camera_depth = depths[i_batch:i_batch+N_rand].to(device)
            #import pdb
            #pdb.set_trace()
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2,:,:3]
            i_batch += N_rand
#             ray_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]

                if args.camera_depth:
                    depths = depths[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            pose = poses[img_i, :3,:4]
            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
                coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)


        rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)
        if args.velocity:
            ##########################
            # Code for integrating scene flow equal to distance

            diff, t_step, ix, reverse = generate_data(loc_t_before, loc_t_after)

            # Old code assuming ground truth object coordinates
            # loc_offset = (torch.rand(loc_before.size(1), 100, loc_before.size(2)).to(bounds.device) - 0.5) * bounds[ix, :, None]
            # loc_before_i = loc_before[ix, :, None] + loc_offset
            # loc_after_i = loc_after[ix, :, None] + loc_offset
            # t_batch = loc_t_before[ix, None, :, :].repeat(loc_offset.size(1), 1, 1)

            loc_offset = (torch.rand(loc_before.size(1), loc_before.size(2)).to(bounds.device) - 0.5) * bounds[ix]
            loc_before_i = loc_before[ix, :] + loc_offset
            loc_after_i = loc_after[ix, :] + loc_offset
            t_batch = loc_t_before[ix]

            # import pdb
            # pdb.set_trace()
            # loc_offset = (torch.rand(loc_before.size(0), loc_before.size(1)).to(bounds.device) - 0.5) * bounds[ix]
            # loc_before_i = loc_before + loc_offset
            # loc_after_i = loc_after + loc_offset
            # t_batch = loc_t_before

            loc_before_i = loc_before_i.view(-1, loc_before_i.size(-1))
            loc_after_i = loc_after_i.view(-1, loc_after_i.size(-1))

            loc_before_i = loc_before_i.to(bounds.device)
            loc_after_i = loc_after_i.to(bounds.device)
            t_step = t_step.to(bounds.device)
            t_batch = t_batch.view(-1, t_batch.size(-1))
            # t_batch = t_batch.view(-1, 1)
            # t_batch = t_batch.view(-1)
            f_options = {'T_batch': t_batch}
            ode_solution = odeint_adjoint(velocity_module, loc_before_i, t_step, method=ode_solver, rtol=rtol, atol=atol, options={}, f_options=f_options)

            # Enforce correspondance across timesteps
            scene_loss = 10.0 * torch.norm(ode_solution[1:2] - loc_after_i, p=2, dim=2).mean()

            # Generate rays to cast consistency over
            network_query_fn_pt = render_kwargs_train['network_query_fn_pt']
            raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw) * dists)

            rand_idx = torch.randperm(rays_rgb.shape[0])
            n = 12
            rays_rgb_select = rays_rgb[rand_idx[:n]].to(device)
            rays_batch = torch.transpose(rays_rgb_select, 0, 1)
            batch_rays, _ = rays_batch[:2], rays_batch[2, :, :3]

            rays_o = rays_batch[0, :, :3]
            rays_d = rays_batch[1, :, :3]

            t_random = rays_batch[0, :, 3]
            t_random = (torch.rand_like(t_random) - 0.5) * 2
            rays_o = torch.cat([rays_o, t_random[:, None]], dim=-1)

            t_vals = torch.linspace(0., 1., steps=args.N_samples).to(rays_d.device)
            z_vals = near * (1.-t_vals) + far * (t_vals)
            model = render_kwargs_train['network_fn']
            model_fine = render_kwargs_train['network_fine']

            z_vals = z_vals.expand([n, args.N_samples])

            # Perturb z_vals
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            t_rand = torch.rand(z_vals.shape).to(rays_d.device)
            z_vals = lower + (upper - lower) * t_rand

            dists = z_vals[...,1:] - z_vals[...,:-1]
            dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(dists.device)], -1)
            dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
            pts = torch.cat([rays_o[:, None, :3] + rays_d[:, None, :] * z_vals[..., :, None], t_random[:, None, None].repeat(1, args.N_samples, 1)], dim=-1)

            raw = network_query_fn_pt(pts, model)
            alpha = raw2alpha(raw[...,3], dists)  # [N_rays, N_samples]

            weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(dists.device), 1.-alpha + 1e-10], -1), -1)[:, :-1]

            z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], args.N_samples, det=False)

            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

            dists = z_vals[...,1:] - z_vals[...,:-1]
            dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(dists.device)], -1)  # [N_rays, N_samples]

            dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
            pts = torch.cat([rays_o[:, None, :3] + rays_d[:, None, :] * z_vals[..., :, None], t_random[:, None, None].repeat(1, dists.size(1), 1)], dim=-1)

            raw = network_query_fn_pt(pts, model_fine)
            raws = raw[..., 3]
            alpha = raw2alpha(raw[...,3], dists)  # [N_rays, N_samples]
            rays = pts.size(1)
            pts_flat = pts.view(-1, 4)

            weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(dists.device), 1.-alpha + 1e-10], -1), -1)[:, :-1]

            t_eval = torch.Tensor([0, np.random.uniform(0, 0.4)])

            if random.uniform(0, 1) > 0.5:
                reverse = True
            else:
                reverse = False

            pos_random = pts_flat[:, :3]
            t_random = pts_flat[:, -1:]
            f_options = {'T_batch': t_random, 'reverse': reverse}
            pos_corr = odeint_adjoint(velocity_module, pos_random, t_eval, method=ode_solver, rtol=rtol, atol=atol, options={}, f_options=f_options)

            pos_corr = pos_corr[1]

            if reverse:
                t_random_next = t_random - t_eval[1].detach()
            else:
                t_random_next = t_random + t_eval[1].detach()

            inp_pts = torch.cat([pos_random, t_random], dim=1)
            inp_pts_corr = torch.cat([pos_corr, t_random_next], dim=1)


            network_query_fn_pt = render_kwargs_train['network_query_fn_pt']
            model = render_kwargs_train['network_fn']
            # model = render_kwargs_train['network_fn_copy']
            model_fine = render_kwargs_train['network_fine']
            # model_fine = render_kwargs_train['network_fine_copy']

            # inp_embed = network_query_fa_pt(inp_pts, model_fine)
            inp_corr_embed = network_query_fn_pt(inp_pts_corr, model_fine)
            inp_embed = network_query_fn_pt(inp_pts, model_fine)

            inp_corr_embed_coarse = network_query_fn_pt(inp_pts_corr, model)
            inp_embed_coarse = network_query_fn_pt(inp_pts, model)

            if args.grad_penalty:
                inp_pts_corr_detach = inp_pts_corr.detach()
                inp_pts_corr_detach.requires_grad = True
                vel_pred = velocity_module.forward_velocity(inp_pts_corr_detach)
                vel_norm = torch.norm(vel_pred, p=2, dim=-1).sum()
                pts_grad = torch.autograd.grad([vel_norm], [inp_pts_corr_detach], create_graph=True)[0]

                grad_pen = torch.norm(pts_grad, p=2, dim=-1).mean()
                scene_loss = scene_loss + 1.0 * grad_pen
            else:
                grad_pen = torch.zeros(1)

            rgb_embed = torch.sigmoid(inp_embed[:, :3])
            rgb_embed_corr = torch.sigmoid(inp_corr_embed[:, :3])

            rgb_embed_coarse = torch.sigmoid(inp_embed_coarse[:, :3])
            rgb_embed_corr_coarse = torch.sigmoid(inp_corr_embed_coarse[:, :3])

            raw_corr = inp_corr_embed[:, 3].view(-1, rays)
            raw_corr_coarse = inp_corr_embed_coarse[:, 3].view(-1, rays)
            raw  = inp_embed[:, 3].view(-1, rays)
            raw_coarse  = inp_embed_coarse[:, 3].view(-1, rays)

            if args.rgb_loss:
                rgb_embed_corr = rgb_embed_corr.view(n, -1, 3)
                rgb_embed = rgb_embed.view(n, -1, 3)
#                 import pdb
#                 pdb.set_trace()
                if args.bkg_no_rgb_loss:
                    bkg_mask = torch.norm(pos_corr - pos_random, dim= -1) > 0.001 * abs(t_eval[1])
                    rgb_embed1 = torch.sigmoid(inp_embed[:, :3])[bkg_mask]
                    rgb_embed_corr1 = torch.sigmoid(inp_corr_embed[:, :3])[bkg_mask]

                    rgb_embed_coarse1 = torch.sigmoid(inp_embed_coarse[:, :3])[bkg_mask]
                    rgb_embed_corr_coarse1 = torch.sigmoid(inp_corr_embed_coarse[:, :3])[bkg_mask]
                    if np.prod(rgb_embed1.size()) != 0:
                        rgb_decode_loss = (torch.abs(rgb_embed_corr1 - rgb_embed1).mean(dim=-1)).mean()
                        rgb_decode_loss = (torch.abs(rgb_embed_corr_coarse1 - rgb_embed_coarse1).mean(dim=-1)).mean() + rgb_decode_loss
                    else:
                        rgb_decode_loss = torch.zeros(1).to(scene_loss.device)

                # rgb_decode_loss = (torch.abs(rgb_embed_corr - rgb_embed).mean(dim=-1) * weights.detach()).mean()
                else:
                    rgb_decode_loss = (torch.abs(rgb_embed_corr - rgb_embed).mean(dim=-1)).mean()
                    rgb_decode_loss = (torch.abs(rgb_embed_corr_coarse - rgb_embed_coarse).mean(dim=-1)).mean() + rgb_decode_loss
                scene_loss = 1e-2 * rgb_decode_loss + scene_loss
            else:
                rgb_decode_loss = torch.zeros(1)

            raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw) * dists)

            raw = raw.view(-1, rays)
            raw_corr = raw_corr.view(-1, rays)
            raw_coarse = raw_coarse.view(-1, rays)
            raw_corr_coarse = raw_corr_coarse.view(-1, rays)

            raw = torch.cat([raw, raw_coarse], dim=0)
            raw_corr = torch.cat([raw_corr, raw_corr_coarse], dim=0)

            if args.depth_loss:
                depth_loss = torch.abs(raw - raw_corr).mean()
                depth_loss = depth_loss + torch.abs(raw_coarse - raw_corr_coarse).mean()

                # dists = z_vals[...,1:] - z_vals[...,:-1]

                # # Repeat for both fine and coarse depth
                # dists = torch.cat([dists, dists], dim=0)
                rays_d_depth = torch.cat([rays_d, rays_d], dim=0)

                scene_loss = 1e-2 * depth_loss + scene_loss
            else:
                depth_loss = torch.zeros(1)

            # Computering losses for suface correspondance
            if args.surface_loss:
                surface_pt = (weights * z_vals).sum(dim=1, keepdim=True)

                surface_pos = rays_o[:n, :3]  + surface_pt * rays_d[:n, :3]
                surface_pt = torch.cat([surface_pos, rays_o[:n, 3:]], dim=-1)
                f_options = {'T_batch': rays_o[:n, 3:], 'reverse': False}

                t_eval = torch.Tensor([0, np.random.uniform(0, 0.1)])
                surface_pos_next = odeint_adjoint(velocity_module, surface_pos, t_eval, method=ode_solver, rtol=rtol, atol=atol, options={}, f_options=f_options)

                offset = surface_pos_next[1] - rays_o[:n, :3]
                offset = offset / torch.norm(offset, p=2, dim=-1, keepdim=True) * torch.norm(rays_d[:n], p=2, dim=-1, keepdim=True)

                t_vals = torch.linspace(0., 1., steps=args.N_samples).to(weights.device)
                z_vals_surface = near * (1.-t_vals) + far * (t_vals)

                z_vals_surface = z_vals_surface.expand([n, args.N_samples])

                dists_surface = z_vals_surface[...,1:] - z_vals_surface[...,:-1]
                dists_surface = torch.cat([dists_surface, torch.Tensor([1e10]).expand(dists_surface[...,:1].shape).to(weights.device)], -1)  # [N_rays, N_samples]

                dists_surface = dists_surface * torch.norm(offset[...,None,:], dim=-1)
                surface_pts = torch.cat([rays_o[:n, None, :3] + offset[:n, None, :] * z_vals_surface[...,:, None], rays_o[:n, None, -1:].repeat(1, z_vals_surface.size(1), 1)], dim=-1)


                raw_surface = network_query_fn_pt(surface_pts, model)
                alpha_surface = raw2alpha(raw_surface[...,3], dists_surface)  # [N_rays, N_samples]

                weights_surface = alpha_surface * torch.cumprod(torch.cat([torch.ones((alpha_surface.shape[0], 1)).to(t_vals.device), 1.-alpha_surface + 1e-10], -1), -1)[:, :-1]


                z_vals_surface_mid = .5 * (z_vals_surface[...,1:] + z_vals_surface[...,:-1])
                z_samples = sample_pdf(z_vals_surface_mid, weights_surface[...,1:-1], args.N_samples, det=False)

                z_samples = z_samples.detach()

                z_vals_surface, _ = torch.sort(torch.cat([z_vals_surface, z_samples], -1), -1)


                dists_surface = z_vals_surface[...,1:] - z_vals_surface[...,:-1]
                dists_surface = torch.cat([dists_surface, torch.Tensor([1e10]).expand(dists_surface[...,:1].shape).to(weights.device)], -1)  # [N_rays, N_samples]

                dists_surface = dists_surface * torch.norm(offset[...,None,:], dim=-1)
                surface_pts = torch.cat([rays_o[:n, None, :3] + offset[:n, None, :] * z_vals_surface[...,:, None], rays_o[:n, None, -1:].repeat(1, z_vals_surface.size(1), 1)], dim=-1)

                raw_surface = network_query_fn_pt(surface_pts, model_fine)
                alpha_surface = raw2alpha(raw_surface[...,3], dists_surface)  # [N_rays, N_samples]

                weights_surface = alpha_surface * torch.cumprod(torch.cat([torch.ones((alpha_surface.shape[0], 1)).to(weights.device), 1.-alpha_surface + 1e-10], -1), -1)[:, :-1]

                surface_pos_cast = (weights_surface * z_vals_surface).sum(dim=1, keepdim=True)

                surface_pos_cast = rays_o[:n, :3] + offset * surface_pos_cast

                surface_loss = 1e-3 * torch.norm(surface_pos_cast - surface_pos_next[1], p=2, dim=-1).mean()

                scene_loss = scene_loss + surface_loss
            else:
                surface_loss = torch.zeros(1)
#             import pdb
#             pdb.set_trace()
            if args.vel_loss:
                dists = z_vals[...,1:] - z_vals[...,:-1]
                dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(dists.device)], -1)  # [N_rays, N_samples]
                dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
                raw = torch.chunk(raw, 2, dim=0)[0]
                alpha = raw2alpha(raw, dists)
                weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha.device), 1.-alpha + 1e-10], -1), -1)[:, :-1]

                vel = velocity_module.forward_velocity(pts).view(-1, 3)
#                 pdb.set_trace()
                weights_cum = torch.cumsum(weights, dim=1)
                weights_mask = (weights_cum.view(-1) < 0.001)
                v_vel = vel[weights_mask]


                if np.prod(v_vel.size()) == 0:
                    vel_loss = torch.zeros(1).to(scene_loss.device)
                else:
                    vel_loss = 1e-3 * torch.norm(v_vel, p=2, dim=-1).mean()
                scene_loss = scene_loss + vel_loss

                bkg_mask = torch.norm(vel, dim=1) < 0.01
                if args.bkg_loss and np.prod(vel[bkg_mask].size()) != 0:
#                     bkg_mask = torch.norm(vel, dim=1) < 0.01
                    bkg_loss = torch.norm(vel[bkg_mask], p=2, dim=-1).mean()
#                     import pdb
#                     pdb.set_trace()
                    scene_loss = scene_loss + bkg_loss
                else:
                    bkg_loss = torch.zeros(1)

                if args.uniform_vel_loss and np.prod(vel[bkg_mask==False].size()) != 0:
                    fore_vel = vel[bkg_mask==False]
                    uniform_vel_loss = torch.norm(fore_vel[1:]-fore_vel[:-1], dim=-1).mean()
                    scene_loss = scene_loss + uniform_vel_loss
                else:
                    uniform_vel_loss = torch.zeros(1)
            else:
#                 import pdb
#                 pdb.set_trace()
                vel_loss = torch.zeros(1)
                bkg_loss = torch.zeros(1)
                uniform_vel_loss = torch.zeros(1)

            scene_loss = scene_loss

        else:
            scene_loss = torch.zeros(1).to(device)
            rgb_decode_loss = torch.zeros(1).to(device)
            depth_loss = torch.zeros(1).to(device)
            vel_loss = torch.zeros(1).to(device)
            surface_loss = torch.zeros(1).to(device)
            grad_pen = torch.zeros(1).to(device)
            bkg_loss = torch.zeros(1)
            uniform_vel_loss = torch.zeros(1)

        # match_loss = img2mse(rgb_before, rgb_after)
        if i % 10 == 0 and args.optical_flow:
            rgb_before, _, _, extras_before = render(H, W, focal, chunk=args.chunk, rays=batch_before,
                                                    verbose=False, retraw=True,
                                                    **render_kwargs_train)
            rgb_after, _, _, extras_after = render(H, W, focal, chunk=args.chunk, rays=batch_after,
                                                    verbose=False, retraw=True,
                                                    **render_kwargs_train)

            match_loss = img2mse(rgb_before, rgb_after)
        else:
            match_loss = torch.zeros(1).to(device)


        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)

        trans = extras['raw'][...,-1]
        # loss = img_loss  + 1e-3 * match_loss
        #INCREASE NERF WEIGHTS
        loss = img_loss + match_loss + scene_loss

        def normalize(x):
            x = x.squeeze()
            x = (x - x.mean()) / x.std()
            return x

        if args.camera_depth:
            depth_batch = camera_depth
            zero_mask = (depth_batch == 0).squeeze()

            depth_map_coarse = extras["depth_map_coarse"]
            depth_map = extras['depth_map']

            # Filter based off the zero_mask
            depth_map_coarse = depth_map_coarse[~zero_mask]
            depth_map = depth_map[~zero_mask]
            depth_batch = depth_batch[~zero_mask]

            # Invert the depth_map
            depth_map =  depth_map
            depth_map_coarse = depth_map_coarse
            depth_batch = depth_batch

            depth_batch = normalize(depth_batch)
            depth_map = normalize(depth_map)
            depth_map_coarse = normalize(depth_map_coarse)
            camera_depth = 1e-2 * ((torch.pow(depth_batch - depth_map, 2))).mean() + 1e-2 * ((torch.pow(depth_batch - depth_map_coarse, 2))).mean()
            loss = loss + camera_depth
        else:
            camera_depth = torch.zeros(1)

        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            #INCREASE NERF WEIGHTS
            img_loss0 = img2mse(extras['rgb0'], target_s)
            psnr0 = mse2psnr(img_loss0)
        else:
            img_loss0 = 0

        # L1 penalty to penalize predictions
        #INCREASE NERF WEIGHTS
        res_loss = torch.abs(extras['rgb_res']).mean()

        # if args.velocity:
        loss = loss # + 1e-2 * res_loss
        # else:
        #     pass

        loss = loss + img_loss0

        loss.backward()
        optimizer.step()
        model_copy(render_kwargs_train['network_fn'], render_kwargs_train['network_fn_copy'])
        model_copy(render_kwargs_train['network_fine'], render_kwargs_train['network_fine_copy'])

        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        warming_up = 1000
        if (
            global_step < warming_up
        ):  # in case images are very dark or very bright, need to keep network from initially building up so much momentum that it kills the gradient
            new_lrate /= 20.0 * (-(global_step - warming_up) / warming_up) + 1.0
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lrate

            ################################

        dt = time.time()-time0
#         import pdb
#         pdb.set_trace()
#             print(rgb_decode_loss)
#             print(depth_loss)
#             print(vel_loss)
#             print(surface_loss)
#             print(grad_pen)
#             print(camera_depth)
        print("Step: {}, Loss: {}, Match Loss: {}, Img Loss: {} Scene Loss: {}, Img Loss0: {}, Time: {}, Res Loss: {}, Rgb Loss: {}, Depth Loss: {}, Velocity Loss: {}, Surface Loss: {}, Grad Pen: {}, Camera Depth: {}, Bkg vel Loss: {}, Uniform_Vel_Loss: {}".format(global_step, loss.item(), match_loss.item(), img_loss, scene_loss, img_loss0, dt, res_loss, rgb_decode_loss.item(), depth_loss.item(), vel_loss.item(), surface_loss.item(), grad_pen.item(), camera_depth.item(), bkg_loss.item(), uniform_vel_loss.item()))
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            if render_kwargs_train['network_fine'] is not None:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
                print('Saved checkpoints at', path)
            else:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
                print('Saved checkpoints at', path)

        # if i%args.i_video==0 and i > 0:
        # if i%args.i_video==0:
        if True:
            # Turn on testing mode

            rgbs = []
            if args.dataset_type == "blender" or args.dataset_type == "gibson":
                for counter in range(render_timesteps.shape[0]):

                    if args.ood_render:
                        path = os.path.join(args.datadir, 'ood_render', 'r_{}.png'.format(counter))
                    elif args.camera_render:
                        path = os.path.join(args.datadir, 'render_camera_linear', 'r_{}.png'.format(counter))
                    elif args.camera_render_after:
                        path = os.path.join(args.datadir, 'render_camera_after', 'r_{}.png'.format(counter))
                    elif args.rotate_render:
                        path = os.path.join(args.datadir, 'render_rotate', 'r_{}.png'.format(counter))
                    else:
                        path = os.path.join(args.datadir, 'render_linear', 'r_{}.png'.format(counter))

                    im = imread(path)[:, :,:3] / 255.
                    H, W = im.shape[:2]


                    if args.half_res:
                        H = H//2
                        W = W//2

                    if args.render_res:
                        im = resize(im, (2*H, 2*W))
                    else:
                        im = resize(im, (H, W))

                    rgbs.append(im)

                rgbs_gt = rgbs

                with torch.no_grad():
                    # images_select = images[-20:].detach().cpu().numpy()
                    rgbs, disps, accs, mses = render_path(render_poses, render_timesteps, hwf, args.chunk, render_kwargs_test, velocity=args.velocity)


                moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
                imageio.mimwrite(moviebase + 'rgb_linear.mp4', to8b(rgbs), fps=10, quality=8)
                imageio.mimwrite(moviebase + 'disp_linear.mp4', to8b(disps), fps=10, quality=8)

                if args.velocity:
                    imageio.mimwrite(moviebase + 'vel_linear.mp4', to8b(accs), fps=10, quality=8)

        global_step += 1
        i += 1


if __name__=='__main__':
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()

