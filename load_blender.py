import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2
import random
from imageio import imread, imwrite
from skimage import color
import os.path as osp


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w



def load_blender_data_test(basedir, args, half_res=False, testskip=1, eval_depth=False):
    with open(os.path.join(basedir, 'transforms_{}.json'.format("test")), 'r') as fp:
            meta = json.load(fp)

    imgs = []
    poses = []
    timesteps = []
    counts = [0]
    no_time = False

    # if args.velocity:
    #     assert args.pouring
    # fname = os.path.join(basedir, frame['file_path'] + '.png')
    example_im = imageio.imread(osp.join(basedir, meta['frames'][0]['file_path']))
    W = example_im.shape[1]
    H = example_im.shape[0]

    if half_res:
        example_im = cv2.resize(example_im, (W // 2, H // 2), interpolation=cv2.INTER_AREA)

    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    depth_dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)

    depths = []

    for i, frame in enumerate(meta['frames']):
        if "flow" in frame['file_path']:
            continue

        if len(imgs) > 100:
            break

        # Skip scene flow config
        fname = os.path.join(basedir, frame['file_path'])

        if 'depth_train_path' in frame and eval_depth:
            depth_fname = frame['depth_train_path']
            depth_img = imread(osp.join(basedir, depth_fname), format="hdr")
        else:
            depth_img = np.zeros((200, 200))

        im = imageio.imread(fname)[:, :, :3]

        if half_res:
            im = cv2.resize(im, (200, 200), interpolation=cv2.INTER_AREA)
        else:
            im = cv2.resize(im, (400, 400), interpolation=cv2.INTER_AREA)

        im = im / 255.
        imgs.append(im)
        poses.append(np.array(frame['transform_matrix']))

        timesteps.append((frame['timestep'] / 449 - 0.5) * 2.)

        depths.append(depth_img)

        # Randomly only sample 100 timesteps
        # if s == 'train':
        #     random_idxs = list(range(len(imgs)))
        #     random.shuffle(random_idxs)
        #     random_idxs = random_idxs[:100]
        #     imgs = [rix] for rix in random_idxs]
        #     poses = [poses[rix] for rix in random_idxs]

    # imgs = np.array(imgs, dtype=np.uint8).astype(np.float32) / 255. # keep all 4 channels (RGBA)
    poses = np.array(poses).astype(np.float32)
    timesteps = np.array(timesteps)

    H, W, C = imgs[0].shape
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    hwf = [H, W, focal]

    return imgs, depths, poses, timesteps, hwf



def load_blender_data(basedir, args, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    all_timesteps = []
    counts = [0]
    no_time = False

    if args.optical_flow:
        keypoints = []
        keypoints_timesteps = []
        keypoints_pose = []

        orb_detector = cv2.ORB_create(5000)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    elif args.scene_flow or args.velocity:
        locations = []
        locations_timesteps = []
        bounds = []

    # if args.velocity:
    #     assert args.pouring
    # fname = os.path.join(basedir, frame['file_path'] + '.png')
    example_im = imageio.imread(osp.join(basedir, metas['train']['frames'][0]['file_path']))
    W = example_im.shape[1]
    H = example_im.shape[0]
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    camera_angle_x = float(metas['train']['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    depth_dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)

    depths = []

    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip

        if s == 'train':
            if args.optical_flow or args.scene_flow or args.velocity:
                chunk = 2
            else:
                chunk = 2

            for i in range(0, len(meta['frames']), chunk):
                print(i, len(meta['frames']))

                if s != 'train' and i > 100:
                    break

                if args.debug and i > 20:
                    break

                # if i // chunk > args.frames:
                #     break

                if args.optical_flow:
                    frame = meta['frames'][i]
                    frame_next = meta['frames'][i+1]

                    if args.pouring:
                        fname = os.path.join(basedir, frame['file_path'])
                        fname_next = os.path.join(basedir, frame_next['file_path'])
                    else:
                        fname = os.path.join(basedir, frame['file_path'] + '.png')
                        fname_next = os.path.join(basedir, frame_next['file_path'] + '.png')

                    img = imageio.imread(fname)
                    img_next = imageio.imread(fname_next)

                    kp1, d1 = orb_detector.detectAndCompute(img, None)
                    kp2, d2 = orb_detector.detectAndCompute(img_next, None)

                    matches = matcher.match(d1, d2)
                    matches.sort(key = lambda x: x.distance)
                    keypoint_i = []

                    for i in range(100):
                        try:
                            keypoint_i.append([*kp1[matches[i].queryIdx].pt, *kp2[matches[i].trainIdx].pt])
                        except:
                            keypoint_i.append([*kp1[matches[0].queryIdx].pt, *kp2[matches[0].trainIdx].pt])

                        # if i == 0:
                        #     x1, y1 = kp1[matches[i].queryIdx].pt
                        #     x2, y2 = kp2[matches[i].queryIdx].pt
                        #     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        #     img[y1-5:y1+5, x1-5:x1+5] = 0.0
                        #     img_next[y2-5:y2+5, x2-5:x2+5] = 0.0

                    # imageio.imwrite("img_prev.png", img)
                    # imageio.imrite("img_next.png", img_next)
                    # assert False

                    keypoints_timesteps.append([frame['timestep'], frame_next['timestep']])
                    keypoints_pose.append(frame['transform_matrix'])

                    keypoints.append(np.array(keypoint_i))
                elif args.scene_flow:
                    frame = meta['frames'][i]
                    frame_next = meta['frames'][i+1]

                    frame_offset = frame['offset']
                    frame_offset_next = frame_next['offset']

                    dim = frame['dimensions']
                    dim_next = frame_next['dimensions']

                    locations.append([frame_offset, frame_offset_next])
                    locations_timesteps.append([frame['timestep'], frame_next['timestep']])
                    bounds.append([dim])
                elif args.velocity:
                    frame = meta['frames'][i]
                    frame_next = meta['frames'][i+1]
                    fname = os.path.join(basedir, frame['file_path'])
                    fname_next = os.path.join(basedir, frame_next['file_path'])

                    img = (color.rgb2gray(imageio.imread(fname)) * 255).astype(np.uint8)
                    img_next = (color.rgb2gray(imageio.imread(fname_next)) * 255).astype(np.uint8)

                    depth_fname = os.path.join(basedir, frame['depth_train_path'])
                    next_depth = frame_next['depth_train_path']
                    next_depth = next_depth[:-8] + "{:04}".format(frame_next['timestep']) + ".hdr"
                    depth_fname_next = os.path.join(basedir, next_depth)

                    depth_img = imread(depth_fname, format="hdr")
                    depth_img_next = imread(depth_fname_next, format="hdr")

                    depth_img = depth_img * depth_dirs
                    depth_img_next = depth_img_next * depth_dirs
                    depth_img = np.concatenate([depth_img, np.ones((depth_img.shape[0], depth_img.shape[1], 1))], axis=-1)
                    depth_img_next = np.concatenate([depth_img_next, np.ones((depth_img_next.shape[0], depth_img_next.shape[1], 1))], axis=-1)

                    feature_params = dict( maxCorners = 1000,
                                           qualityLevel = 0.15,
                                           minDistance = 7,
                                           blockSize = 7 )


                    lk_params = dict( winSize  = (14,14),
                                      maxLevel = 2,
                                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                    # p0 = cv2.goodFeaturesToTrack(img, mask = None, **feature_params)

                    # if p0 is None:
                    #     continue

                    # p1, st, err = cv2.calcOpticalFlowPyrLK(img, img_next, p0, None, **lk_params)
                    # flow = cv2.calcOpticalFlowFarneback(img, img_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    fp = frame['file_path']
                    fp = fp.split("/")[1].split(".")[0]
                    frame_num = int(fp[2:])

                    flow_fname = os.path.join(basedir, osp.split(frame['file_path'])[0], "flow_{}.npy".format(frame_num))
                    flow = np.load(flow_fname)
                    f = 8

                    x, y = np.meshgrid(np.arange(flow.shape[0]), np.arange(flow.shape[1]))
                    coord = np.stack([y, x], axis=2)

                    flow = flow[::f, ::f]
                    coord = coord[::f, ::f]
                    mag_diff = np.linalg.norm(flow, ord=2, axis=-1)


                    mag_diff = mag_diff.reshape(-1)
                    flow = flow.reshape((-1, 2))
                    coord = coord.reshape((-1, 2))

                    idx = np.argsort(mag_diff)

                    top = 200
                    max_idx = idx[-top:]
                    # random_idx = np.random.permutation(flow.shape[0])[:500]
                    # max_idx = np.concatenate([max_idx, random_idx], axis=0)

                    flow_max = flow[max_idx]
                    coord_max = coord[max_idx]

                    # good_new = p1[st==1]
                    # good_old = p0[st==1]

                    bounds_i = []
                    frame_offset = []
                    frame_offset_next = []

                    def clip(x):
                        return min(max(x, 0), img.shape[0] - 1)

                    # p0, p1 = p0[:, 0, :], p1[:, 0, :]
                    # print(p0.shape)

                    while True:
                        # for (x1, y1), (x2, y2) in zip(p0, p1):
                        for ix in range(coord_max.shape[0]):
                            x1, y1 = coord_max[ix, 1], coord_max[ix, 0]
                            flow_x, flow_y = flow_max[ix, 0], flow_max[ix, 1]
                            x2, y2 = clip(round(x1 + flow_x)), clip(round(y1 + flow_y))

                            x1, y1, x2, y2 = clip(int(x1)), clip(int(y1)), clip(int(x2)), clip(int(y2))

                            diff_x = abs(x1 - x2)
                            diff_y = abs(y2 - y1)

                            # if diff_x > 0 and diff_y > 0:
                            tf = np.array(frame['transform_matrix'])
                            # import pdb
                            # pdb.set_trace()
                            # print("depth orig ", depth_img[y1, x1])
                            # print("depth new ", depth_img_next[y1, x1])
                            # print("frame orig ", frame["offset"], frame['dimensions'])
                            # print("frame new ", frame_next["offset"], frame_next['dimensions'])
                            # print("set", x1, y1, x2, y2)

                            # if diff_x > 0 and diff_y > 0:
                            output_frame = (tf @ depth_img[y1, x1,:,  None])
                            output_frame_next = (tf @ depth_img_next[y2, x2,:,  None])
                            output_frame = output_frame[:3, 0]
                            output_frame_next = output_frame_next[:3, 0]

                            frame_offset.append(output_frame)
                            frame_offset_next.append(output_frame_next)

                            # print("print dimensions ", frame['offset'])
                            # print("print next dimensions ", frame_next['offset'])
                            # print("predict offset ", frame_offset[-1])
                            # print("predict next offset ", frame_offset_next[-1])
                            bounds_i.append([0.05, 0.05, 0.05])

                            # size = 5
                            # img[clip(y1-size):clip(y1+size), clip(x1-size):clip(x1+size)] = 255.0
                            # img_next[clip(y2-size):clip(y2+size), clip(x2-size):clip(x2+size)] = 255.0
                            # print("set", x1, y1, x2, y2)

                            if len(bounds_i) == 500:
                                # assert False
                                break

                        if len(bounds_i) == 500:
                            # imwrite("{}_im_before.png".format(i), img)
                            # imwrite("{}_im_after.png".format(i), img_next)
                            # assert False
                            break

                        if len(bounds_i) == 0:
                            break

                if (args.velocity) and (len(bounds_i) != 0):
                    locations.append([frame_offset, frame_offset_next])
                    if args.pouring:
                        locations_timesteps.append([(frame['timestep'] / 449 - 0.5) * 2.0, (frame_next['timestep'] / 449 - 0.5) * 2.0])
                    else:
                        locations_timesteps.append([frame['timestep'], frame_next['timestep']])
                    bounds.append(bounds_i)

                frame = meta['frames'][i]

                if args.pouring:
                    fname = os.path.join(basedir, frame['file_path'])
                else:
                    fname = os.path.join(basedir, frame['file_path'] + '.png')

                if s == "train" and args.camera_depth:
                    depth_fname = os.path.join(basedir, frame['depth_train_path'])
                    depth_img = imread(depth_fname, format="hdr")
                    depth_img = depth_img * depth_dirs
                    depth_shape = depth_img.shape
                    depth_img = np.concatenate([depth_img, np.ones((depth_img.shape[0], depth_img.shape[1], 1))], axis=-1)
                    tf = np.array(frame['transform_matrix'])
                    depth_img = depth_img.reshape((-1, 4))
                    depth_parse = tf @ depth_img.transpose((1, 0))
                    depth_parse = depth_parse.transpose((1, 0))[:, :3].reshape(depth_shape)
                    depth_parse = np.linalg.norm(depth_parse, ord=2, axis=-1)
                    # imwrite("test_depth.png", 1 / (1 + depth_parse))
                    depths.append(depth_parse)

                imgs.append(imageio.imread(fname))
                poses.append(np.array(frame['transform_matrix']))

                if 'timestep' in frame.keys():
                    if args.pouring:
                        all_timesteps.append((frame['timestep'] / 449 - 0.5) * 2)
                    else:
                        all_timesteps.append(frame['timestep'])
                else:
                    all_timesteps.append(1)
                    no_time = True
        else:
            for i, frame in enumerate(meta['frames'][::skip]):
                # Skip scene flow config
                if 'flow' in frame['file_path']:
                    continue

                if args.pouring:
                    fname = os.path.join(basedir, frame['file_path'])
                else:
                    fname = os.path.join(basedir, frame['file_path'] + '.png')
                imgs.append(imageio.imread(fname))
                poses.append(np.array(frame['transform_matrix']))

                if 'timestep' in frame.keys():
                    all_timesteps.append(frame['timestep'])
                else:
                    all_timesteps.append(1)
                    no_time = True

                print(i)
                if (not args.render_test) and (i > 30):
                    break

        # Randomly only sample 100 timesteps
        # if s == 'train':
        #     random_idxs = list(range(len(imgs)))
        #     random.shuffle(random_idxs)
        #     random_idxs = random_idxs[:100]
        #     imgs = [rix] for rix in random_idxs]
        #     poses = [poses[rix] for rix in random_idxs]

        print("Loading ims")
        imgs = np.array(imgs, dtype=np.uint8).astype(np.float32) / 255. # keep all 4 channels (RGBA)
        print("Loading poses")
        poses = np.array(poses).astype(np.float32)
        print("Finished both")
        counts.append(counts[-1] + imgs.shape[0])

        all_imgs.append(imgs)
        all_poses.append(poses)

        print("loading data")

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    print("finished concatenation")

    H, W, C = imgs[0].shape
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)


    if args.rotate_render:
        render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[::-1]], 0)

        if no_time:
            render_timesteps = np.ones(render_poses.shape[0])
        else:
            render_timesteps = np.linspace(-1, 1, render_poses.shape[0])
    elif args.camera_render:
        angle = 0.1 / np.pi * 180
        render_poses = torch.stack([pose_spherical(-angle, -angle, 4.0) for _ in np.linspace(-180,180,40+1)[::-1]], 0)

        if no_time:
            render_timesteps = np.ones(render_poses.shape[0])
        else:
            render_timesteps = np.linspace(-1, 1, render_poses.shape[0])
    elif args.camera_render_after:
        angle_up = 0.8 / np.pi * 180
        angle_rotate = 0.2 / np.pi * 180
        render_poses = torch.stack([pose_spherical(-angle_rotate, -angle_up, 4.0) for _ in np.linspace(-180,180,40+1)[::-1]], 0)

        if no_time:
            render_timesteps = np.ones(render_poses.shape[0])
        else:
            render_timesteps = np.linspace(-1, 1, render_poses.shape[0])
    elif args.ood_render:
        render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.ones(41)*-180], 0)
        render_timesteps = np.linspace(1, 1.5, render_poses.shape[0])
    else:
        render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.ones(41)*-180], 0)

        if no_time:
            render_timesteps = np.ones(render_poses.shape[0])
        else:
            render_timesteps = np.linspace(-1, 1, render_poses.shape[0])

        with open(os.path.join(basedir, 'transforms_render.json'), 'r') as fp:
            render_data = json.load(fp)

        render_poses = [f['transform_matrix'] for f in render_data['frames']]
        render_poses = np.array(render_poses)

        if args.pouring:
            render_timesteps = np.array([(f['timestep'] / 449 - 0.5) * 2. for f in render_data['frames']])
        else:
            render_timesteps = np.array([f['timestep'] for f in render_data['frames']])

        # render_poses = render_poses[:30]
        # render_timesteps = render_timesteps[:30]

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, C))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    all_timesteps = np.array(all_timesteps)
    # all_timesteps =  2 * (all_timesteps - all_timesteps.min()) / (all_timesteps.max() - all_timesteps.min() + 1e-5) - 1


    if args.optical_flow:
        return imgs, poses, render_poses, render_timesteps, [H, W, focal], i_split, all_timesteps, keypoints, keypoints_timesteps, keypoints_pose, depths
    elif args.scene_flow or args.velocity:
        return imgs, poses, render_poses, render_timesteps, [H, W, focal], i_split, all_timesteps, locations, locations_timesteps, bounds, depths
    else:
        return imgs, poses, render_poses, render_timesteps, [H, W, focal], i_split, all_timesteps, depths


if __name__ == "__main__":
    imgs, poses, render_poses, render_timesteps, info, splits, timesteps = load_blender_data("/data/vision/billf/scratch/yilundu/nerf-pytorch/data/nerf_synthetic/table_nerf_time/")
    import pdb
    pdb.set_trace()
    print("here!")


