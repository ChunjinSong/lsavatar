import os
import hydra
import cv2
import numpy as np
import torch
from lib.utils import utils
import h5py

class Dataset(torch.utils.data.Dataset):
    def __init__(self, metainfo, split):

        if split.type == 'Video' or split.type == 'VideoVal'  or split.type == 'VideoTest' or split.type == 'Canonical':
            path = os.path.join("../data", metainfo.data_dir)
        elif split.type == 'VideoNovelPose':
            path = os.path.join("../data", metainfo.data_dir.replace('training', 'novel_pose'))
        elif split.type == 'VideoNovelView':
            path = os.path.join("../data", metainfo.data_dir.replace('training', 'novel_view'))

        if split.type == 'Canonical':
            self.use_3d_box = False
        else:
            self.use_3d_box = True

        self.dataset_path = hydra.utils.to_absolute_path(path)
        self.dataset_h5py = h5py.File(self.dataset_path, 'r')
        self.n_images = self.dataset_h5py['img_shape'][0]
        self.img_shape = self.dataset_h5py['img_shape'][1:]

        self.bgcolor = metainfo.bgcolor
        self.img_scale_factor = metainfo.img_scale_factor
        self.img_size = tuple(np.array(self.img_shape[:2] * self.img_scale_factor).astype(np.int))
        self.start_frame = metainfo.start_frame
        self.end_frame = self.n_images
        self.skip_step = metainfo.skip
        self.training_indices = list(range(self.start_frame, self.end_frame, self.skip_step))
        self.frames_name = self.dataset_h5py['frames_name'][self.training_indices]
        self.shape = self.dataset_h5py['mean_shape'][:]
        self.poses = self.dataset_h5py['poses'][self.training_indices]
        self.trans = self.dataset_h5py['normalize_trans'][self.training_indices]

        # cameras
        scale_mats = self.dataset_h5py['scales'][self.training_indices].astype(np.float32)
        world_mats = self.dataset_h5py['cameras'][self.training_indices].astype(np.float32)

        self.scale = 1 / scale_mats[0][0, 0]

        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = utils.load_K_Rt_from_P(None, P)
            intrinsics[:2] *= self.img_scale_factor
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())
        assert len(self.intrinsics_all) == len(self.pose_all)

        # other properties
        self.num_sample = split.num_sample
        if split.type == 'Video':
            self.sampling_strategy = split.sampling_strategy
            self.patch_size = split.patch_size
            self.sample_subject_ratio = split.sample_subject_ratio
            self.sample_bbox_ratio = split.sample_bbox_ratio
            self.N_patch = split.N_patch

        skeleton = self.dataset_h5py['joints'][self.training_indices].astype(np.float32)
        min_xyz = np.min(skeleton, axis=1) - metainfo.bbox_offset
        max_xyz = np.max(skeleton, axis=1) + metainfo.bbox_offset
        self.bboxs_min = min_xyz * self.scale
        self.bboxs_max = max_xyz * self.scale

    def __len__(self):
        return len(self.training_indices)

    def init_dataset(self):

        if self.dataset_h5py is not None:
            return
        print('init dataset')

        self.dataset_h5py = h5py.File(self.dataset_path, 'r')

    def load_image(self, idx):
        bgcolor = (np.ones(3) * self.bgcolor).astype('float32')

        self.init_dataset()

        idx = self.start_frame + idx * self.skip_step
        orig_img = self.dataset_h5py['images'][idx].reshape(self.img_shape).astype('float32')
        alpha_mask = self.dataset_h5py['masks'][idx].reshape(self.img_shape[0], self.img_shape[1]).astype('float32')
        img = alpha_mask[..., None] * orig_img + (1.0 - alpha_mask[..., None]) * bgcolor[None, None, :] * 255.

        if self.img_scale_factor != 1.:
            img = cv2.resize(img, None,
                             fx=self.img_scale_factor,
                             fy=self.img_scale_factor,
                             interpolation=cv2.INTER_LANCZOS4)
            alpha_mask = cv2.resize(alpha_mask, None,
                                    fx=self.img_scale_factor,
                                    fy=self.img_scale_factor,
                                    interpolation=cv2.INTER_LINEAR)

        img = (img / 255.).astype('float32')
        img = np.clip(img, 0., 1.)

        return img, alpha_mask, bgcolor

    def weighted_sampling(self, data):
        data_sampling = {}
        data_sampling['rgb'] = data['ray_img']
        data_sampling['cam_loc'] = data['cam_loc']
        data_sampling['ray_dirs'] = data['ray_dirs']
        data_sampling['near'] = data['near']
        data_sampling['far'] = data['far']

        samples, index_inside, index_outside = utils.weighted_sampling(data_sampling,
                                                                       data['object_mask'],
                                                                       data['ray_mask'],
                                                                       self.num_sample,
                                                                       self.sample_bbox_ratio)

        inputs = {
            "cam_loc": samples['cam_loc'],
            "ray_dirs": samples['ray_dirs'],
            "near": samples['near'],
            "far": samples['far'],
            "smpl_scale": data['smpl_params'][0],
            "smpl_pose": data['smpl_params'][4:76],
            "smpl_shape": data['smpl_params'][76:],
            "smpl_trans": data['smpl_params'][1:4],
            'bgcolor': data['bgcolor'],
            "idx": self.start_frame + data['idx'] * self.skip_step
        }
        images = {"rgb": samples["rgb"],
                  'index_inside': index_inside,
                  'index_outside': index_outside,
                  }
        return inputs, images

    def patch_sampling(self, data):
        data_patch = {}
        data_patch['rgb'] = data['ray_img']
        data_patch['cam_loc'] = data['cam_loc']
        data_patch['ray_dirs'] = data['ray_dirs']
        data_patch['near'] = data['near']
        data_patch['far'] = data['far']

        samples, patch_info, index_inside, index_outside = utils.patch_sampling(data_patch,
                                                                                data['rgb'],
                                                                                data['object_mask'],
                                                                                data['ray_mask'],
                                                                                self.img_size,
                                                                                self.patch_size,
                                                                                self.N_patch,
                                                                                self.sample_subject_ratio)

        images = {"rgb": samples["rgb"],
                  'patch_div_indices': patch_info['patch_div_indices'],
                  'patch_masks': patch_info['mask'],
                  'target_patches': patch_info['target_patches'],
                  'bgcolor': data['bgcolor'],
                  'index_inside': index_inside,
                  'index_outside': index_outside,
                  }

        inputs = {
            "cam_loc": samples["cam_loc"],
            "ray_dirs": samples["ray_dirs"],
            "near": samples["near"],
            "far": samples["far"],
            "smpl_scale": data['smpl_params'][0],
            "smpl_pose": data['smpl_params'][4:76],
            "smpl_shape": data['smpl_params'][76:],
            "smpl_trans": data['smpl_params'][1:4],
            'bgcolor': data['bgcolor'],
            "idx": self.start_frame + data['idx'] * self.skip_step
        }
        return inputs, images

    def __getitem__(self, idx):
        # normalize RGB
        img, mask, bgcolor = self.load_image(idx)

        uv = np.mgrid[:self.img_size[0], :self.img_size[1]].astype(np.int32)
        uv = np.flip(uv, axis=0).copy().transpose(1, 2, 0).astype(np.float32)

        smpl_params = torch.zeros([86]).float()
        smpl_params[0] = torch.from_numpy(np.asarray(self.scale)).float()

        smpl_params[1:4] = torch.from_numpy(self.trans[idx]).float()
        smpl_params[4:76] = torch.from_numpy(self.poses[idx]).float()
        smpl_params[76:] = torch.from_numpy(self.shape).float()

        ray_dirs, cam_loc = utils.get_camera_params(torch.tensor(uv.reshape(1, -1, 2).astype(np.float32)),
                                                    self.pose_all[idx].reshape(1, 4, 4),
                                                    self.intrinsics_all[idx].reshape(1, 4, 4))

        cam_loc = cam_loc.reshape(1, 1, 3).expand(-1, ray_dirs.shape[1], -1)
        cam_loc = cam_loc.numpy().reshape(-1, 3).astype(np.float32)
        ray_dirs = ray_dirs.numpy().reshape(-1, 3).astype(np.float32)

        if self.use_3d_box:
            near, far, ray_mask = utils.rays_intersect_3d_bbox(self.bboxs_min[idx],
                                                           self.bboxs_max[idx],
                                                           cam_loc, ray_dirs)
        else:
            near, far, ray_mask = utils.rays_intersect_3d_bbox([-1,-1,-1],
                                                           [1,1,1],
                                                           cam_loc, ray_dirs)

        ray_img = img.reshape(-1, 3)
        cam_loc = cam_loc[ray_mask]
        ray_dirs = ray_dirs[ray_mask]
        ray_img = ray_img[ray_mask]
        near = near[:, None].astype('float32')
        far = far[:, None].astype('float32')

        if self.num_sample > 0:
            data = {
                "rgb": img,
                "ray_img": ray_img,
                "cam_loc": cam_loc,
                "ray_dirs": ray_dirs,
                "near": near,
                "far": far,
                "ray_mask": ray_mask,
                "object_mask": mask,
                "bgcolor": bgcolor,
                "smpl_params": smpl_params,
                "idx": idx,
            }

            if self.sampling_strategy == 'weighted_sampling':
                inputs, target_pts = self.weighted_sampling(data)
                images = {
                    "rgb_pts": target_pts["rgb"],
                    'bgcolor': data['bgcolor'],
                    'index_inside_pts': target_pts['index_inside'],
                    'index_outside_pts': target_pts['index_outside'],
                    "num_pts": self.num_sample,
                    "num_patch": 0
                }

            elif self.sampling_strategy == 'patch_sampling':
                inputs, target_patch = self.patch_sampling(data)

                images = {"rgb_patch": target_patch["rgb"],
                          'patch_div_indices': target_patch['patch_div_indices'],
                          'patch_masks': target_patch['patch_masks'],
                          'target_patches': target_patch['target_patches'],
                          'bgcolor': data['bgcolor'],
                          'index_inside_patch': target_patch['index_inside'],
                          'index_outside_patch': target_patch['index_outside'],
                          "num_pts": 0,
                          "num_patch": len(target_patch["rgb"])
                          }

            else:
                input_patch, target_patch = self.patch_sampling(data)
                input_pts, target_pts = self.weighted_sampling(data)

                inputs = {
                    "cam_loc": np.concatenate((input_patch['cam_loc'], input_pts['cam_loc']), axis=0),
                    "ray_dirs": np.concatenate((input_patch['ray_dirs'], input_pts['ray_dirs']), axis=0),
                    "near": np.concatenate((input_patch['near'], input_pts['near']), axis=0),
                    "far": np.concatenate((input_patch['far'], input_pts['far']), axis=0),
                    "smpl_scale": data['smpl_params'][0],
                    "smpl_pose": data['smpl_params'][4:76],
                    "smpl_shape": data['smpl_params'][76:],
                    "smpl_trans": data['smpl_params'][1:4],
                    'bgcolor': data['bgcolor'],
                    "idx": self.start_frame + data['idx'] * self.skip_step
                }

                images = {"rgb_patch": target_patch["rgb"],
                          "rgb_pts": target_pts["rgb"],
                          'patch_div_indices': target_patch['patch_div_indices'],
                          'patch_masks': target_patch['patch_masks'],
                          'target_patches': target_patch['target_patches'],
                          'bgcolor': data['bgcolor'],
                          'index_inside_patch': target_patch['index_inside'],
                          'index_outside_patch': target_patch['index_outside'],
                          'index_inside_pts': target_pts['index_inside'],
                          'index_outside_pts': target_pts['index_outside'],
                          "num_pts": self.num_sample,
                          "num_patch": len(target_patch["rgb"])
                          }

        else:
            inputs = {
                "cam_loc": cam_loc,
                "ray_dirs": ray_dirs,
                "near": near,
                "far": far,
                "ray_mask": ray_mask,
                "smpl_scale": smpl_params[0],
                "smpl_pose": smpl_params[4:76],
                "smpl_shape": smpl_params[76:],
                "smpl_trans": smpl_params[1:4],
                'bgcolor': bgcolor,
                "idx": self.start_frame + idx * self.skip_step
            }
            images = {
                "rgb": img.reshape(-1, 3).astype(np.float32),
                "ray_mask": ray_mask,
                "img_size": self.img_size
            }

        return inputs, images


class ValDataset(torch.utils.data.Dataset):
    def __init__(self, metainfo, split):
        self.dataset = Dataset(metainfo, split)
        self.pixel_per_batch = split.pixel_per_batch

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image_id = int(np.random.choice(len(self.dataset), 1))
        self.data = self.dataset[image_id]
        inputs, images = self.data
        total_pixels = inputs["cam_loc"].shape[0]

        inputs = {
            "cam_loc": inputs["cam_loc"],
            "ray_dirs": inputs['ray_dirs'],
            "near": inputs['near'],
            "far": inputs['far'],
            "smpl_scale": inputs["smpl_scale"],
            "smpl_pose": inputs["smpl_pose"],
            "smpl_shape": inputs["smpl_shape"],
            "smpl_trans": inputs["smpl_trans"],
            'bgcolor': inputs["bgcolor"],
            'idx_pose': image_id,
            "idx": inputs['idx']
        }
        images = {
            "rgb_gt": images["rgb"],
            'bgcolor': inputs["bgcolor"],
            "ray_mask": images["ray_mask"],
            "img_size": images["img_size"],
            'pixel_per_batch': self.pixel_per_batch,
            'total_pixels': total_pixels
        }
        return inputs, images


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, metainfo, split):
        self.dataset = Dataset(metainfo, split)
        self.pixel_per_batch = split.pixel_per_batch
        data_dir = metainfo.data_dir

        self.n_cam = 1
        self.frame_skip = 1

        if split.type == 'VideoTest':
            pass
        elif split.type == 'VideoNovelPose':
            split.use_refined_pose = False
        elif split.type == 'VideoNovelView':
            if 'mocap' in data_dir:
                if '313' in data_dir:
                    self.n_cam = 20
                else:
                    self.n_cam = 22
                self.frame_skip = 30

        self.use_refined_pose = split.use_refined_pose

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        shape = self.dataset.shape
        inputs, images = data
        total_pixels = inputs["cam_loc"].shape[0]

        if self.use_refined_pose:
            idx_pose = int(inputs['idx'] / self.n_cam) * self.frame_skip
        else:
            idx_pose = -1

        inputs = {
            "cam_loc": inputs["cam_loc"],
            "ray_dirs": inputs['ray_dirs'],
            "near": inputs['near'],
            "far": inputs['far'],
            "ray_mask": images["ray_mask"],
            "smpl_scale": inputs["smpl_scale"],
            "smpl_pose": inputs["smpl_pose"],
            "smpl_shape": inputs["smpl_shape"],
            "smpl_trans": inputs["smpl_trans"],
            'bgcolor': inputs["bgcolor"],
            "idx": inputs['idx'],
            "idx_pose": idx_pose,
            'shape': shape
        }
        images = {
            "rgb_gt": images["rgb"],
            'bgcolor': inputs["bgcolor"],
            "ray_mask": images["ray_mask"],
            "img_size": images["img_size"],
            'pixel_per_batch': self.pixel_per_batch,
            'total_pixels': total_pixels,
            'n_cam': self.n_cam
        }
        return inputs, images

class CanonicalDataset(torch.utils.data.Dataset):
    def __init__(self, metainfo, split):
        self.dataset = Dataset(metainfo, split)
        self.pixel_per_batch = split.pixel_per_batch

        data_dir = metainfo.data_dir

        self.n_cam = 1
        self.frame_skip = 1

        if split.type == 'VideoTest':
            pass
        elif split.type == 'VideoNovelPose':
            split.use_refined_pose = False
        elif split.type == 'VideoNovelView':
            if 'mocap' in data_dir:
                if '313' in data_dir:
                    self.n_cam = 20
                else:
                    self.n_cam = 22
                self.frame_skip = 30

        self.use_refined_pose = split.use_refined_pose

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        data = self.dataset[idx]
        shape = self.dataset.shape
        inputs, images = data
        total_pixels = inputs["cam_loc"].shape[0]

        if self.use_refined_pose:
            idx_pose = int(inputs['idx'] / self.n_cam) * self.frame_skip
        else:
            idx_pose = -1

        inputs = {
            "cam_loc": inputs["cam_loc"],
            "ray_dirs": inputs['ray_dirs'],
            "near": inputs['near'],
            "far": inputs['far'],
            "ray_mask": images["ray_mask"],
            "smpl_scale": inputs["smpl_scale"],
            "smpl_pose": inputs["smpl_pose"],
            "smpl_shape": inputs["smpl_shape"],
            "smpl_trans": inputs["smpl_trans"],
            'bgcolor': inputs["bgcolor"],
            "idx": inputs['idx'],
            "idx_pose": idx_pose,
            'shape': shape
        }
        images = {
            "rgb_gt": images["rgb"],
            'bgcolor': inputs["bgcolor"],
            "ray_mask": images["ray_mask"],
            "img_size": images["img_size"],
            'pixel_per_batch': self.pixel_per_batch,
            'total_pixels': total_pixels,
            'n_cam': self.n_cam
        }
        return inputs, images
