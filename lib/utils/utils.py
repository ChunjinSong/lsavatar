import numpy as np
import cv2
import torch
from torch.nn import functional as F


def split_input(model_input, total_pixels, device, n_pixels = 10000):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of n_pixels in case of cuda out of memory error.
     '''

    split = []

    for i, indx in enumerate(torch.split(torch.arange(total_pixels, device=device), n_pixels, dim=0)):
        data = model_input.copy()
        data['cam_loc'] = torch.index_select(model_input['cam_loc'], 1, indx)
        data['ray_dirs'] = torch.index_select(model_input['ray_dirs'], 1, indx)
        data['near'] = torch.index_select(model_input['near'], 1, indx)
        data['far'] = torch.index_select(model_input['far'], 1, indx)
        split.append(data)
    return split


def merge_output(res, total_pixels, batch_size):
    ''' Merge the split output. '''

    model_outputs = {}
    for entry in res[0]:
        if res[0][entry] is None:
            continue
        if len(res[0][entry].shape) == 1:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, 1) for r in res],
                                             1).reshape(batch_size * total_pixels)
        else:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, r[entry].shape[-1]) for r in res],
                                             1).reshape(batch_size * total_pixels, -1)
    return model_outputs


def get_psnr(img1, img2, normalize_rgb=False):
    if normalize_rgb: # [-1,1] --> [0,1]
        img1 = (img1 + 1.) / 2.
        img2 = (img2 + 1. ) / 2.

    mse = torch.mean((img1 - img2) ** 2)
    psnr = -10. * torch.log(mse) / torch.log(torch.Tensor([10.], devlce=img1.device))

    return psnr


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose


def get_camera_params(uv, pose, intrinsics):
    if pose.shape[1] == 7: #In case of quaternion vector representation
        cam_loc = pose[:, 4:]
        R = quat_to_rot(pose[:,:4])
        p = torch.eye(4).repeat(pose.shape[0],1,1, device=uv.device).float()
        p[:, :3, :3] = R
        p[:, :3, 3] = cam_loc
    else: # In case of pose matrix representation
        cam_loc = pose[:, :3, 3]
        p = pose

    batch_size, num_samples, _ = uv.shape

    depth = torch.ones((batch_size, num_samples), device=uv.device)
    x_cam = uv[:, :, 0].view(batch_size, -1)
    y_cam = uv[:, :, 1].view(batch_size, -1)
    z_cam = depth.view(batch_size, -1)

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics)

    # permute for batch matrix product
    pixel_points_cam = pixel_points_cam.permute(0, 2, 1)

    world_coords = torch.bmm(p, pixel_points_cam).permute(0, 2, 1)[:, :, :3]
    ray_dirs = world_coords - cam_loc[:, None, :]
    ray_dirs = F.normalize(ray_dirs, dim=2)

    return ray_dirs, cam_loc

def lift(x, y, z, intrinsics):
    # parse intrinsics
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    sk = intrinsics[:, 0, 1]

    x_lift = (x - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z
    y_lift = (y - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z

    # homogeneous
    return torch.stack((x_lift, y_lift, z, torch.ones_like(z, device=intrinsics.device)), dim=-1)


def quat_to_rot(q):
    batch_size, _ = q.shape
    q = F.normalize(q, dim=1)
    R = torch.ones((batch_size, 3,3), device=q.device)
    qr=q[:,0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]
    R[:, 0, 0]=1-2 * (qj**2 + qk**2)
    R[:, 0, 1] = 2 * (qj *qi -qk*qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1-2 * (qi**2 + qk**2)
    R[:, 1, 2] = 2*(qj*qk - qi*qr)
    R[:, 2, 0] = 2 * (qk * qi-qj * qr)
    R[:, 2, 1] = 2 * (qj*qk + qi*qr)
    R[:, 2, 2] = 1-2 * (qi**2 + qj**2)
    return R


def rot_to_quat(R):
    batch_size, _,_ = R.shape
    q = torch.ones((batch_size, 4), device=R.device)

    R00 = R[:, 0,0]
    R01 = R[:, 0, 1]
    R02 = R[:, 0, 2]
    R10 = R[:, 1, 0]
    R11 = R[:, 1, 1]
    R12 = R[:, 1, 2]
    R20 = R[:, 2, 0]
    R21 = R[:, 2, 1]
    R22 = R[:, 2, 2]

    q[:,0]=torch.sqrt(1.0+R00+R11+R22)/2
    q[:, 1]=(R21-R12)/(4*q[:,0])
    q[:, 2] = (R02 - R20) / (4 * q[:, 0])
    q[:, 3] = (R10 - R01) / (4 * q[:, 0])
    return q


def get_sphere_intersections(cam_loc, ray_directions, r = 1.0):
    # Input: n_rays x 3 ; n_rays x 3
    # Output: n_rays x 1, n_rays x 1 (close and far)

    ray_cam_dot = torch.bmm(ray_directions.view(-1, 1, 3),
                            cam_loc.view(-1, 3, 1)).squeeze(-1)
    under_sqrt = ray_cam_dot ** 2 - (cam_loc.norm(2, 1, keepdim=True) ** 2 - r ** 2)

    # sanity check
    if (under_sqrt <= 0).sum() > 0:
        print('BOUNDING SPHERE PROBLEM!')
        exit()

    sphere_intersections = torch.sqrt(under_sqrt) * torch.Tensor([-1, 1], device=ray_directions.device).float() - ray_cam_dot
    sphere_intersections = sphere_intersections.clamp_min(0.0)

    return sphere_intersections

def bilinear_interpolation(xs, ys, dist_map):
    x1 = np.floor(xs).astype(np.int32)
    y1 = np.floor(ys).astype(np.int32)
    x2 = x1 + 1
    y2 = y1 + 1

    dx = np.expand_dims(np.stack([x2 - xs, xs - x1], axis=1), axis=1)
    dy = np.expand_dims(np.stack([y2 - ys, ys - y1], axis=1), axis=2)
    Q = np.stack([
        dist_map[x1, y1], dist_map[x1, y2], dist_map[x2, y1], dist_map[x2, y2]
    ], axis=1).reshape(-1, 2, 2)
    return np.squeeze(dx @ Q @ dy)  # ((x2 - x1) * (y2 - y1)) = 1

def get_index_outside_of_bbox(samples_uniform, bbox_min, bbox_max):
    samples_uniform_row = samples_uniform[:, 0]
    samples_uniform_col = samples_uniform[:, 1]
    index_outside = np.where((samples_uniform_row < bbox_min[0]) | (samples_uniform_row > bbox_max[0]) | (samples_uniform_col < bbox_min[1]) | (samples_uniform_col > bbox_max[1]))[0]
    return index_outside

def _process_mask(mask, border: int = 1, ignore_value: int = 128):
    mask = mask.astype(np.uint8)
    kernel = np.ones((border, border), np.uint8)
    mask_erode = cv2.erode(mask.copy(), kernel)
    mask_dilate = cv2.dilate(mask.copy(), kernel)
    mask[mask_dilate != mask_erode] = ignore_value
    return mask

def get_index_in_and_outside_of_mask(indices, mask):
    mask_samp = _process_mask(mask)
    mask_inds = mask_samp.reshape(-1)[indices]
    index_inside = np.where(mask_inds == 1)[0]
    index_outside = np.where(mask_inds == 0)[0]
    return index_inside, index_outside

def weighted_sampling(data, obj_mask, ray_mask, num_sample, bbox_ratio=0.9):
    """
    More sampling within the bounding box
    """
    ray_mask = ray_mask.reshape(obj_mask.shape)
    # calculate bounding box
    where = np.asarray(np.where(obj_mask))
    bbox_min = where.min(axis=1)
    bbox_max = where.max(axis=1)
    bbox_mask = np.zeros_like(obj_mask)
    bbox_mask[bbox_min[0]:bbox_max[0], bbox_min[1]:bbox_max[1]] = 1

    inds_bbox = np.where(bbox_mask.reshape(-1))[0]
    inds_uniform = np.where(ray_mask.reshape(-1))[0]

    num_sample_bbox = int(num_sample * bbox_ratio)
    samples_bbox = np.random.choice(inds_bbox, num_sample_bbox, replace=False)

    num_sample_uniform = num_sample - num_sample_bbox
    samples_uniform = np.random.choice(inds_uniform, num_sample_uniform, replace=False)

    select_masked_inds = np.concatenate([samples_bbox, samples_uniform], axis=0)
    # indices = np.floor(indices).astype(np.int32)

    masked_indices = np.cumsum(ray_mask) - 1
    select_inds = masked_indices[select_masked_inds]

    output = {}
    for key, val in data.items():
        dim_val = val.shape[-1]
        val = val.reshape(-1, dim_val)
        output[key] = val[select_inds]
    index_inside, index_outside = get_index_in_and_outside_of_mask(select_masked_inds, obj_mask)

    return output, index_inside, index_outside

# from humannerf without change
def get_patch_ray_indices(ray_mask, candidate_mask, patch_size, H, W):
    assert len(ray_mask.shape) == 1
    assert ray_mask.dtype == np.bool
    assert candidate_mask.dtype == np.bool

    valid_ys, valid_xs = np.where(candidate_mask)

    # determine patch center
    select_idx = np.random.choice(valid_ys.shape[0],
                                  size=[1], replace=False)[0]
    center_x = valid_xs[select_idx]
    center_y = valid_ys[select_idx]

    # determine patch boundary
    half_patch_size = patch_size // 2
    x_min = np.clip(a=center_x - half_patch_size,
                    a_min=0,
                    a_max=W - patch_size)
    x_max = x_min + patch_size
    y_min = np.clip(a=center_y - half_patch_size,
                    a_min=0,
                    a_max=H - patch_size)
    y_max = y_min + patch_size

    sel_ray_mask = np.zeros_like(candidate_mask)
    sel_ray_mask[y_min:y_max, x_min:x_max] = True

    #####################################################
    ## Below we determine the selected ray indices
    ## and patch valid mask

    sel_ray_mask = sel_ray_mask.reshape(-1)
    inter_mask = np.bitwise_and(sel_ray_mask, ray_mask)

    select_masked_inds = np.where(inter_mask)[0]

    masked_indices = np.cumsum(ray_mask) - 1
    select_inds = masked_indices[select_masked_inds]

    inter_mask = inter_mask.reshape(H, W)

    return select_masked_inds, select_inds, \
           inter_mask[y_min:y_max, x_min:x_max], \
           np.array([x_min, y_min]), np.array([x_max, y_max])


def patch_sampling(data, full_img, subject_mask, ray_mask, img_size, patch_size, N_patch, sample_subject_ratio=0.8):
    # calculate bounding box
    subject_mask = subject_mask.astype(bool)
    ray_mask = ray_mask.astype(bool)
    bbox_mask = ray_mask.reshape(img_size)
    bbox_exclude_subject_mask = np.bitwise_and(
            bbox_mask,
            np.bitwise_not(subject_mask)
        )

    H, W = img_size

    list_select_masked_inds = []
    list_ray_indices = []
    list_mask = []
    list_xy_min = []
    list_xy_max = []

    total_rays = 0
    patch_div_indices = [total_rays]
    for i in range(N_patch):
        # let p = cfg.patch.sample_subject_ratio
        # prob p: we sample on subject area
        # prob (1-p): we sample on non-subject area but still in bbox
        if np.random.rand(1)[0] < sample_subject_ratio:
            candidate_mask = subject_mask
        else:
            candidate_mask = bbox_exclude_subject_mask

        select_masked_inds, ray_indices, mask, xy_min, xy_max = get_patch_ray_indices(ray_mask, candidate_mask,
                                        patch_size, H, W)

        assert len(ray_indices.shape) == 1
        total_rays += len(ray_indices)

        list_select_masked_inds.append(select_masked_inds)
        list_ray_indices.append(ray_indices)
        list_mask.append(mask)
        list_xy_min.append(xy_min)
        list_xy_max.append(xy_max)

        patch_div_indices.append(total_rays)

    select_inds_img = np.concatenate(list_select_masked_inds, axis=0)
    select_inds = np.concatenate(list_ray_indices, axis=0)
    patch_info = {
        'mask': np.stack(list_mask, axis=0),
        'xy_min': np.stack(list_xy_min, axis=0),
        'xy_max': np.stack(list_xy_max, axis=0)
    }
    patch_div_indices = np.array(patch_div_indices)

    output = {}
    for key, val in data.items():
        dim_val = val.shape[-1]
        output[key] = val.reshape(-1, dim_val)[select_inds]

    index_inside, index_outside = get_index_in_and_outside_of_mask(select_inds_img, subject_mask)

    targets = []
    # img_targets = np.zeros((img_size[0], img_size[1], 3))
    for i in range(N_patch):
        x_min, y_min = patch_info['xy_min'][i]
        x_max, y_max = patch_info['xy_max'][i]
        targets.append(full_img[y_min:y_max, x_min:x_max])
        # img_targets[y_min:y_max, x_min:x_max] = full_img[y_min:y_max, x_min:x_max]

    target_patches = np.stack(targets, axis=0)  # (N_patche, P, P, 3)
    patch_info['target_patches'] = target_patches
    patch_info['patch_div_indices'] = patch_div_indices

    return output, patch_info, index_inside, index_outside

def rays_intersect_3d_bbox(bounds_min, bounds_max, ray_o, ray_d):
    r"""calculate intersections with 3d bounding box
        Args:
            - bounds: dictionary or list
            - ray_o: (N_rays, 3)
            - ray_d, (N_rays, 3)
        Output:
            - near: (N_VALID_RAYS, )
            - far: (N_VALID_RAYS, )
            - mask_at_box: (N_RAYS, )
    """

    bounds = np.stack([bounds_min, bounds_max], axis=0)
    assert bounds.shape == (2,3)

    bounds = bounds + np.array([-0.01, 0.01])[:, None]
    nominator = bounds[None] - ray_o[:, None] # (N_rays, 2, 3)
    # calculate the step of intersections at six planes of the 3d bounding box
    ray_d[np.abs(ray_d) < 1e-5] = 1e-5
    d_intersect = (nominator / ray_d[:, None]).reshape(-1, 6) # (N_rays, 6)
    # calculate the six interections
    p_intersect = d_intersect[..., None] * ray_d[:, None] + ray_o[:, None] # (N_rays, 6, 3)
    # calculate the intersections located at the 3d bounding box
    min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
    eps = 1e-6
    p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                    (p_intersect[..., 0] <= (max_x + eps)) * \
                    (p_intersect[..., 1] >= (min_y - eps)) * \
                    (p_intersect[..., 1] <= (max_y + eps)) * \
                    (p_intersect[..., 2] >= (min_z - eps)) * \
                    (p_intersect[..., 2] <= (max_z + eps))  # (N_rays, 6)
    # obtain the intersections of rays which intersect exactly twice
    mask_at_box = p_mask_at_box.sum(-1) == 2  #(N_rays, )
    p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(
        -1, 2, 3) # (N_VALID_rays, 2, 3)

    # calculate the step of intersections
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]
    norm_ray = np.linalg.norm(ray_d, axis=1)
    d0 = np.linalg.norm(p_intervals[:, 0] - ray_o, axis=1) / norm_ray
    d1 = np.linalg.norm(p_intervals[:, 1] - ray_o, axis=1) / norm_ray
    near = np.minimum(d0, d1)
    far = np.maximum(d0, d1)

    return near, far, mask_at_box

def rotate_x(phi):
    cos = np.cos(phi)
    sin = np.sin(phi)
    return np.array([[1,   0,    0, 0],
                     [0, cos, -sin, 0],
                     [0, sin,  cos, 0],
                     [0,   0,    0, 1]], dtype=np.float32)

def rotate_z(psi):
    cos = np.cos(psi)
    sin = np.sin(psi)
    return np.array([[cos, -sin, 0, 0],
                     [sin,  cos, 0, 0],
                     [0,      0, 1, 0],
                     [0,      0, 0, 1]], dtype=np.float32)
def rotate_y(theta):
    cos = np.cos(theta)
    sin = np.sin(theta)
    return np.array([[cos,   0,  sin, 0],
                     [0,     1,    0, 0],
                     [-sin,  0,  cos, 0],
                     [0,   0,      0, 1]], dtype=np.float32)
