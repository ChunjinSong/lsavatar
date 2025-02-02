import torch
from torch import nn
from lib.lpips import LPIPS
from lib.S3IM import S3IM

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

class Loss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.base_weight = opt.base_weight
        self.eikonal_weight = opt.eikonal_weight
        self.mask_weight = opt.mask_weight
        self.lpips_weight = opt.lpips_weight
        self.volscale_weight = opt.volscale_weight
        self.kick_in_iter = opt.kick_in_iter
        self.kick_out_iter = opt.kick_out_iter
        self.rgb_weight = opt.rgb_weight
        self.s3im_weight = opt.s3im_weight
        self.delta_x_weight = opt.delta_x_weight
        self.eps = 1e-6
        self.milestone = 200
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.l2_loss = nn.MSELoss(reduction='mean')

        if self.lpips_weight > 0:
            self.lpips = LPIPS(net='vgg')
            set_requires_grad(self.lpips, requires_grad=False)
            # self.lpips = nn.DataParallel(self.lpips)
        if self.s3im_weight > 0:
            self.s3im_func = S3IM(patch_height=opt.s3im_patch, patch_width=opt.s3im_patch,
                             kernel_size=opt.s3im_kernel, stride=opt.s3im_stride,
                             repeat_time=opt.s3im_repeat)
    
    # L1 reconstruction loss for RGB values
    def get_rgb_loss(self, rgb_values, rgb_gt):
        rgb_loss = self.l1_loss(rgb_values, rgb_gt)
        return rgb_loss
    
    # Eikonal loss introduced in IGR
    def get_eikonal_loss(self, grad_theta):
        eikonal_loss = ((grad_theta.norm(2, dim=-1) - 1)**2).mean()
        return eikonal_loss

    # BCE loss for clear boundary
    def get_bce_loss(self, acc_map):
        binary_loss = -1 * (acc_map * (acc_map + self.eps).log() + (1-acc_map) * (1 - acc_map + self.eps).log()).mean() * 2
        return binary_loss

    # Global opacity sparseness regularization 
    def get_opacity_sparse_loss(self, acc_map, index_off_surface):
        opacity_sparse_loss = self.l1_loss(acc_map[index_off_surface], torch.zeros_like(acc_map[index_off_surface]))
        return opacity_sparse_loss

    def get_mask_loss(self, acc_map, index_inside, index_outside):
        mask_loss = 0
        if index_inside.shape[0] > 0:
            mask_loss += self.l1_loss(acc_map[index_inside], torch.ones_like(acc_map[index_inside]))
        if index_outside.shape[0] > 0:
            mask_loss += self.l1_loss(acc_map[index_outside], torch.zeros_like(acc_map[index_outside]))
        return mask_loss

    # Optional: This loss helps to stablize the training in the very beginning
    def get_in_shape_loss(self, acc_map, index_in_surface):
        in_shape_loss = self.l1_loss(acc_map[index_in_surface], torch.ones_like(acc_map[index_in_surface]))
        return in_shape_loss

    def unpack_imgs(self, rgbs, patch_masks, bgcolor, targets, div_indices):
        N_patch = len(div_indices) - 1
        assert patch_masks.shape[0] == N_patch
        assert targets.shape[0] == N_patch

        patch_imgs = bgcolor.expand(targets.shape).clone()  # (N_patch, H, W, 3)
        for i in range(N_patch):
            patch_imgs[i, patch_masks[i]] = rgbs[div_indices[i]:div_indices[i + 1]]

        return patch_imgs

    def scale_for_lpips(self, image_tensor):
        return image_tensor * 2. - 1.

    def forward(self, model_outputs, ground_truth, global_step, use_non_rigid_deformer):
        results = {}
        loss = 0.
        # nan_filter = ~torch.any(model_outputs['rgb_values'].isnan(), dim=1)
        num_patch = ground_truth['num_patch'][0]

        pred_pts = model_outputs['rgb'][num_patch:, ...]
        pred_patch = model_outputs['rgb'][:num_patch, ...]
        acc_map_pts = model_outputs['acc_map'][num_patch:, ...]
        acc_map_patch = model_outputs['acc_map'][:num_patch, ...]

        rgb_gt = None
        if 'rgb_patch' in ground_truth:
            gt_patch = ground_truth['rgb_patch'][0]
            rgb_gt = gt_patch
        if 'rgb_pts' in ground_truth:
            gt_pts = ground_truth['rgb_pts'][0]
            if rgb_gt is None:
                rgb_gt = gt_pts
            else:
                rgb_gt = torch.cat([rgb_gt, gt_pts], dim=0)

        if self.rgb_weight > 0:
            rgb_loss = self.get_rgb_loss(model_outputs['rgb'], rgb_gt)
            loss += rgb_loss * self.rgb_weight
            results['rgb_loss'] = rgb_loss

        if self.rgb_weight > 0 and self.base_weight > 0 and 'rgb_base' in model_outputs:
            if global_step < self.kick_in_iter:
                weight_factor = 1.0
            elif global_step < self.kick_out_iter:
                weight_factor = 1. - (global_step - self.kick_in_iter) / (self.kick_out_iter - self.kick_in_iter)
            else:
                weight_factor = 0.0

            if weight_factor > 0.0:
                rgb_base_loss = self.get_rgb_loss(model_outputs['rgb_base'], rgb_gt)
                loss += rgb_base_loss * self.rgb_weight * self.base_weight * weight_factor
                results['rgb_base_loss'] = rgb_base_loss

        if self.eikonal_weight > 0:
            eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
            loss += eikonal_loss * self.eikonal_weight
            results['eikonal_loss'] = eikonal_loss

        # if self.eikonal_weight > 0 and self.base_weight > 0 and model_outputs['grad_theta_base'] is not None:
        #     eikonal_base_loss = self.get_eikonal_loss(model_outputs['grad_theta_base'])
        #     loss += eikonal_base_loss * self.eikonal_weight * self.base_weight
        #     results['eikonal_base_loss'] = eikonal_base_loss

        if self.mask_weight > 0:
            mask_loss = 0
            if 'index_inside_pts' in ground_truth:
                mask_loss_pts = self.get_mask_loss(acc_map_pts, ground_truth['index_inside_pts'][0], ground_truth['index_outside_pts'][0])
                mask_loss = mask_loss_pts
            if 'index_inside_patch' in ground_truth:
                mask_loss_patch = self.get_mask_loss(acc_map_patch, ground_truth['index_inside_patch'][0], ground_truth['index_outside_patch'][0])
                if mask_loss == 0:
                    mask_loss = mask_loss_patch
                else:
                    mask_loss = (mask_loss + mask_loss_patch) * 0.5

            loss += mask_loss * self.mask_weight
            results['mask_loss'] = mask_loss

        if use_non_rigid_deformer and self.lpips_weight > 0 and global_step > 30000:
            rgb_patch = self.unpack_imgs(pred_patch,
                             ground_truth['patch_masks'][0],
                             model_outputs['bgcolor'][0], # [0,1]
                             ground_truth['target_patches'][0],
                             ground_truth['patch_div_indices'][0])

            pred = self.scale_for_lpips(rgb_patch.permute(0, 3, 1, 2))
            gt = self.scale_for_lpips(ground_truth['target_patches'][0].permute(0, 3, 1, 2))

            lpips_loss = self.lpips(pred, gt)
            lpips_loss = torch.mean(lpips_loss)

            loss += lpips_loss * self.lpips_weight
            results['lpips_loss'] = lpips_loss

        if self.s3im_weight > 0:
            s3im_loss = self.s3im_func(pred_pts, gt_pts)
            loss += s3im_loss * self.s3im_weight
            results['s3im_loss'] = s3im_loss

        if self.delta_x_weight > 0 and 'delta_x' in model_outputs:
            delta_x = model_outputs['delta_x']
            delta_x_loss = ((delta_x.norm(2, dim=-1)) ** 2).mean()
            loss += delta_x_loss * self.delta_x_weight
            results['delta_x_loss'] = delta_x_loss

        if use_non_rigid_deformer and self.volscale_weight > 0 and 'vol_scale' in model_outputs:
            vol_scale = model_outputs['vol_scale']
            scale_loss = (torch.prod(vol_scale, dim=-1) * model_outputs['valid_joints']).sum()
            loss += scale_loss * self.volscale_weight
            results['scale_loss'] = scale_loss

        results['loss'] = loss

        return results