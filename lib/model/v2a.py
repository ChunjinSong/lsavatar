
import torch
import torch.nn as nn
from torch.autograd import grad
from lib.model.utils.rotation_format import axisang_to_rot6d
from lib.model.module.density import LaplaceDensity
from lib.model.sampler.ray_sampler import UniformSampler
from lib.smpl.deformer import SMPLDeformer
from lib.smpl.smpl import SMPLServer
from lib.model.utils.skeleton_utils import create_axis_boxes, boxes_to_pose, SMPLSkeleton

from lib.model import \
    load_deformer, \
    load_sdf_net, \
    load_rgb_net



class V2A(nn.Module):
    def __init__(self, opt, betas, gender):
        super().__init__()
        self.opt = opt
        self.use_non_rigid_feat = True if self.opt.non_rigid_deformer.d_deformed_feature > 0 else False
        self.use_non_rigid_deformer = True
        self._smpl_deformer = SMPLDeformer(betas=betas[None], gender=gender)
        self.smpl_server = SMPLServer(betas=betas[None], gender=gender)
        smpl_joints_tpose = self.smpl_server.joints_tpose.reshape(-1, 3).cpu().numpy()

        self.ray_sampler = UniformSampler(**opt.ray_sampler)
        self.density = LaplaceDensity(**opt.density)
        self.non_rigid_deformer = load_deformer(opt.non_rigid_deformer, smpl_joints_tpose)
        self.sdf_net = load_sdf_net(opt.sdf_net)
        self.rgb_net = load_rgb_net(opt.rgb_net)


    def sdf_func_with_non_rigid_deformer(self, pts_o, pts_c, cond, is_gradient=False):
        '''
        :param x: canonical coordinate
        :param cond:
        :return:
            sdf
            feature,
            gradient, [optional]
            sdf_base, [optional]
            feature_base,[optional]
            gradient_base, [optional]
            feature_def [optional]
        '''
        if self.use_non_rigid_deformer:
            outputs = self.non_rigid_deformer.forward(pts_o, pts_c, cond, self.sdf_net, is_gradient=is_gradient, is_training=self.training)
        else:
            outputs = self.sdf_net.forward_sdf(pts_c, cond, is_gradient=is_gradient, is_training=self.training)

            if self.use_non_rigid_feat:
                outputs['feature_def'] = torch.zeros((pts_c.shape[0], self.opt.non_rigid_deformer.d_deformed_feature), device=pts_c.device)

        return outputs

    def sdf_func_with_deformer(self, x, cond, smpl_tfs, smpl_verts, smpl_w2l):
        x_c, outlier_mask = self._smpl_deformer.forward(x, smpl_tfs, return_weights=False, inverse=True, smpl_verts=smpl_verts)
        sdf = self.sdf_func_with_non_rigid_deformer(x_c, cond, smpl_w2l, is_gradient=False)['sdf']
        if not self.training:
            sdf[outlier_mask] = 4.  # set a large SDF value for outlier points
        return sdf

    def smpl_deformer(self, x, smpl_tfs, smpl_verts):
        x_c, _ = self._smpl_deformer.forward(x, smpl_tfs, return_weights=False, inverse=True, smpl_verts=smpl_verts)
        return x_c

    def smpl_deformer_xc2xo(self, pts_c, smpl_tfs):
        pts_o = self._smpl_deformer.forward_skinning(pts_c.unsqueeze(0), None, smpl_tfs).squeeze(0)
        return pts_o

    def get_valid(self, pts_c, pts_o, cond):
        out_xl = self.non_rigid_deformer.w2l_func(pts_c, cond['smpl_w2l_c'])
        valid_idxs = out_xl['valid_pts']
        valid_joints = out_xl['valid_jts']

        if len(valid_idxs) > 0:
            pts_o = pts_o[valid_idxs]
            valid_idxs_xo = self.non_rigid_deformer.w2l_func(pts_o, cond['smpl_w2l'])['valid_pts']
            valid_idxs = valid_idxs[valid_idxs_xo]

        return valid_idxs, valid_joints

    def rendering(self, cond, view,
                  pts_o, pts_c,
                  smpl_tfs,
                  N_samples, z_vals, z_max, input):

        outputs = {'grad_theta': None,
                   'rgb': None,
                   'acc_map': None,
                   'normal': None,
                   'sdf': None,
                   'fg_rgb': None,
                   }

        shape = pts_c.shape

        if self.use_non_rigid_deformer and self.use_valid:
            valid_idxs, valid_joints = self.get_valid(pts_c, pts_o, cond)
        else:
            valid_idxs = torch.arange(len(pts_c))
            valid_joints = torch.zeros(24, device=pts_c.device)

        fg_rgb = torch.zeros(shape, device=pts_c.device)
        normal_values = torch.zeros(shape, device=pts_c.device)
        density_values = torch.zeros(shape[0], 1, device=pts_c.device)

        if self.use_non_rigid_deformer:
            fg_rgb_base = torch.zeros(shape, device=pts_c.device)
            normal_base = torch.zeros(shape, device=pts_c.device)
            density_base = torch.zeros(shape[0], 1, device=pts_c.device)

        if len(valid_idxs) > 0:
            pts_o = pts_o[valid_idxs]
            pts_c = pts_c[valid_idxs]

            outputs_rgb = self.get_rbg_value(pts_o, pts_c,
                                         view,
                                         cond, smpl_tfs,
                                         is_training=self.training)

            fg_rgb[valid_idxs] = outputs_rgb['rgb']
            normal_values[valid_idxs] = outputs_rgb['normal']
            density_values[valid_idxs] = self.density(outputs_rgb['sdf'])

            outputs['grad_theta'] = outputs_rgb['gradient']

            if self.use_non_rigid_deformer:
                fg_rgb_base[valid_idxs] = outputs_rgb['rgb_base']
                normal_base[valid_idxs] = outputs_rgb['normal_base']
                density_base[valid_idxs] = self.density(outputs_rgb['sdf_base'])

                outputs['delta_x'] = outputs_rgb['delta_x']
                outputs['delta_x_avg'] = torch.norm(outputs_rgb['delta_x'], dim=1).mean()
                outputs['delta_max'] = torch.norm(outputs_rgb['delta_x'], dim=1, keepdim=True).max()
                outputs['vol_scale'] = self.non_rigid_deformer.get_axis_scale()
                outputs['valid_joints'] = valid_joints

        fg_rgb = fg_rgb.reshape(-1, N_samples, 3)
        normal_values = normal_values.reshape(-1, N_samples, 3)
        weights, bg_transmittance = self.volume_rendering(z_vals, z_max, density_values)
        fg_rgb_values = torch.sum(weights.unsqueeze(-1) * fg_rgb, 1)
        bg_rgb_values = torch.ones_like(fg_rgb_values, device=fg_rgb_values.device) * input['bgcolor'][0]
        bg_rgb_values = bg_transmittance.unsqueeze(-1) * bg_rgb_values
        rgb_values = fg_rgb_values + bg_rgb_values

        bg_rgb_values_normal = torch.ones_like(fg_rgb_values, device=fg_rgb_values.device)
        bg_rgb_values_normal = bg_transmittance.unsqueeze(-1) * bg_rgb_values_normal
        normal_values = (torch.sum(weights.unsqueeze(-1) * normal_values, 1) + 1) / 2 + bg_rgb_values_normal

        outputs['rgb'] = rgb_values
        outputs['acc_map'] = torch.sum(weights, -1)
        outputs['normal'] = normal_values

        if self.training:
            outputs['bgcolor'] = input['bgcolor']
        else:
            fg_output_rgb = fg_rgb_values + bg_transmittance.unsqueeze(-1) * torch.ones_like(fg_rgb_values, device=fg_rgb_values.device)
            outputs['fg_rgb'] = fg_output_rgb

        if self.use_non_rigid_deformer:
            fg_rgb_base = fg_rgb_base.reshape(-1, N_samples, 3)
            normal_base = normal_base.reshape(-1, N_samples, 3)
            weights_base, bg_transmittance_base = self.volume_rendering(z_vals, z_max, density_base)
            fg_rgb_base = torch.sum(weights_base.unsqueeze(-1) * fg_rgb_base, 1)
            bg_rgb_base = torch.ones_like(fg_rgb_base, device=fg_rgb_base.device) * input['bgcolor'][0]
            bg_rgb_base = bg_transmittance_base.unsqueeze(-1) * bg_rgb_base
            rgb_base = fg_rgb_base + bg_rgb_base

            bg_rgb_values_normal = torch.ones_like(fg_rgb_values, device=fg_rgb_values.device)
            bg_rgb_values_normal = bg_transmittance.unsqueeze(-1) * bg_rgb_values_normal
            normal_base = (torch.sum(weights_base.unsqueeze(-1) * normal_base, 1) + 1) / 2 + bg_rgb_values_normal

            outputs['rgb_base'] = rgb_base
            outputs['acc_map_base'] = torch.sum(weights_base, -1)
            outputs['normal_base'] = normal_base

        return outputs

    def forward(self, input):
        torch.set_grad_enabled(True)
        if input['global_step'] < self.opt.non_rigid_deformer.kick_in_iter:
            self.use_non_rigid_deformer = False
        else:
            self.use_non_rigid_deformer = True

        if input['global_step'] < self.opt.non_rigid_deformer.valid_in_iter:
            self.use_valid = False
        else:
            self.use_valid = True

        cam_loc = input["cam_loc"].reshape(-1, 3)
        ray_dirs = input["ray_dirs"].reshape(-1, 3)
        near = input["near"].reshape(-1, 1)
        far = input["far"].reshape(-1, 1)
        batch_size, num_pixels, _ = input["ray_dirs"].shape

        pose6d = axisang_to_rot6d(input["smpl_pose"].reshape(-1, 24, 3)).reshape(1, -1)

        smpl_output = self.smpl_server(input['smpl_scale'], input["smpl_trans"], input["smpl_pose"], input["smpl_shape"])

        smpl_tfs = smpl_output['smpl_tfs']
        smpl_l2w = smpl_output['smpl_l2w'][0]
        smpl_w2l = torch.linalg.inv(smpl_l2w)

        cond = {}
        cond['smpl_pose'] = pose6d
        cond['global_step'] = input['global_step']
        cond['current_epoch'] = input['current_epoch']
        cond['smpl_w2l_c'] = self.smpl_server.w2l_c
        cond['smpl_w2l'] = smpl_w2l

        z_vals = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, near, far, self)

        z_max = z_vals[:, -1]
        z_vals = z_vals[:, :-1]
        N_samples = z_vals.shape[1]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        pts_o = points.reshape(-1, 3)

        dirs = ray_dirs.unsqueeze(1).repeat(1, N_samples, 1)
        view = -dirs.reshape(-1, 3)
        pts_c = self.smpl_deformer(pts_o, smpl_tfs, smpl_output['smpl_verts'])

        output = self.rendering(cond, view,
                                pts_o, pts_c,
                                smpl_tfs,
                                N_samples, z_vals, z_max, input)

        return output

    def get_rbg_value(self, pts_o, pnts_c, view_dirs, cond, smpl_tfs, is_training=True):

        outputs_fgrad = self.forward_gradient(pts_o, pnts_c, cond, smpl_tfs, create_graph=is_training, retain_graph=is_training)
        rgb_vals = self.rgb_net(outputs_fgrad['xc'], outputs_fgrad['normal'], view_dirs, cond, outputs_fgrad['feature'])

        outputs = {}
        outputs['rgb'] = rgb_vals
        outputs['normal'] = outputs_fgrad['normal']
        outputs['sdf'] = outputs_fgrad['sdf']
        outputs['gradient'] = outputs_fgrad['gradient']

        if 'delta_x' in outputs_fgrad:
            outputs['delta_x'] = outputs_fgrad['delta_x']

        if 'sdf_base' in outputs_fgrad:
            rgb_base = self.rgb_net(pnts_c, outputs_fgrad['normal_base'], view_dirs, cond, outputs_fgrad['feature_base'])
            outputs['rgb_base'] = rgb_base
            outputs['sdf_base'] = outputs_fgrad['sdf_base']
            outputs['normal_base'] = outputs_fgrad['normal_base']

        return outputs

    def forward_gradient(self, pts_o, pnts_c, cond, smpl_tfs, create_graph=True, retain_graph=True):

        if pnts_c.shape[0] == 0:
            return pnts_c.detach()
        pnts_c.requires_grad_(True)

        pnts_d = self._smpl_deformer.forward_skinning(pnts_c.unsqueeze(0), None, smpl_tfs).squeeze(0)
        num_dim = pnts_d.shape[-1]
        grads = []
        for i in range(num_dim):
            d_out = torch.zeros_like(pnts_d, requires_grad=False, device=pnts_d.device)
            d_out[:, i] = 1
            grad = torch.autograd.grad(
                outputs=pnts_d,
                inputs=pnts_c,
                grad_outputs=d_out,
                create_graph=create_graph,
                retain_graph=True if i < num_dim - 1 else retain_graph,
                only_inputs=True)[0]
            grads.append(grad)
        grads = torch.stack(grads, dim=-2)
        grads_inv = grads.inverse()

        outputs_def = self.sdf_func_with_non_rigid_deformer(pts_o, pnts_c, cond, is_gradient=True)

        gradient_xo = torch.einsum('bi,bij->bj', outputs_def['gradient'], grads_inv)
        if 'gradient_local' in outputs_def:
            gradient_local = outputs_def['gradient_local']
            gradient_xo = gradient_xo + gradient_local

        normal_xo = torch.nn.functional.normalize(gradient_xo, dim=1)

        feature = outputs_def['feature_sdf']
        if 'feature_def' in outputs_def:
            feature = torch.cat([feature, outputs_def['feature_def']], dim=-1)

        outputs = {}
        outputs['sdf'] = outputs_def['sdf']
        outputs['feature'] = feature
        outputs['normal'] = normal_xo
        outputs['gradient'] = gradient_xo

        outputs['xc'] = pnts_c

        if 'delta_x' in outputs_def:
            outputs['delta_x'] = outputs_def['delta_x']
            outputs['xc'] = outputs_def['xc']

        if 'gradient_base' in outputs_def:
            normal_base = torch.nn.functional.normalize(torch.einsum('bi,bij->bj', outputs_def['gradient_base'], grads_inv), dim=1)
            outputs['sdf_base'] = outputs_def['sdf_base']
            outputs['feature_base'] = outputs_def['feature_sdf_base']
            outputs['normal_base'] = normal_base
            if 'feature_def' in outputs_def:
                feature_def_base = torch.zeros_like(outputs_def['feature_def'])
                outputs['feature_base'] = torch.cat([outputs_def['feature_sdf_base'], feature_def_base], dim=-1)

        return outputs

    def volume_rendering(self, z_vals, z_max, density):
        density = density.reshape(-1, z_vals.shape[1]) # (batch_size * num_pixels) x N_samples

        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, z_max.unsqueeze(-1) - z_vals[:, -1:]], -1)

        # LOG SPACE
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1, device=density.device), free_energy], dim=-1)  # add 0 for transperancy 1 at t_0
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        fg_transmittance = transmittance[:, :-1]
        weights = alpha * fg_transmittance  # probability of the ray hits something here
        bg_transmittance = transmittance[:, -1]  # factor to be multiplied with the bg volume rendering

        return weights, bg_transmittance

    def get_box(self, l2w):
        rest_pose = self.smpl_server.joints_tpose[0].detach().cpu().numpy()
        volume_scales = self.non_rigid_deformer.get_axis_scale().detach().cpu().numpy()
        volume_boxes = create_axis_boxes(volume_scales)
        posed_boxes, child_idxs = boxes_to_pose(l2w, volume_boxes, rest_pose, skel_type=SMPLSkeleton)
        return posed_boxes, child_idxs


def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    return points_grad