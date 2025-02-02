import torch
import torch.nn as nn
import numpy as np
from lib.model.embedder.embedders import get_embedder
from lib.model.embedder.pose_embedders import BoneAlignEmbedder
from lib.model.module.gnn_net import PoseGNN, ParallelLinear

class DeformerMLP(nn.Module):
    def __init__(self, opt):
        super().__init__()

        dims_hidden = []
        for i in range(opt.n_layers):
            dims_hidden.append(opt.d_hid)

        dims = [opt.d_in + 3] + list(dims_hidden) + [opt.d_out + opt.d_deformed_feature]
        self.num_layers = len(dims)
        self.skip_in = opt.skip_in
        self.embed_fn = None
        self.opt = opt

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                dim_out = dims[l + 1] - dims[0]
            else:
                dim_out = dims[l + 1]

            lin = nn.Linear(dims[l], dim_out)
            if opt.weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, input, xc):
        # x = input
        x = torch.cat([input, xc], 1)
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.activation(x)
        # x = torch.tanh(x)
        return x

    def forward_sdf(self, xc, cond, local_pose=None, sdf_net=None, is_training=False):

        out_sdf_base = sdf_net.forward_sdf(xc, cond, is_gradient=True, is_training=is_training)
        # normal = torch.nn.functional.normalize(out_sdf_base['gradient'].detach(), dim=1)
        # w_sdf = 1.0 / (1 + (out_sdf_base['sdf'].detach() / 0.05) ** 4)
        w_sdf = 1.0

        outputs = {}

        out_def = self.forward(local_pose, xc)
        delta_x = out_def[..., :self.opt.d_out] * w_sdf
        xc = xc + delta_x
        out_sdf = sdf_net(xc, cond)

        outputs['sdf_base'] = out_sdf_base['sdf']
        outputs['feature_sdf_base'] = out_sdf_base['feature_sdf']
        outputs['gradient_base'] = out_sdf_base['gradient']
        outputs['sdf'] = out_sdf['sdf']
        outputs['delta_x'] = delta_x
        outputs['xc'] = xc
        outputs['feature_sdf'] = out_sdf['feature_sdf']

        if self.opt.d_deformed_feature > 0:
            outputs['feature_def'] = out_def[..., self.opt.d_out:]

        return outputs

class Deformer(nn.Module):
    def __init__(self, opt, rest_pose):
        super().__init__()
        self.alpha = 2.
        self.beta = 6.
        self.N_joints = 24
        self.opt = opt

        self.dim_pose = opt.posi_pose.d_in
        # positional encoding for pose
        if opt.posi_pose.multires > 0:
            embed_fn_pose, dim_pose = get_embedder(opt.posi_pose)
            self.embed_fn_pose = embed_fn_pose
            opt.graph_net.d_in = dim_pose

        # positional encoding for point
        self.dim_pts = opt.posi_pts.d_in
        if opt.posi_pts.multires > 0:
            embed_fn_pts, dim_pts = get_embedder(opt.posi_pts)
            self.embed_fn_pts = embed_fn_pts
            self.dim_pts = dim_pts

        # mapping coordinate of query point to bone coordinate
        self._w2l_func = BoneAlignEmbedder(rest_pose)

        self.gnn_net = PoseGNN(opt.graph_net, rest_pose)
        self.dim_pose = opt.graph_net.d_out

        self.lin_local = ParallelLinear(self.N_joints, self.dim_pts + self.dim_pose, opt.d_pose_feature, is_reset_parameters=False)

        self.deformermlp = DeformerMLP(opt.deformer_mlp)


    def get_axis_scale(self):
        scale = self.gnn_net.get_axis_scale()
        return scale

    def w2l_func(self, xo, w2l):
        xl = self._w2l_func(xo, w2l).reshape(-1, self.N_joints, 3)  # [N_rays, N_samples, N_joints, 3]
        xl = xl / self.gnn_net.get_axis_scale().reshape(1, self.N_joints, 3).abs()
        window = torch.exp(-self.alpha * ((xl ** self.beta).sum(-1))).detach()
        # window = window / (torch.sum(window, dim=-1, keepdim=True) + 1e-8)
        invalid = ((xl.abs() > 1).sum(-1) > 0).float().detach()
        valid = (1 - invalid).reshape(-1, self.N_joints)
        valid_pts = torch.where(valid.sum(-1) > 0)[0]
        valid_jts = (valid.reshape(-1, 24)).sum(dim=0) > 0

        outputs = {'valid_pts': valid_pts,
                   'valid_jts': valid_jts,
                   'xl': xl,
                   'window': window}

        return outputs

    def forward(self, xo, xc, cond, sdf_net=None, is_gradient=False, is_training=False):
        pose = cond['smpl_pose']

        if is_gradient:
            xo.requires_grad_()
            xc.requires_grad_()

        window_xc = self.w2l_func(xc, cond['smpl_w2l_c'])['window']

        out_xl = self.w2l_func(xo, cond['smpl_w2l'])
        xl = out_xl['xl']
        window = out_xl['window'] * window_xc
        window = window / (torch.sum(window, dim=-1, keepdim=True) + 1e-8)

        if self.embed_fn_pts is not None:
            # embed_fn_pts, _ = get_embedder(self.opt.posi_pts, cond['global_step'])  #[N, 24, dim]
            xl = self.embed_fn_pts(xl)

        if self.embed_fn_pose is not None:
            pose = self.embed_fn_pose(pose.reshape(1, 24, 6))

        local_pose = self.gnn_net(pose)  #[1, 24, dim]
        local_pose = local_pose.expand(xl.shape[0], -1, -1)
        local_pose = torch.cat([local_pose, xl], dim=-1)
        local_pose = self.lin_local(local_pose)
        local_pose = local_pose * window[..., None]
        local_pose = local_pose.sum(dim=1)

        out_def = self.deformermlp.forward_sdf(xc, cond, local_pose, sdf_net, is_training)

        outputs = out_def

        if is_gradient:
            sdf = out_def['sdf']

            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            gradients_xc = torch.autograd.grad(
                outputs=sdf,
                inputs=xc,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
            gradients_xc = gradients_xc.reshape(gradients_xc.shape[0], -1)

            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            gradients_xo = torch.autograd.grad(
                outputs=sdf,
                inputs=xo,
                grad_outputs=d_output,
                create_graph=is_training,
                retain_graph=is_training,
                only_inputs=True)[0]
            gradients_xo = gradients_xo.reshape(gradients_xo.shape[0], -1)

            outputs['gradient_local'] = gradients_xo
            outputs['gradient'] = gradients_xc

        return outputs
