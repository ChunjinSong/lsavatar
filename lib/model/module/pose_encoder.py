import torch.nn as nn
import torch
from lib.model.embedder.embedders import get_embedder
from lib.model.embedder.pose_embedders import BoneAlignEmbedder
from lib.model.module.gnn_net import PoseGNN

class PoseEncoder(nn.Module):
    def __init__(self, rest_pose, opt):
        super().__init__()
        self.alpha = 2.
        self.beta = 6.

        self.dim_pose = opt.posi_pose.d_in
        if opt.posi_pose.multires > 0:
            embed_fn_pose, dim_pose = get_embedder(opt.posi_pose)
            self.embed_fn_pose = embed_fn_pose
            opt.graph_net.d_in = dim_pose

        self.dim_pts = opt.posi_pts.d_in
        if opt.posi_pts.multires > 0:
            embed_fn_pts, dim_pts = get_embedder(opt.posi_pts)
            self.embed_fn_pts = embed_fn_pts
            self.dim_pts = dim_pts

        self.w2l_func = BoneAlignEmbedder(rest_pose)
        self.gnn_net = PoseGNN(opt.graph_net, rest_pose)

        self.layer = nn.Linear(self.dim_pts + opt.graph_net.d_out, opt.d_out)

    def forward(self, pts, cond, w2l):
        N_joints = 24
        pose = cond['pose']
        # get pts_t (3d points in local space which has been aligned)
        x_l = self.w2l_func(pts, cond, w2l).reshape(-1, N_joints, 3)  # [N_rays, N_samples, N_joints, 3]
        x_l = x_l / self.gnn_net.get_axis_scale().reshape(1, N_joints, 3).abs()
        window = torch.exp(-self.alpha * ((x_l ** self.beta).sum(-1))).detach()

        invalid = ((x_l.abs() > 1).sum(-1) > 0).float()
        valid = (1 - invalid).reshape(-1, N_joints)
        valid_pts = torch.where(valid.sum(-1) > 0)[0]
        valid_jts = (valid.reshape(-1, 24)).sum(dim=0) > 0

        if self.embed_fn_pts is not None:
            x_l = self.embed_fn_pts(x_l) #[N, 24, dim]

        if self.embed_fn_pose is not None:
            pose = self.embed_fn_pose(pose.reshape(1, 24, 6))
        feat_pose = self.gnn_net(pose)  #[1, 24, dim]

        feat_pose = feat_pose.expand(x_l.shape[0], -1, -1)

        feat_pose = torch.cat([feat_pose, x_l], dim=-1)
        feat_pose = self.layer(feat_pose)
        feat_pose = feat_pose * window[..., None]
        feat_pose = feat_pose.sum(dim=1)

        outputs = {'valid_pts': valid_pts,
                   'valid_jts': valid_jts,
                   'local_pose': feat_pose}

        return outputs