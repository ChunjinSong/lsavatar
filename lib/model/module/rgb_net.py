import torch.nn as nn
import torch
from lib.model.embedder.embedders import get_embedder

class RGBNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        dims_hidden = []
        for i in range(opt.n_layers):
            dims_hidden.append(opt.d_hid)

        self.mode = opt.mode
        dims = [opt.d_in + opt.d_sdf_feature + opt.d_deformed_feature] + list(dims_hidden) + [opt.d_out]

        self.embedview_fn = None
        if opt.multires_view > 0:
            embedview_fn, input_ch = get_embedder(opt.multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        if self.mode == 'nerf_frame_encoding':
            dims[0] += opt.dim_frame_encoding

        if self.mode == 'pose':
            self.dim_cond_embed = 8
            self.cond_dim = 23*6  # dimension of the body pose, global orientation excluded.
            # lower the condition dimension
            self.lin_pose = torch.nn.Linear(self.cond_dim, self.dim_cond_embed)
            dims[0] += 8

        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 1):
            dim_out = dims[l + 1]
            lin = nn.Linear(dims[l], dim_out)
            nn.init.kaiming_normal_(lin.weight, mode='fan_out', nonlinearity='relu')

            if opt.weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, points, normals, view_dirs, cond, feature_vectors, frame_latent_code=None):
        if self.embedview_fn is not None:
            normals = self.embedview_fn(normals)
        if self.mode == 'pose':
            body_pose = self.lin_pose(cond['smpl'])
            num_points = points.shape[0]
            body_pose = body_pose.unsqueeze(1).expand(-1, num_points, -1).reshape(num_points, -1)
            rendering_input = torch.cat([points, normals, body_pose, feature_vectors], dim=-1)
        else:
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)

        x = rendering_input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)
        x = self.sigmoid(x)
        return x

