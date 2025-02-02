import torch.nn as nn
import torch
import numpy as np
from lib.model.embedder.embedders import get_embedder

class SDFNet(nn.Module):
    def __init__(self, opt):
        super().__init__()

        dims_hidden = []
        for i in range(opt.n_layers):
            dims_hidden.append(opt.d_hid)

        dims = [opt.d_in] + list(dims_hidden) + [opt.d_out + opt.d_sdf_feature]
        self.num_layers = len(dims)
        self.skip_in = opt.skip_in
        self.embed_fn = None
        self.opt = opt
        self.d_in = opt.d_in

        if opt.posi_pts.multires > 0:
            embed_fn, input_ch = get_embedder(opt.posi_pts)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.cond = opt.cond
        if self.cond == 'smpl':
            self.cond_layer = [0]
            # self.cond_dim = 23*6
            self.cond_dim = opt.comd_dim * 23

        elif self.cond == 'frame':
            self.cond_layer = [0]
            self.cond_dim = opt.dim_frame_encoding

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                dim_out = dims[l + 1] - dims[0]
            else:
                dim_out = dims[l + 1]

            dim_in = dims[l]

            if self.cond != 'none' and l in self.cond_layer:
                lin = nn.Linear(dim_in + self.cond_dim, dim_out)
            else:
                lin = nn.Linear(dim_in, dim_out)
            if opt.init == 'geometry':
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight,
                                          mean=np.sqrt(np.pi) /
                                               np.sqrt(dims[l]),
                                          std=0.0001)
                    torch.nn.init.constant_(lin.bias, -opt.bias)
                elif opt.posi_pts.multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, opt.d_in:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :opt.d_in], 0.0,
                                          np.sqrt(2) / np.sqrt(dim_out))
                elif opt.posi_pts.multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(dim_out))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - opt.d_in):],
                                            0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(dim_out))
            if opt.init == 'zero':
                init_val = 1e-5
                if l == self.num_layers - 2:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.uniform_(lin.weight, -init_val, init_val)
            if opt.weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, cond):
        '''
        :param input:
        :param cond:
        :return: sdf, feature
        '''
        num_point, num_dim = input.shape
        if num_point == 0: return input

        if self.cond != 'none':
            input_cond = cond[self.cond]
            num_batch, num_cond = input_cond.shape
            input_cond = input_cond.unsqueeze(1).expand(num_batch, num_point, num_cond)
            input_cond = input_cond.reshape(num_batch * num_point, num_cond)

        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if self.cond != 'none' and l in self.cond_layer:
                x = torch.cat([x, input_cond], dim=-1)
            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.softplus(x)
        x = torch.tanh(x)

        outputs = {'sdf': x[..., :1], 'feature_sdf': x[..., 1:]}

        return outputs

    def forward_sdf(self, x, cond, is_gradient=False, is_training=False):
        '''
        :param x: canonical coordinate
        :param cond:
        :param istraining:
        :return: sdf, feature gradient
        '''
        if is_gradient:
            x.requires_grad_()
            outputs = self.forward(x, cond)
            y = outputs['sdf']
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(outputs=y,
                                            inputs=x,
                                            grad_outputs=d_output,
                                            create_graph=is_training,
                                            retain_graph=is_training,
                                            only_inputs=True)[0]
            gradients = gradients.reshape(gradients.shape[0], -1)
            outputs['gradient'] = gradients
        else:
            outputs = self.forward(x, cond)

        return outputs


