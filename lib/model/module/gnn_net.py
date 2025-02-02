import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from lib.model.utils.skeleton_utils import Skeleton, SMPLSkeleton, get_skel_profile_from_rest_pose

from typing import Optional, List, Union, Callable

'''
Modified from Skeleton-aware Networks https://github.com/DeepMotionEditing/deep-motion-editing
'''

class ParallelLinear(nn.Module):

    def __init__(self, n_parallel, in_feat, out_feat, share=False, bias=True, is_reset_parameters=True):
        super().__init__()
        self.n_parallel = n_parallel
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.share = share

        if not self.share:
            self.register_parameter('weight',
                                    nn.Parameter(torch.randn(n_parallel, in_feat, out_feat),
                                                 requires_grad=True)
                                   )
            if bias:
                self.register_parameter('bias',
                                        nn.Parameter(torch.randn(1, n_parallel, out_feat),
                                                     requires_grad=True)
                                       )
        else:
            self.register_parameter('weight', nn.Parameter(torch.randn(1, in_feat, out_feat),
                                                           requires_grad=True))
            if bias:
                self.register_parameter('bias', nn.Parameter(torch.randn(1, 1, out_feat), requires_grad=True))
        if not hasattr(self, 'bias'):
            self.bias = None
        if is_reset_parameters:
            self.reset_parameters()

    def reset_parameters(self):

        for n in range(self.n_parallel):
            # transpose because the weight order is different from nn.Linear
            nn.init.kaiming_uniform_(self.weight[n].T.data, a=math.sqrt(5))

        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.)

    def forward(self, x):
        weight, bias = self.weight, self.bias
        if self.share:
            weight = weight.expand(self.n_parallel, -1, -1)
            if bias is not None:
                bias = bias.expand(-1, self.n_parallel, -1)
        out = torch.einsum("bkl,klj->bkj", x, weight.to(x.device))
        if bias is not None:
            out = out + bias.to(x.device)
        return out

    def extra_repr(self):
        return "n_parallel={}, in_features={}, out_features={}, bias={}".format(
            self.n_parallel, self.in_feat, self.out_feat, self.bias is not None
        )

def init_volume_scale(base_scale, skel_profile, skel_type):
    # TODO: hard-coded some parts for now ...
    # TODO: deal with multi-subject
    joint_names = skel_type.joint_names
    N_joints = len(joint_names)
    bone_lens = skel_profile['bone_lens'][0]
    bone_lens_to_child = skel_profile['bone_lens_to_child'][0]

    # indices to all body parts
    head_idxs = skel_profile['head_idxs']
    torso_idxs = skel_profile['torso_idxs']
    arm_idxs = skel_profile['arm_idxs']
    leg_idxs = skel_profile['leg_idxs']
    collar_idxs = skel_profile['collar_idxs']

    # some widths
    shoulder_width = skel_profile['shoulder_width'][0]
    knee_width = skel_profile['knee_width'][0]
    collar_width = skel_profile['knee_width'][0]

    # init the scale for x, y and z
    # width, depth
    x_lens = torch.ones(N_joints) * base_scale
    y_lens = torch.ones(N_joints) * base_scale

    # half-width of thighs cannot be wider than the distant between knees in rest pose
    x_lens[leg_idxs] = knee_width * 0.8
    y_lens[leg_idxs] = knee_width * 0.8

    #  half-width of your body and head cannot be wider than shoulder distance (to some scale)
    x_lens[torso_idxs] = shoulder_width * 0.50
    y_lens[torso_idxs] = shoulder_width * 0.50
    x_lens[collar_idxs] = collar_width * 0.40
    y_lens[collar_idxs] = collar_width * 0.40

    #  half-width of your arms cannot be wider than collar distance (to some scale)
    x_lens[arm_idxs] = collar_width * 0.40
    y_lens[arm_idxs] = collar_width * 0.40

    # set scale along the bone direction
    # don't need full bone lens because the volume is supposed to centered at the middle of a bone
    z_lens = torch.tensor(bone_lens_to_child.copy().astype(np.float32))
    z_lens = z_lens * 0.8

    # deal with end effectors: make them grow freely
    x_lens[head_idxs] = shoulder_width * 0.30
    y_lens[head_idxs] = shoulder_width * 0.35
    # TODO: hack: assume at index 1 we have the head
    y_lens[head_idxs[1]] = shoulder_width * 0.6
    z_lens[head_idxs] = z_lens.max() * 0.30

    # find the lengths from end effector to their parents
    end_effectors = np.array([i for i, v in enumerate(z_lens) if v < 0 and i not in head_idxs])
    z_lens[end_effectors] = torch.tensor(skel_profile['bone_lens_to_child'][0][skel_type.joint_trees[end_effectors]].astype(np.float32))

    # handle hands and foots
    scale = torch.stack([x_lens, y_lens, z_lens], dim=-1)

    return scale

def skeleton_to_graph(skel: Optional[Skeleton] = None, edges: Optional[List[Union[List, np.ndarray]]] = None):
    ''' Turn skeleton definition to adjacency matrix and edge list
    '''

    if skel is not None:
        edges = []
        for i, j in enumerate(skel.joint_trees):
            if i == j:
                continue
            edges.append([j, i])
    else:
        assert edges is not None

    n_nodes = np.max(edges) + 1
    adj = np.eye(n_nodes, dtype=np.float32)

    for edge in edges:
        adj[edge[0], edge[1]] = 1.0
        adj[edge[1], edge[0]] = 1.0

    return adj, edges


def clamp_deform_to_max(x_d: torch.Tensor, max_deform: float = 0.04):
    x_d_scale = x_d.detach().norm(dim=-1)
    masks = (x_d_scale < max_deform)[..., None]
    return x_d * masks


class DenseWGCN(nn.Module):
    """ Basic GNN layer with learnable adj weights
    """

    def __init__(
            self,
            adj: torch.Tensor,
            in_channels: int,
            out_channels: int,
            init_adj_w: float = 0.05,
            bias: bool = True,
            **kwargs
    ):
        super(DenseWGCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        adj = adj.clone()
        idx = torch.arange(adj.shape[-1])
        adj[:, idx, idx] = 1

        init_w = init_adj_w
        perturb = 0.1
        adj_w = (adj.clone() * (init_w + (torch.rand_like(adj) - 0.5) * perturb).clamp(min=0.01, max=1.0))
        adj_w[:, idx, idx] = 1.0

        self.lin = nn.Linear(in_channels, out_channels)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True)
        else:
            self.register_parameter('bias', None)

        self.register_buffer('adj', adj)  # fixed, not learnable
        self.register_parameter('adj_w', nn.Parameter(adj_w, requires_grad=True))  # learnable

    def get_adjw(self):
        adj, adj_w = self.adj, self.adj_w

        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        adj_w = adj_w.unsqueeze(0) if adj_w.dim() == 2 else adj_w
        adj_w = adj_w * adj  # masked out not connected part

        return adj_w

    def forward(self, x: torch.Tensor):

        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj_w = self.get_adjw().to(x.device)

        out = self.lin(x)
        out = torch.matmul(adj_w, out)

        if self.bias is not None:
            out = out + self.bias

        return out


class DensePNGCN(DenseWGCN):
    """ Basic GNN layer with learnable adj weights, and each node has its own linear layer
    """

    def __init__(
            self,
            adj: torch.Tensor,
            in_channel: int,
            out_channel: int,
            *args,
            **kwargs
    ):
        super(DensePNGCN, self).__init__(
            adj,
            in_channel,
            out_channel,
            *args,
            **kwargs
        )
        self.lin = ParallelLinear(adj.shape[-1], in_channel, out_channel, bias=False)


class BasicGNN(nn.Module):
    """ A basic GNN with several graph layers
    """

    def __init__(self, opt, rest_pose, skel_type):
        """
        mask_root: Bool, to remove root input so everything is in relative coord
        """

        super(BasicGNN, self).__init__()

        skel_profile = get_skel_profile_from_rest_pose(rest_pose, skel_type)

        gcn_module = DensePNGCN

        self.skel_profile = skel_profile
        self.skel_type = skel_profile['skel_type']
        self.d_in = opt.d_in
        self.d_out = opt.d_out
        self.keep_extra_joint = opt.keep_extra_joint

        self.rigid_idxs = skel_profile['rigid_idxs']
        self.mask_root = opt.mask_root
        self.W = opt.W
        self.D = opt.D

        adj_matrix, _ = skeleton_to_graph(self.skel_type)
        self.adj_matrix = adj_matrix
        self.gcn_module_kwargs = {}

        self.nl = F.relu
        self.init_network(gcn_module)

    def init_network(self, gcn_module: Callable):
        adj_matrix, skel_type = self.adj_matrix, self.skel_type
        d_in = self.d_in
        W, D = self.W, self.D

        n_nodes = adj_matrix.shape[-1]
        adj_matrix = torch.tensor(adj_matrix).view(1, n_nodes, n_nodes)

        layers = [gcn_module(adj_matrix, d_in, W, **self.gcn_module_kwargs)]
        for i in range(D - 2):
            layers += [gcn_module(adj_matrix, W, W, **self.gcn_module_kwargs)]

        layers += [gcn_module(adj_matrix, W, self.d_out, **self.gcn_module_kwargs)]
        self.layers = nn.ModuleList(layers)

        if self.mask_root:
            # mask root inputs, so that everything is in relative coordinate
            mask = torch.ones(1, len(self.skel_type.joint_names), 1)
            mask[:, self.skel_type.root_id, :] = 0.
            self.register_buffer('mask', mask)

    def forward(self, inputs: torch.Tensor, **kwargs):

        n = inputs
        if self.mask_root:
            n = n * self.mask

        for i, l in enumerate(self.layers):
            n = l(n)
            if (i + 1) < len(self.layers) and self.nl is not None:
                n = self.nl(n, inplace=True)
            if (i + 2) == len(self.layers) and self.rigid_idxs is not None and not self.keep_extra_joint:
                n = n[:, self.rigid_idxs]
        return n

    def get_adjw(self):
        adjw_list = []

        for m in self.modules():
            if hasattr(m, 'adj_w'):
                adjw_list.append(m.get_adjw())

        return adjw_list


class MixGNN(BasicGNN):

    def __init__(self, opt, rest_pose, skel_type):
        """
        Parameters
        ----------
        fc_D: int, start using fc_module at layer fc_D
        fc_module: nn.Module, module to use at layer fc_D
        """
        self.fc_module = ParallelLinear
        self.fc_D = opt.fc_D
        self.legacy = opt.legacy

        super(MixGNN, self).__init__(opt, rest_pose, skel_type)

    def init_network(self, gcn_module: Callable):
        adj_matrix, skel_type = self.adj_matrix, self.skel_type
        d_in = self.d_in
        W, D, fc_D = self.W, self.D, self.fc_D

        n_nodes = adj_matrix.shape[-1]
        adj_matrix = torch.tensor(adj_matrix).view(1, n_nodes, n_nodes)

        layers = [gcn_module(adj_matrix, d_in, W, **self.gcn_module_kwargs)]
        for i in range(D - 2):
            if i + 1 < fc_D:
                layers += [gcn_module(adj_matrix, W, W, **self.gcn_module_kwargs)]
            else:
                layers += [self.fc_module(n_nodes, W, W)]

        if self.fc_module in [ParallelLinear]:
            n_nodes = len(self.rigid_idxs) if self.rigid_idxs is not None and not self.keep_extra_joint else n_nodes
            layers += [self.fc_module(n_nodes, W, self.d_out)]
        else:
            layers += [self.fc_module(adj_matrix, W, self.d_out, **self.gcn_module_kwargs)]

        if self.mask_root or self.legacy:
            mask = torch.ones(1, len(self.skel_type.joint_names), 1)
            mask[:, self.skel_type.root_id, :] = 0.
            self.register_buffer('mask', mask)

        self.layers = nn.ModuleList(layers)


class PoseGNN(MixGNN):

    def __init__(self, opt, rest_pose):
        self.opt_scale = opt.opt_scale
        skel_type = SMPLSkeleton
        super(PoseGNN, self).__init__(opt, rest_pose, skel_type)

        self.legacy = opt.legacy
        self.init_scale(opt.base_scale)

    def init_scale(self, base_scale: float = 0.5):
        N_joints = len(self.skel_type.joint_names)

        scale = torch.ones(N_joints, 3) * base_scale
        if self.skel_profile is not None:
            if self.skel_type == SMPLSkeleton:
                scale = init_volume_scale(base_scale, self.skel_profile, self.skel_type) * 1.5
            else:
                raise NotImplementedError
        self.register_buffer('base_scale', scale.clone())
        self.register_parameter('axis_scale', nn.Parameter(scale, requires_grad=self.opt_scale))

    def get_axis_scale(self):
        axis_scale = self.axis_scale.abs()
        if self.legacy:
            return axis_scale
        diff = axis_scale.detach() - self.base_scale * 0.95
        return torch.maximum(axis_scale, axis_scale - diff)

    def check_invalid(self, x: torch.Tensor):
        """ Assume points are in volume space already

        Args:
            x (torch.Tensor): (N_rays, N_samples, N_joints, 3)

        Returns:
            x_v (torch.Tensor): (N_rays, N_samples, N_joints, 3) points scaled by volume scales
            invalid (torch.Tensor): (N_rays, N_samples, N_joints) invalid mask indicating whether the sampled points are out of the volume
        """
        x_v = x / self.get_axis_scale().reshape(1, 1, -1, 3).abs()
        invalid = ((x_v.abs() > 1).sum(-1) > 0).float()
        return x_v, invalid
