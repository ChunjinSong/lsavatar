import torch.nn as nn
from lib.model.utils.skeleton_utils import *
import trimesh

def transform_batch_pts(pts: torch.Tensor, w2l: torch.Tensor):
    '''
    Transform points/vectors from world space to local space

    Parameters
    ----------
    pts: Tensor (..., 3) in world space
    skt: Tensor (..., N_joints, 4, 4) world-to-local transformation
    '''

    pts = torch.cat([pts, torch.ones(*pts.shape[:-1], 1, device=pts.device)], dim=-1)
    pts_l = (w2l @ pts.T).permute(2, 0, 1)

    return pts_l[..., :-1] # don't need the homogeneous part

class BoneAlignEmbedder(nn.Module):

    def __init__(self, rest_pose):
        super(BoneAlignEmbedder, self).__init__()
        self.rest_pose = rest_pose
        self.skel_type = SMPLSkeleton

        transforms, child_idxs = get_bone_align_transforms(rest_pose, self.skel_type)
        self.child_idxs = np.array(child_idxs)
        self.register_buffer('transforms', transforms)

    def forward(self, pts, w2l, rigid_idxs=None):
        if rigid_idxs is not None:
            w2l = w2l[..., rigid_idxs, :, :]
        pts_jt = transform_batch_pts(pts, w2l)
        pts_t = self.align_pts(pts_jt, self.transforms, rigid_idxs=rigid_idxs)

        return pts_t

    def align_pts(self, pts, align_transforms=None, rigid_idxs=None):
        if align_transforms is None:
            align_transforms = self.transforms
        if rigid_idxs is not None:
            align_transforms = align_transforms[rigid_idxs]
        pts_t = (align_transforms[..., :3, :3] @ pts[..., None]).squeeze(-1) \
                + align_transforms[..., :3, -1]
        return pts_t

    def unalign_pts(self, pts_t, align_transforms=None, rigid_idxs=None):
        if align_transforms is None:
            align_transforms = self.transforms
        if rigid_idxs is not None:
            align_transforms = align_transforms[rigid_idxs]
        pts_t = pts_t - align_transforms[..., :3, -1]
        pts = align_transforms[..., :3, :3].transpose(-1, -2) @ pts_t[..., None]
        return pts.squeeze(-1)