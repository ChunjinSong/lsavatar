import torch
import torch.nn.functional as F
import pytorch3d.transforms.rotation_conversions as p3dr


def rot_to_axisang(rot):
    return p3dr.matrix_to_axis_angle(rot)

def rot_to_rot6d(rot):
    return rot[..., :3, :2].flatten(start_dim=-2)

def axisang_to_rot(axisang):
    return p3dr.axis_angle_to_matrix(axisang)

def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (*,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x_shape = x.shape[:-1]
    x = x.reshape(-1,3,2)

    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1).reshape(*x_shape, 3, 3)

def axisang_to_rot6d(axisang):
    return rot_to_rot6d(axisang_to_rot(axisang))

def rot6d_to_axisang(rot6d):
    return rot_to_axisang(rot6d_to_rotmat(rot6d))