import numpy as np
import cv2

def img_mix_2d_verts(img, verts, point_size=5):
    for idx, pt in enumerate(verts):
        cv2.circle(img, (pt[0], pt[1]), point_size, (0, 255, 0), -1)
    return img
def render_3d_2_2d(K, E, verts):
    E4 = E
    if K.shape[0] == 3:
        K4 = np.eye(4)
        K4[:3, :3] = K
    else:
        K4 = K

    verts_ex = np.ones((verts.shape[0], 1))
    verts = np.concatenate((verts, verts_ex), axis=-1)
    verts_cam = np.sum(verts[..., np.newaxis, :] * E4, -1)
    verts_cam = verts_cam / verts_cam[..., 2:3]
    verts_px = np.sum(verts_cam[..., np.newaxis, :] * K4, -1)
    verts_px = verts_px[..., :2]
    verts_px = verts_px.astype(np.int)

    return verts_px
