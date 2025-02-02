import pytorch_lightning as pl
import torch.optim as optim
from lib.utils.meshing import generate_mesh
from lib.model.module.body_model_params import BodyModelParams
from lib.smpl.deformer import SMPLDeformer
import cv2
import torch
import hydra
import os
import numpy as np
import trimesh
from lib.smpl.deformer import skinning
from lib.utils import utils
import h5py
from lib.model import \
    create_model, \
    load_loss
from lib.model.utils.rotation_format import axisang_to_rot6d, axisang_to_rot, rot_to_axisang
import math
import time

class V2AModel(pl.LightningModule):
    def __init__(self, opt) -> None:
        super().__init__()

        self.opt = opt
        self.path_h5py = os.path.join(hydra.utils.to_absolute_path('..'), 'data', opt.dataset.metainfo.data_dir)

        dataset_h5py = h5py.File(self.path_h5py, 'r')
        n_images = dataset_h5py['img_shape'][0]
        num_training_frames = n_images
        training_indices = list(range(0, n_images, 1))
        shape = dataset_h5py['mean_shape'][:]
        poses = dataset_h5py['poses'][training_indices]
        trans = dataset_h5py['normalize_trans'][training_indices]
        dataset_h5py.close()

        self.gender = opt.dataset.metainfo.gender
        self.model = create_model(opt.model, shape, self.gender)
        self.start_frame = opt.dataset.metainfo.start_frame
        self.end_frame = n_images
        self.training_modules = ["model"]

        self.body_model_params = BodyModelParams(num_training_frames, model_type='smpl')
        self.load_body_model_params(shape, poses, trans)
        optim_params = self.body_model_params.param_names
        for param_name in optim_params:
            self.body_model_params.set_requires_grad(param_name, requires_grad=True)
        self.training_modules += ['body_model_params']
        self.loss = load_loss(opt.model.loss)

    def load_body_model_params(self, shape, poses, trans):
        body_model_params = {param_name: [] for param_name in self.body_model_params.param_names}
        body_model_params['betas'] = torch.tensor(shape[None], dtype=torch.float32)
        body_model_params['global_orient'] = torch.tensor(poses[:, :3], dtype=torch.float32)
        body_model_params['body_pose'] = torch.tensor(poses[:, 3:], dtype=torch.float32)
        body_model_params['transl'] = torch.tensor(trans, dtype=torch.float32)

        for param_name in body_model_params.keys():
            self.body_model_params.init_parameters(param_name, body_model_params[param_name], requires_grad=False)


    def get_optimizer(self):
        
        _optimizers = {
            'adam': optim.Adam
        }
        
        optimizer = _optimizers[self.opt.model.optimizer.optimizer]

        cus_lr_names = [k[3:] for k in self.opt.model.optimizer.keys() if k.startswith('lr_')]
        params = []
        print('\n\n********** learnable parameters **********\n')
        for key, value in self.model.named_parameters():
            if not value.requires_grad:
                continue

            is_assigned_lr = False
            for lr_name in cus_lr_names:
                if lr_name in key:
                    params += [{"params": [value],
                                "lr": self.opt.model.optimizer[f'lr_{lr_name}'],
                                "name": lr_name}]
                    print(f"{key}: lr = {self.opt.model.optimizer[f'lr_{lr_name}']}")
                    is_assigned_lr = True

            if not is_assigned_lr:
                params += [{"params": [value],
                            "name": key}]
                print(f"{key}: lr = {self.opt.model.optimizer.lr}")

        params += [{'params': self.body_model_params.parameters(),
                    'lr': self.opt.model.optimizer['lr_body_model_params'],
                    'name': 'body_model_params'}]
        print(f"body_model_params: lr = {self.opt.model.optimizer[f'lr_body_model_params']}")

        print('\n******************************************\n\n')

        if self.opt.model.optimizer.optimizer == 'adam':
            optimizer = optimizer(params, lr=self.opt.model.optimizer.lr, betas=(0.9, 0.999))
        else:
            assert False, "Unsupported Optimizer."

        return optimizer

    def update_lr(self, optimizer, iter_step):
        decay_rate = 0.1
        decay_steps = self.opt.model.optimizer.lrate_decay * 1000
        decay_value = decay_rate ** (iter_step / decay_steps)
        for param_group in optimizer.param_groups:
            if f"lr_{param_group['name']}" in self.opt.model.optimizer:
                base_lr = self.opt.model.optimizer[f"lr_{param_group['name']}"]
                new_lrate = base_lr * decay_value
            else:
                new_lrate = self.opt.model.optimizer.lr * decay_value
            param_group['lr'] = new_lrate

    def configure_optimizers(self):
        self.optimizer = self.get_optimizer()
        return [self.optimizer]

    def training_step(self, batch):

        inputs, targets = batch

        inputs['global_step'] = self.global_step
        inputs['current_epoch'] = self.current_epoch
        inputs['frame_idx'] = inputs["idx"]

        batch_idx = inputs["idx"]
        body_model_params = self.body_model_params(batch_idx)
        inputs['smpl_pose'] = torch.cat((body_model_params['global_orient'], body_model_params['body_pose']), dim=1)
        inputs['smpl_shape'] = body_model_params['betas']
        inputs['smpl_trans'] = body_model_params['transl']

        model_outputs = self.model(inputs)

        loss_output = self.loss(model_outputs, targets, self.global_step, self.model.use_non_rigid_deformer)

        if 'delta_x' in model_outputs:
            dx_avg = model_outputs['delta_x'].detach().abs().mean(0)
            dx_x, dx_y, dx_z = dx_avg
            self.log('dx/dx_x', dx_x.item())
            self.log('dx/dx_y', dx_y.item())
            self.log('dx/dx_z', dx_z.item())
            self.log('dx/dx_avg', model_outputs['delta_x_avg'].item())
            self.log('dx/dx_max', model_outputs['delta_max'].item())

        if 'vol_scale' in model_outputs:
            scale_avg = model_outputs['vol_scale'].detach().mean(0)
            scale_x, scale_y, scale_z = scale_avg
            self.log('vol_scale/scale_x', scale_x.item())
            self.log('vol_scale/scale_y', scale_y.item())
            self.log('vol_scale/scale_z', scale_z.item())

        for k, v in loss_output.items():
            if self.opt.model.mode == 'sockeye':
                self.log(k, v.item())
            else:
                self.log(k, v.item(), prog_bar=True, on_step=True)

        return loss_output["loss"]

    def optimizer_step(
            self,
            epoch: int,
            batch_idx: int,
            optimizer,
            optimizer_idx: int = 0,
            optimizer_closure=None,
            on_tpu: bool = False,
            using_native_amp: bool = False,
            using_lbfgs: bool = False,
    ) -> None:

        self.update_lr(self.optimizer, self.global_step)
        optimizer.step(closure=optimizer_closure)

    def training_epoch_end(self, outputs) -> None:
        return super().training_epoch_end(outputs)

    def validation_step(self, batch, *args, **kwargs):
        inputs, targets = batch
        inputs['current_epoch'] = self.current_epoch
        inputs['global_step'] = self.global_step
        inputs['frame_idx'] = inputs['idx_pose']
        self.model.eval()

        # update pose params
        body_model_params = self.body_model_params(inputs['idx_pose'])
        inputs['smpl_pose'] = torch.cat((body_model_params['global_orient'], body_model_params['body_pose']), dim=1)
        inputs['smpl_shape'] = body_model_params['betas']
        inputs['smpl_trans'] = body_model_params['transl']

        self.get_save_boxes(path='box')
        self.get_save_meshes(inputs, base_name=str(self.current_epoch), path='rendering')
        self.get_save_imgs(inputs, targets, base_name=str(self.current_epoch), path='rendering')

    def _test(self, batch):
        inputs, targets = batch
        idx = inputs['idx'].cpu().numpy()
        n_cam = targets['n_cam'].cpu().numpy()
        idx_pose = inputs['idx_pose']
        inputs['current_epoch'] = 1e10
        inputs['global_step'] = 1e10

        # update pose params
        if idx_pose > -1:
            body_model_params = self.body_model_params(inputs['idx_pose'])
            inputs["smpl_shape"] = body_model_params['betas'] if body_model_params['betas'].dim() == 2 else \
            body_model_params['betas'].unsqueeze(0)
            # smpl_trans = body_model_params['transl']
            inputs["smpl_pose"] = torch.cat((inputs["smpl_pose"][..., :3], body_model_params['body_pose']), dim=1)

        self.get_save_imgs(inputs, targets, base_name=f'{int(idx):04d}',
                           path=f'{self.opt.dataset.testing.type}/test_rendering')

    def test_canonical(self, batch):
        inputs, targets = batch
        idx = inputs['idx'].cpu().numpy()
        inputs['current_epoch'] = 1e10
        inputs['global_step'] = 1e10

        # reset pose param to render img with canonical shape
        smpl_params_canoical = self.model._smpl_deformer.smpl.param_canonical.clone()
        smpl_params_canoical[:, 76:] = torch.tensor(inputs["smpl_shape"]).float().to(smpl_params_canoical.device)
        cano_scale, cano_transl, cano_thetas, cano_betas = torch.split(smpl_params_canoical, [1, 3, 72, 10], dim=1)

        if idx == 0:
            pose_global = cano_thetas[0, :3]
            pose_mat = torch.eye(4)
            pose_mat[:3, :3] = axisang_to_rot(torch.tensor(pose_global))
            pose_rot = torch.tensor(utils.rotate_y(math.pi)) @ pose_mat
            pose_rot = rot_to_axisang(pose_rot[:3, :3])
            cano_thetas[0, :3] = pose_rot

        inputs['smpl_pose'] = cano_thetas

        self.get_save_imgs(inputs, targets, base_name=f'{int(idx):04d}',
                           path=f'{self.opt.dataset.testing.type}/test_rendering')


    def test_step(self, batch, *args, **kwargs):
        if self.opt.dataset.testing.type == 'canonical':
            self.test_canonical(batch)
        else:
            self._test(batch)


    def query_oc(self, pts_c, cond, smpl_tfs):
        pts_c = pts_c.reshape(-1, 3)
        shape = pts_c.shape
        pts_o = self.model.smpl_deformer_xc2xo(pts_c, smpl_tfs)

        sdf = torch.ones(shape[0], 1, device=pts_c.device) * 4.0
        sdf_base = torch.ones(shape[0], 1, device=pts_c.device) * 4.0

        valid_idxs, _ = self.model.get_valid(pts_c, pts_o, cond)
        if len(valid_idxs) > 0:
            pts_o = pts_o[valid_idxs]
            pts_c = pts_c[valid_idxs]

            torch.set_grad_enabled(True)
            out = self.model.sdf_func_with_non_rigid_deformer(pts_o, pts_c, cond)
            sdf[valid_idxs] = out['sdf'].reshape(-1, 1)

            if 'sdf_base' in out:
                sdf_base[valid_idxs] = out['sdf_base'].reshape(-1, 1)
        outputs = {'sdf': sdf}
        outputs['sdf_base'] = sdf_base

        return outputs

    def query_wc(self, x):
        
        x = x.reshape(-1, 3)
        w = self.model._smpl_deformer.query_weights(x)
    
        return w

    def query_od(self, pts_o, cond, smpl_tfs, smpl_verts):
        pts_o = pts_o.reshape(-1, 3)
        pts_c, _ = self.model._smpl_deformer.forward(pts_o, smpl_tfs, return_weights=False, inverse=True, smpl_verts=smpl_verts)

        return self.query_oc(pts_c, cond, smpl_tfs)

    def get_deformed_mesh_fast_mode(self, verts, smpl_tfs):
        verts = torch.tensor(verts, device=self.device).float()
        weights = self.model._smpl_deformer.query_weights(verts)
        verts_deformed = skinning(verts.unsqueeze(0),  weights, smpl_tfs).data.cpu().numpy()[0]
        return verts_deformed

    def get_save_boxes(self, path):
        smpl_jnts_cnl = self.model.smpl_server.joints_c.squeeze(0).detach().cpu().numpy()
        smpl_verts_cnl = self.model.smpl_server.verts_c.squeeze(0).detach().cpu().numpy()
        smpl_l2w_cnl = self.model.smpl_server.w2l_c.squeeze(0).inverse()
        smpl_l2w_cnl = smpl_l2w_cnl.detach().cpu().numpy()
        boxes_cnl, child_idxs = self.model.get_box(smpl_l2w_cnl)
        self.save_boxes(boxes_cnl, smpl_verts_cnl, smpl_jnts_cnl, child_idxs, path)

    def generate_meshes(self, cond, smpl_scale, smpl_trans, smpl_pose, smpl_shape, vol_type='cnl'):
        self.model._smpl_deformer = SMPLDeformer(betas=smpl_shape, gender=self.gender, K=7).to(self.device)
        smpl_outputs = self.model.smpl_server(smpl_scale, smpl_trans, smpl_pose, smpl_shape)
        smpl_w2l = smpl_outputs['smpl_l2w'].squeeze(0).inverse()
        smpl_tfs = smpl_outputs['smpl_tfs']
        smpl_verts = self.model.smpl_server.verts_c[0]
        cond['smpl_w2l'] = smpl_w2l
        cond['smpl_w2l_c'] = self.model.smpl_server.w2l_c

        canonical_mesh_base = None
        deformed_mesh_base = None
        canonical_mesh = None
        deformed_mesh = None

        if vol_type=='cnl':
            smpl_verts = self.model.smpl_server.verts_c[0]
            canonical_mesh_base = generate_mesh(lambda x: self.query_oc(x, cond, smpl_tfs), smpl_verts, type='sdf_base', point_batch=10000, res_up=3, device=self.device)
            if canonical_mesh_base is not None:
                canonical_mesh_base = trimesh.Trimesh(canonical_mesh_base.vertices, canonical_mesh_base.faces)
                verts_deformed_base = self.get_deformed_mesh_fast_mode(canonical_mesh_base.vertices, smpl_tfs)
                deformed_mesh_base = trimesh.Trimesh(vertices=verts_deformed_base, faces=canonical_mesh_base.faces, process=False)

            canonical_mesh = generate_mesh(lambda x: self.query_oc(x, cond, smpl_tfs), smpl_verts, type='sdf', point_batch=10000, res_up=3, device=self.device)
            if canonical_mesh is not None:
                canonical_mesh = trimesh.Trimesh(canonical_mesh.vertices, canonical_mesh.faces)
                verts_deformed = self.get_deformed_mesh_fast_mode(canonical_mesh.vertices, smpl_tfs)
                deformed_mesh = trimesh.Trimesh(vertices=verts_deformed, faces=canonical_mesh.faces, process=False)

        else:
            smpl_verts = smpl_outputs['smpl_verts']
            deformed_mesh_base = generate_mesh(lambda x: self.query_od(x, cond, smpl_tfs, smpl_verts), smpl_verts[0], type='sdf_base', point_batch=10000, res_up=3, device=self.device)

            if deformed_mesh_base is not None:
                deformed_mesh_base = trimesh.Trimesh(deformed_mesh_base.vertices, deformed_mesh_base.faces)
                verts_deformed_base = torch.tensor(deformed_mesh_base.vertices, device=self.device).float()
                verts_canonical_base = self.model.smpl_deformer(verts_deformed_base, smpl_tfs, smpl_verts)
                verts_canonical_base = verts_canonical_base.detach().cpu().numpy()
                canonical_mesh_base = trimesh.Trimesh(vertices=verts_canonical_base, faces=deformed_mesh_base.faces, process=False)

            deformed_mesh = generate_mesh(lambda x: self.query_od(x, cond, smpl_tfs, smpl_verts), smpl_verts[0], type='sdf', point_batch=10000, res_up=3, device=self.device)
            if deformed_mesh is not None:
                deformed_mesh = trimesh.Trimesh(deformed_mesh.vertices, deformed_mesh.faces)
                verts_deformed = torch.tensor(deformed_mesh.vertices, device=self.device).float()
                verts_canonical = self.model.smpl_deformer(verts_deformed, smpl_tfs, smpl_verts)
                verts_canonical = verts_canonical.detach().cpu().numpy()
                canonical_mesh = trimesh.Trimesh(vertices=verts_canonical, faces=deformed_mesh.faces, process=False)

        return canonical_mesh_base, deformed_mesh_base, canonical_mesh, deformed_mesh

    def get_save_meshes(self, inputs, base_name, path='rendering'):
        pose_axis = inputs["smpl_pose"].reshape(-1, 24, 3)
        pose6d = axisang_to_rot6d(pose_axis)
        pose6d = pose6d.reshape(1, -1)

        cond = {}
        cond['smpl_pose'] = pose6d
        cond['global_step'] = inputs['global_step']
        cond['current_epoch'] = inputs['current_epoch']

        canonical_mesh_base, deformed_mesh_base, canonical_mesh, deformed_mesh = self.generate_meshes(cond, inputs[
            'smpl_scale'], inputs['smpl_trans'], inputs['smpl_pose'], inputs['smpl_shape'])

        self.save_meshes({#'canonical_mesh': canonical_mesh,
                          'deformed_mesh': deformed_mesh,
                          #'deformed_mesh_base': deformed_mesh_base
                          },
                         base_name=base_name, path=path)

    def get_save_imgs(self, inputs, targets, base_name, path='rendering'):
        outputs = {}
        split = utils.split_input(inputs, targets["total_pixels"][0], device=self.device, n_pixels=min(targets['pixel_per_batch'], targets["img_size"][0] * targets["img_size"][1]))
        res = []

        for s in split:
            out = self.model(s)
            for k, v in out.items():
                try:
                    out[k] = v.detach()
                except:
                    out[k] = v
            if 'rgb_base' in out:
                res.append({
                    'rgb': out['rgb'].detach(),
                    'normal': out['normal'].detach(),
                    'fg_rgb': out['fg_rgb'].detach(),
                    'acc_map': out['acc_map'].detach(),
                    'rgb_base': out['rgb_base'].detach(),
                    'normal_base': out['normal_base'].detach(),
                    'acc_map_base': out['acc_map_base'].detach(),
                })
            else:
                res.append({
                    'rgb': out['rgb'].detach(),
                    'normal': out['normal'].detach(),
                    'fg_rgb': out['fg_rgb'].detach(),
                    'acc_map': out['acc_map'].detach()
                })

        batch_size = targets['rgb_gt'].shape[0]

        model_outputs = utils.merge_output(res, targets["total_pixels"][0], batch_size)

        if 'rgb_base' in model_outputs:
            outputs.update({
                "rgb": model_outputs["rgb"].detach().clone(),
                "normal": model_outputs["normal"].detach().clone(),
                "fg_rgb": model_outputs["fg_rgb"].detach().clone(),
                "acc_map": model_outputs["acc_map"].detach().clone(),
                "rgb_base": model_outputs["rgb_base"].detach().clone(),
                "normal_base": model_outputs["normal_base"].detach().clone(),
                "acc_map_base": model_outputs["acc_map_base"].detach().clone(),
                **targets,
            })
        else:
            outputs.update({
                "rgb": model_outputs["rgb"].detach().clone(),
                "normal": model_outputs["normal"].detach().clone(),
                "fg_rgb": model_outputs["fg_rgb"].detach().clone(),
                "acc_map": model_outputs["acc_map"].detach().clone(),
                **targets,
            })

        self.save_imgs(outputs, base_name=base_name, path=path)

    def save_boxes(self, boxes, verts, joints, idxs, path='box'):
        os.makedirs(path, exist_ok=True)

        faces = np.array([
            [0, 1, 2],  # front face
            [2, 1, 4],
            [0, 2, 3],  # back face
            [3, 2, 5],
            [3, 5, 6],  # left face
            [6, 5, 7],
            [3, 6, 1],  # right face
            [3, 1, 0],
            [1, 6, 7],  # top face
            [1, 7, 4],
            [4, 7, 5],  # bottom face
            [4, 5, 2]
        ])

        pcd = trimesh.PointCloud(verts)
        pcd.export(f"{path}/verts.ply")

        for i in range(len(boxes)):
            box = boxes[i]
            pcd = trimesh.Trimesh(vertices=box, faces=faces, process=False)
            pcd.export(f"{path}/box_{i}.ply")

    def save_meshes(self, meshes, base_name, path='rendering'):
        os.makedirs(path, exist_ok=True)
        for k, v in meshes.items():
            if v is not None:
                v.export(f"{path}/{base_name}_{k}.ply")

    def save_imgs(self, outputs, base_name, path='rendering'):
        os.makedirs(path, exist_ok=True)
        img_size = outputs["img_size"]
        rgb_bg = (torch.ones_like(outputs["rgb_gt"]) * outputs["bgcolor"][0]).reshape(-1, 3)
        ray_mask = outputs["ray_mask"][0].reshape(-1)

        rgb_pred_ray = outputs["rgb"]
        rgb_pred = rgb_bg.clone()
        rgb_pred[ray_mask] = rgb_pred_ray
        rgb_pred = rgb_pred.reshape(*img_size, -1)

        normal_pred_ray = outputs["normal"]

        normal_pred = torch.ones_like(rgb_bg)
        normal_pred[ray_mask] = normal_pred_ray

        normal_pred = normal_pred.reshape(*img_size, -1)

        pred_mask_ray = outputs["acc_map"]
        pred_mask = rgb_bg[..., 0].clone()
        pred_mask[ray_mask] = pred_mask_ray
        pred_mask = pred_mask.reshape(*img_size, -1)

        rgb_gt = outputs["rgb_gt"]
        rgb_gt = rgb_gt.reshape(*img_size, -1)

        rgb_combine = torch.cat([rgb_gt, rgb_pred, normal_pred, pred_mask.expand(-1, -1, 3)], dim=1)

        if 'rgb_base' in outputs:
            rgb_pred_ray = outputs["rgb_base"]
            rgb_pred = rgb_bg.clone()
            rgb_pred[ray_mask] = rgb_pred_ray
            rgb_pred = rgb_pred.reshape(*img_size, -1)

            normal_pred_ray = outputs["normal_base"]

            normal_pred = torch.ones_like(rgb_bg)
            normal_pred[ray_mask] = normal_pred_ray

            normal_pred = normal_pred.reshape(*img_size, -1)

            pred_mask_ray = outputs["acc_map_base"]
            pred_mask = rgb_bg[..., 0].clone()
            pred_mask[ray_mask] = pred_mask_ray
            pred_mask = pred_mask.reshape(*img_size, -1)

            rgb_base_combine = torch.cat([rgb_gt, rgb_pred, normal_pred, pred_mask.expand(-1, -1, 3)], dim=1)
            rgb_combine = torch.cat([rgb_combine, rgb_base_combine], dim=0)

        rgb_combine = (255. * np.clip(rgb_combine.cpu().numpy(), 0., 1.)).astype(np.uint8)
        cv2.imwrite(f"{path}/{base_name}.png", rgb_combine[:, :, ::-1])





