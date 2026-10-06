"""SMPL-X fitting to COCO-17 keypoints (SMPLify-style), then colour by projecting the source image."""
import numpy as np
import torch
import smplx
import trimesh
from scipy.spatial.transform import Rotation as R

from config import SMPLX_DIR
from human.texture import colorize

COCO = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]   # shoulders, elbows, wrists, hips, knees, ankles
SMPLX_J = [16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]  # matching SMPL-X joint indices
TORSO = [0, 1, 6, 7]
_model = None


def _get(dev):
    global _model
    if _model is None:
        _model = smplx.create(str(SMPLX_DIR), model_type="smplx", gender="neutral", use_pca=False,
                              flat_hand_mean=True, num_betas=10, batch_size=1).to(dev)
        for p in _model.parameters():
            p.requires_grad_(False)
    return _model


def _fit(model, kp, cf, W, H, bbox, yaw, dev):
    f, c = 1.2 * max(W, H), torch.tensor([W / 2, H / 2], device=dev)
    bh = max(bbox[3] - bbox[1], 1.0)
    k = torch.tensor(np.array(kp)[COCO], dtype=torch.float32, device=dev)
    cfa = np.array(cf)[COCO]
    w = torch.tensor(np.where(cfa > 0.3, cfa, 0.0), dtype=torch.float32, device=dev)
    rot0 = (R.from_rotvec([np.pi, 0, 0]) * R.from_rotvec([0, yaw, 0])).as_rotvec()
    z0 = f * 1.7 / bh
    cxb, cyb = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    orient = torch.tensor([rot0], dtype=torch.float32, device=dev, requires_grad=True)
    transl = torch.tensor([[(cxb - W / 2) * z0 / f, (cyb - H / 2) * z0 / f, z0]], dtype=torch.float32, device=dev, requires_grad=True)
    pose = torch.zeros(1, 63, device=dev, requires_grad=True)
    betas = torch.zeros(1, 10, device=dev, requires_grad=True)

    def fwd(verts=False):
        return model(betas=betas, global_orient=orient, body_pose=pose, transl=transl, return_verts=verts)

    def loss_fn(idx, pw):
        J = fwd().joints[0, :22][SMPLX_J]
        uv = f * J[:, :2] / J[:, 2:3].clamp(min=0.3) + c
        err = (uv[idx] - k[idx]) / bh * 10
        knee = torch.relu(-pose[0, [9, 12]]).sum()  # no knee hyperextension
        return (w[idx, None] * err ** 2).sum() + pw * pose.pow(2).sum() + 0.05 * betas.pow(2).sum() + 5 * knee

    for params, lr, iters, idx, pw in [([orient, transl], 0.02, 150, TORSO, 0.0),
                                       ([orient, transl, pose, betas], 0.01, 300, list(range(12)), 0.05)]:
        opt = torch.optim.Adam(params, lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            loss = loss_fn(idx, pw)
            loss.backward()
            opt.step()
    with torch.no_grad():
        final = float(loss_fn(list(range(12)), 0.05))
        out = fwd(True)
    return final, out.vertices[0].cpu().numpy().astype(np.float64), f


def reconstruct_person(bgr, mask, kp, kpc, bbox, out_path, device="cpu"):
    H, W = bgr.shape[:2]
    model = _get(device)
    best = min((_fit(model, kp, kpc, W, H, bbox, yaw, device) for yaw in (0.0, np.pi)), key=lambda t: t[0])
    _, verts, f = best
    mesh = trimesh.Trimesh(verts, model.faces.astype(np.int64), process=False).subdivide()
    colors = colorize(mesh, bgr, mask, f, W / 2, H / 2)
    v = mesh.vertices * np.array([1, -1, -1])  # camera frame (y down, z forward) -> glTF (y up, faces +z)
    trimesh.Trimesh(v, mesh.faces, vertex_colors=colors, process=False).export(str(out_path))
