"""
Reconstruction evaluator.

Reports metrics for the base mesh and final reconstruction:
- AbsRel (%) over sampled frames
- Accuracy (m)
- Completeness (m)
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pycolmap

from s1_sfm import CameraModel, Pose, parse_reconstruction

logger = logging.getLogger(__name__)

@dataclass
class SyntheticFrame:
    image_name: str
    R: np.ndarray
    t: np.ndarray
    depth_z_path: Path

    def pose(self) -> Pose:
        return Pose(R=self.R, t=self.t)

    def camera_center(self) -> np.ndarray:
        return (-self.R.T @ self.t).ravel()

@dataclass
class SyntheticDataset:
    root: Path
    camera_model: CameraModel
    frames: list

    @property
    def gt_dir(self) -> Path:
        return self.root / "gt"

    @property
    def mesh_path(self) -> Path:
        return self.gt_dir / "scene_mesh.ply"

def load_synthetic_dataset(dataset_root: Path) -> SyntheticDataset:
    root = Path(dataset_root)
    gt_dir = root / "gt"
    with open(gt_dir / "poses.json") as f:
        manifest = json.load(f)
    K = np.load(gt_dir / manifest["intrinsics_path"])
    camera_model = CameraModel(K=K, width=int(manifest["width"]), height=int(manifest["height"]))
    frames = [
        SyntheticFrame(
            image_name=fr["image_name"],
            R=np.asarray(fr["R"], dtype=np.float64),
            t=np.asarray(fr["t"], dtype=np.float64),
            depth_z_path=root / fr["depth_z_path"],
        )
        for fr in manifest["frames"]
    ]
    return SyntheticDataset(root=root, camera_model=camera_model, frames=frames)

def estimate_sim3(est_points: np.ndarray, gt_points: np.ndarray):
    mu_est = est_points.mean(axis=0)
    mu_gt = gt_points.mean(axis=0)
    est_centered = est_points - mu_est
    gt_centered = gt_points - mu_gt
    cov = (gt_centered.T @ est_centered) / len(est_points)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3, dtype=np.float64)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt
    var_est = np.mean(np.sum(est_centered**2, axis=1))
    scale = np.sum(D * np.diag(S)) / max(var_est, 1e-12)
    t = mu_gt - scale * (R @ mu_est)
    return float(scale), R, t

def load_gt_depth(depth_path: Path, camera: CameraModel) -> np.ndarray:
    """Load GT depth from either PNG millimeters or NPY meters."""
    if depth_path.suffix.lower() == ".npy":
        gt_depth = np.load(depth_path).astype(np.float64)
    else:
        gt_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if gt_raw is None:
            raise FileNotFoundError(f"Could not read GT depth: {depth_path}")
        gt_depth = gt_raw.astype(np.float64) / 5000.0
    if gt_depth.shape != (camera.height, camera.width):
        gt_depth = cv2.resize(gt_depth, (camera.width, camera.height), interpolation=cv2.INTER_NEAREST)
    return gt_depth

def render_z_depth(scene: o3d.t.geometry.RaycastingScene, pose: Pose, camera: CameraModel) -> np.ndarray:
    """Render camera-frame z-depth from a mesh scene."""
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = pose.R
    extrinsic[:3, 3] = pose.t.ravel()
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intrinsic_matrix=o3d.core.Tensor(camera.K.astype(np.float64)),
        extrinsic_matrix=o3d.core.Tensor(extrinsic),
        width_px=camera.width,
        height_px=camera.height,
    )
    result = scene.cast_rays(rays)
    t_hit = result["t_hit"].numpy().astype(np.float64)
    rays_np = rays.numpy().astype(np.float64)
    origins = rays_np[..., :3]
    directions = rays_np[..., 3:6]
    hit_points = origins + directions * t_hit[..., None]
    hit_flat = hit_points.reshape(-1, 3)
    hit_cam = pose.world_to_camera(hit_flat).reshape(hit_points.shape)
    z_depth = hit_cam[..., 2]
    z_depth[np.isinf(t_hit)] = 0.0
    z_depth[np.isnan(z_depth)] = 0.0
    return z_depth

def load_context(workspace: Path, dataset):
    """Load reconstruction and compute Sim(3) alignment to GT poses."""
    recon_path = workspace / "dense" / "sparse"
    image_dir = workspace / "dense" / "images"
    recon = pycolmap.Reconstruction(str(recon_path))
    sfm = parse_reconstruction(recon, image_dir, max_reproj_error=4.0, min_track_length=3)
    keyframes_by_name = {kf.image_name: kf for kf in sfm.keyframes}

    frames_by_name = {f.image_name: f for f in dataset.frames}
    est_centers, gt_centers = [], []
    for image in recon.images.values():
        if not image.has_pose or image.name not in frames_by_name:
            continue
        rigid = image.cam_from_world()
        R = rigid.rotation.matrix()
        t = rigid.translation
        est_centers.append(-R.T @ t)
        gt_centers.append(frames_by_name[image.name].camera_center())
    scale, rot, trans = estimate_sim3(np.asarray(est_centers), np.asarray(gt_centers))
    return sfm, keyframes_by_name, scale, rot, trans

def load_mesh_and_scene(mesh_path: Path):
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty() or len(mesh.triangles) == 0:
        raise RuntimeError(f"Empty mesh: {mesh_path}")
    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh()
    mesh_t.vertex.positions = o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32))
    mesh_t.triangle.indices = o3d.core.Tensor(np.asarray(mesh.triangles, dtype=np.int32))
    scene.add_triangles(mesh_t)
    return mesh, scene

def select_sample_names(sfm, frames_by_name, n_frames):
    matched = [kf.image_name for kf in sorted(sfm.keyframes, key=lambda k: k.image_name)
               if kf.image_name in frames_by_name]
    n_pick = min(n_frames, len(matched))
    idx = np.linspace(0, len(matched) - 1, n_pick, dtype=int)
    return [matched[i] for i in idx]

def compute_absrel(mesh_scene, keyframes_by_name, frames_by_name, camera_model, scale, sample_names):
    all_rel = []
    for name in sample_names:
        kf = keyframes_by_name.get(name)
        gt_frame = frames_by_name.get(name)
        if kf is None or gt_frame is None:
            continue
        rendered_z = render_z_depth(mesh_scene, kf.pose, camera_model) * scale
        gt_depth = load_gt_depth(gt_frame.depth_z_path, camera_model)
        valid = (rendered_z > 0) & (gt_depth > 0)
        if not valid.any():
            continue
        rel = np.abs(rendered_z[valid] - gt_depth[valid]) / np.maximum(gt_depth[valid], 1e-12)
        all_rel.append(rel)
    if not all_rel:
        return float("nan")
    return float(np.mean(np.concatenate(all_rel))) * 100.0

def compute_surface(mesh, gt_mesh, scale, rot, trans, sample_points=50000):
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    transformed = scale * (rot @ vertices.T).T + trans
    aligned = o3d.geometry.TriangleMesh(mesh)
    aligned.vertices = o3d.utility.Vector3dVector(transformed)

    pred_pcd = aligned.sample_points_uniformly(number_of_points=sample_points)
    gt_pcd = gt_mesh.sample_points_uniformly(number_of_points=sample_points)
    pred_to_gt = np.asarray(pred_pcd.compute_point_cloud_distance(gt_pcd))
    gt_to_pred = np.asarray(gt_pcd.compute_point_cloud_distance(pred_pcd))
    return float(np.mean(pred_to_gt)), float(np.mean(gt_to_pred))

def evaluate_mesh(name, mesh_path, sfm, keyframes_by_name, frames_by_name,
                  scale, rot, trans, gt_mesh, sample_names):
    mesh, mesh_scene = load_mesh_and_scene(mesh_path)
    absrel = compute_absrel(mesh_scene, keyframes_by_name, frames_by_name,
                            sfm.camera_model, scale, sample_names)
    accuracy, completeness = compute_surface(mesh, gt_mesh, scale, rot, trans)
    print()
    print(f"{name} ({mesh_path.name})")
    print(f"AbsRel: {absrel:.2f} %")
    print(f"Accuracy: {accuracy:.4f} m")
    print(f"Completeness: {completeness:.4f} m")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", help="Workspace directory")
    parser.add_argument("--dataset", required=True, help="Synthetic dataset root")
    parser.add_argument("--frames", type=int, default=10, help="Number of frames for depth eval")
    args = parser.parse_args()

    workspace = Path(args.workspace)
    dataset = load_synthetic_dataset(Path(args.dataset))
    frames_by_name = {f.image_name: f for f in dataset.frames}

    sfm, keyframes_by_name, scale, rot, trans = load_context(workspace, dataset)
    sample_names = select_sample_names(sfm, frames_by_name, args.frames)

    gt_mesh_path = dataset.gt_dir / "foreground_mesh.ply"
    if not gt_mesh_path.exists():
        gt_mesh_path = dataset.mesh_path
    gt_mesh = o3d.io.read_triangle_mesh(str(gt_mesh_path))

    methods = []
    base_path = workspace / "initial_textured_poisson.ply"
    if base_path.exists():
        methods.append(("Base Mesh", base_path))
    final_path = workspace / "reconstruction.ply"
    if final_path.exists():
        methods.append(("Final Reconstruction", final_path))
    if not methods:
        raise FileNotFoundError(f"No meshes found in {workspace}")

    for name, mesh_path in methods:
        evaluate_mesh(name, mesh_path, sfm, keyframes_by_name, frames_by_name, scale, rot, trans, gt_mesh, sample_names)

if __name__ == "__main__":
    logging.basicConfig(format="[%(filename)s:%(lineno)d] %(message)s", level=logging.WARNING)
    main()
