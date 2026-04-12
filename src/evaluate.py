"""
Reconstruction evaluator.

Reports metrics for the base mesh and final reconstruction:
- AbsRel (%) over overlapping valid pixels in sampled frames
- Depth overlap coverage (%) against valid GT pixels
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

    @property
    def foreground_mesh_path(self) -> Path:
        return self.gt_dir / "foreground_mesh.ply"

@dataclass
class DepthMetrics:
    absrel_pct: float
    overlap_pct: float
    overlap_pixels: int
    gt_valid_pixels: int

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

def load_gt_depth(depth_path: Path, camera: CameraModel, png_depth_scale: float = 5000.0) -> np.ndarray:
    """Load GT z-depth from NPY meters or TUM-style PNG depth / scale."""
    if depth_path.suffix.lower() == ".npy":
        gt_depth = np.load(depth_path).astype(np.float64)
    else:
        gt_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if gt_raw is None:
            raise FileNotFoundError(f"Could not read GT depth: {depth_path}")
        gt_depth = gt_raw.astype(np.float64) / png_depth_scale
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
    if len(est_centers) < 3:
        raise RuntimeError(
            f"Need at least 3 matched camera poses for Sim(3) alignment; got {len(est_centers)}."
        )
    scale, rot, trans = estimate_sim3(np.asarray(est_centers), np.asarray(gt_centers))
    return sfm, scale, rot, trans

def mesh_to_scene(mesh: o3d.geometry.TriangleMesh):
    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh()
    mesh_t.vertex.positions = o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32))
    mesh_t.triangle.indices = o3d.core.Tensor(np.asarray(mesh.triangles, dtype=np.int32))
    scene.add_triangles(mesh_t)
    return scene

def load_mesh(mesh_path: Path):
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty() or len(mesh.triangles) == 0:
        raise RuntimeError(f"Empty mesh: {mesh_path}")
    return mesh

def align_mesh_sim3(mesh: o3d.geometry.TriangleMesh, scale: float, rot: np.ndarray, trans: np.ndarray):
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    transformed = scale * (rot @ vertices.T).T + trans
    aligned = o3d.geometry.TriangleMesh(mesh)
    aligned.vertices = o3d.utility.Vector3dVector(transformed)
    return aligned

def select_sample_names(sfm, frames_by_name, n_frames):
    if n_frames <= 0:
        raise ValueError("--frames must be positive")
    matched = [kf.image_name for kf in sorted(sfm.keyframes, key=lambda k: k.image_name)
               if kf.image_name in frames_by_name]
    if not matched:
        raise RuntimeError("No registered SfM keyframes match the synthetic GT manifest.")
    n_pick = min(n_frames, len(matched))
    idx = np.linspace(0, len(matched) - 1, n_pick, dtype=int)
    return [matched[i] for i in idx]

def compute_absrel(mesh_scene, frames_by_name, camera_model, sample_names, gt_depth_scene=None):
    all_rel = []
    overlap_pixels = 0
    gt_valid_pixels = 0
    for name in sample_names:
        gt_frame = frames_by_name.get(name)
        if gt_frame is None:
            continue
        rendered_z = render_z_depth(mesh_scene, gt_frame.pose(), camera_model)
        if gt_depth_scene is None:
            gt_depth = load_gt_depth(gt_frame.depth_z_path, camera_model)
        else:
            gt_depth = render_z_depth(gt_depth_scene, gt_frame.pose(), camera_model)
        gt_valid = gt_depth > 0
        valid = (rendered_z > 0) & gt_valid
        gt_valid_pixels += int(gt_valid.sum())
        overlap_pixels += int(valid.sum())
        if not valid.any():
            continue
        rel = np.abs(rendered_z[valid] - gt_depth[valid]) / np.maximum(gt_depth[valid], 1e-12)
        all_rel.append(rel)
    if not all_rel:
        absrel = float("nan")
    else:
        absrel = float(np.mean(np.concatenate(all_rel))) * 100.0
    overlap = overlap_pixels / gt_valid_pixels * 100.0 if gt_valid_pixels else float("nan")
    return DepthMetrics(
        absrel_pct=absrel,
        overlap_pct=float(overlap),
        overlap_pixels=overlap_pixels,
        gt_valid_pixels=gt_valid_pixels,
    )

def compute_surface(aligned_mesh, gt_mesh, sample_points=50000):
    pred_pcd = aligned_mesh.sample_points_uniformly(number_of_points=sample_points)
    gt_pcd = gt_mesh.sample_points_uniformly(number_of_points=sample_points)
    pred_to_gt = np.asarray(pred_pcd.compute_point_cloud_distance(gt_pcd))
    gt_to_pred = np.asarray(gt_pcd.compute_point_cloud_distance(pred_pcd))
    return float(np.mean(pred_to_gt)), float(np.mean(gt_to_pred))

def evaluate_mesh(name, mesh_path, frames_by_name, camera_model,
                  scale, rot, trans, depth_targets, surface_targets, sample_names):
    mesh = load_mesh(mesh_path)
    aligned_mesh = align_mesh_sim3(mesh, scale, rot, trans)
    mesh_scene = mesh_to_scene(aligned_mesh)
    print()
    print(f"{name} ({mesh_path.name})")

    for target_name, gt_depth_scene in depth_targets:
        depth = compute_absrel(mesh_scene, frames_by_name, camera_model, sample_names, gt_depth_scene)
        print(f"Depth [{target_name}] AbsRel (overlap): {depth.absrel_pct:.2f} %")
        print(
            f"Depth [{target_name}] overlap: "
            f"{depth.overlap_pct:.2f} % ({depth.overlap_pixels}/{depth.gt_valid_pixels} GT-valid pixels)"
        )

    for target_name, gt_mesh in surface_targets:
        accuracy, completeness = compute_surface(aligned_mesh, gt_mesh)
        print(f"Surface [{target_name}] Accuracy: {accuracy:.4f} m")
        print(f"Surface [{target_name}] Completeness: {completeness:.4f} m")

def selected_targets(selection: str) -> list[str]:
    if selection == "both":
        return ["foreground", "scene"]
    return [selection]

def resolve_gt_mesh_paths(dataset: SyntheticDataset, surface_gt: str) -> list[tuple[str, Path]]:
    paths = []
    for target in selected_targets(surface_gt):
        if target == "foreground":
            path = dataset.foreground_mesh_path
            if not path.exists():
                raise FileNotFoundError(f"Foreground GT mesh not found: {path}")
        else:
            path = dataset.mesh_path
        paths.append((target, path))
    return paths

def load_surface_targets(dataset: SyntheticDataset, surface_gt: str):
    targets = []
    for target_name, path in resolve_gt_mesh_paths(dataset, surface_gt):
        mesh = o3d.io.read_triangle_mesh(str(path))
        if mesh.is_empty() or len(mesh.triangles) == 0:
            raise RuntimeError(f"Empty GT mesh: {path}")
        targets.append((target_name, mesh))
    return targets

def resolve_depth_gt_scene(dataset: SyntheticDataset, depth_gt: str):
    if depth_gt == "scene":
        return None
    if depth_gt == "foreground":
        path = dataset.foreground_mesh_path
        if not path.exists():
            raise FileNotFoundError(f"Foreground GT mesh not found: {path}")
        mesh = load_mesh(path)
        return mesh_to_scene(mesh)
    raise ValueError(f"Unknown depth GT target: {depth_gt}")

def load_depth_targets(dataset: SyntheticDataset, depth_gt: str):
    return [(target, resolve_depth_gt_scene(dataset, target)) for target in selected_targets(depth_gt)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", help="Workspace directory")
    parser.add_argument("--dataset", required=True, help="Synthetic dataset root")
    parser.add_argument("--frames", type=int, default=10, help="Number of frames for depth eval")
    parser.add_argument(
        "--surface-gt",
        choices=("scene", "foreground", "both"),
        default="both",
        help="GT mesh used for surface metrics.",
    )
    parser.add_argument(
        "--depth-gt",
        choices=("scene", "foreground", "both"),
        default="both",
        help="GT geometry used for depth metrics.",
    )
    args = parser.parse_args()

    workspace = Path(args.workspace)
    dataset = load_synthetic_dataset(Path(args.dataset))
    frames_by_name = {f.image_name: f for f in dataset.frames}

    sfm, scale, rot, trans = load_context(workspace, dataset)
    sample_names = select_sample_names(sfm, frames_by_name, args.frames)

    depth_targets = load_depth_targets(dataset, args.depth_gt)
    surface_targets = load_surface_targets(dataset, args.surface_gt)
    print(f"Depth GT: {', '.join(name for name, _ in depth_targets)}")
    print(f"Surface GT: {', '.join(name for name, _ in surface_targets)}")

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
        evaluate_mesh(
            name,
            mesh_path,
            frames_by_name,
            dataset.camera_model,
            scale,
            rot,
            trans,
            depth_targets,
            surface_targets,
            sample_names,
        )

if __name__ == "__main__":
    logging.basicConfig(format="[%(filename)s:%(lineno)d] %(message)s", level=logging.WARNING)
    main()
