"""
Wraps pycolmap to produce SfM outputs (Section 2.2)

Needs to produce:
- Camera intrinsics K and image dimensions (pinhole model)
- Per-keyframe extrinsics (R, t) with projection ultilties
- Filtered sparse 3D point close with estimated surface normals
- Handle distortion (all subsequent steps assume undistorted image)

Pose convention:
- WORLD-TO-CAMERA (cam_from_world):
    x_cam = R @ x_world + t
- Camera center in world coordinates:
    C = -R^T @ t
- Normals point toward the camera (outward from surface).
"""

import numpy as np
import pycolmap
import logging
from dataclasses import dataclass
from pathlib import Path
import sys

logger = logging.getLogger(__name__)

@dataclass
class CameraModel:
    K: np.ndarray # 3x3 intrinsic matrix
    width: int
    height: int

@dataclass
class Pose:
    """World-to-camera rigid transfrom: x_cam = R @ x_world + t"""
    R: np.ndarray # 3x3 rotation
    t: np.ndarray # 3x1 translation

    def camera_center(self) -> np.ndarray:
        """Camera center in world coordinates: C = -R^T @ t.  Returns (3,)."""
        return (-self.R.T @ self.t).ravel()

    def world_to_camera(self, X: np.ndarray) -> np.ndarray:
        """Transform (3,) or (N, 3) world points into camera frame."""
        t_flat = self.t.ravel()
        if X.ndim == 1:
            return self.R @ X + t_flat
        return (self.R @ X.T).T + t_flat

    def camera_to_world(self, X_cam: np.ndarray) -> np.ndarray:
        """Transform (3,) or (N, 3) camera points into world frame."""
        t_flat = self.t.ravel()
        if X_cam.ndim == 1:
            return self.R.T @ (X_cam - t_flat)
        return (self.R.T @ (X_cam - t_flat).T).T

    def projection_matrix(self, K: np.ndarray) -> np.ndarray:
        """3x4 projection matrix P = K [R | t]"""
        return K @ np.hstack([self.R, self.t])

@dataclass
class Observation:
    """One observation of a 3D point in a keyframe."""
    image_id: int
    point2D_index: int

@dataclass
class SparsePoint:
    """A 3D point."""
    point_id: int
    xyz: np.ndarray # (3,) coordinates in world frame
    color: np.ndarray # RGB color of point
    normal: np.ndarray # (3,) estimated surface normal
    observations: list[Observation]
    reproj_error: float
    track_length: int

@dataclass
class Keyframe:
    """A registered camera pose / image."""
    image_id: int
    frame_index: int # sequential index in temporal order
    image_name: str
    image_path: str
    pose: Pose
    P: np.ndarray # 3x4 projection matrix

@dataclass
class SfMResult:
    """All SfM outputs."""
    camera_model: CameraModel
    keyframes: list[Keyframe] # sorted by frame_index
    sparse_points: list[SparsePoint]

def run_colmap(image_dir: Path, workspace_dir: Path) -> tuple[pycolmap.Reconstruction, Path]:
    """Runs COLMAP SfM pipeline."""
    db_path = workspace_dir / "colmap.db"
    sparse_dir = workspace_dir / "sparse"
    dense_dir = workspace_dir / "dense"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    dense_dir.mkdir(parents=True, exist_ok=True)

    if db_path.exists():
        db_path.unlink()
        logger.info(f"Cleared stale COLMAP database: {db_path}")

    # Feature extraction
    logger.info("Extracting features...")
    pycolmap.extract_features(
        database_path=db_path,
        image_path=image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
    )

    # Sequential matching
    logger.info("Matching features...")
    pycolmap.match_sequential(database_path=db_path)

    # Incremental SfM
    logger.info("Running incremental mapper...")
    reconstructions = pycolmap.incremental_mapping(
        database_path=db_path,
        image_path=image_dir,
        output_path=sparse_dir,
    )

    if not reconstructions:
        raise RuntimeError("COLMAP failed to produce any reconstructions.")

    recon = max(reconstructions.values(), key=lambda r: r.num_reg_images())
    logger.info(f"Reconstruction: {recon.num_reg_images()} images, {recon.num_points3D()} points")

    # Undistort images
    logger.info("Undistorting images...")
    recon_path = sparse_dir / "0"
    recon_path.mkdir(parents=True, exist_ok=True)
    recon.write(recon_path)

    pycolmap.undistort_images(
        output_path=dense_dir,
        input_path=recon_path,
        image_path=image_dir,
    )

    # Load undistorted reconstruction
    undist_recon = pycolmap.Reconstruction(dense_dir / "sparse")
    undist_image_dir = dense_dir / "images"

    logger.info(f"Undistorted images at {undist_image_dir}")
    return undist_recon, undist_image_dir

def _assign_frame_indices(keyframes: list[Keyframe]) -> None:
    """Sort keyframes by filename and assign sequential frame indices."""
    keyframes.sort(key=lambda k: k.image_name)
    for i, k in enumerate(keyframes):
        k.frame_index = i

def _estimate_normals(points_xyz: np.ndarray, point_obs: list[list[int]], view_dirs: dict[int, np.ndarray]) -> np.ndarray:
    """Estimate per-point surface normals by averaging view directions of all camera that observe each point."""
    normals = np.zeros_like(points_xyz)
    for i, obs_ids in enumerate(point_obs):
        acc = np.zeros(3)
        for img_id in obs_ids:
            acc -= view_dirs[img_id]
        norm = np.linalg.norm(acc)
        normals[i] = acc / norm if norm > 1e-8 else np.array([0.0, 0.0, 1.0])
    return normals

def parse_reconstruction(recon: pycolmap.Reconstruction, image_dir: Path, max_reproj_error=4.0, min_track_length=3) -> SfMResult:
    """Convert a pycolmap reconstruction proper structures"""
    if recon.num_cameras() != 1:
        raise RuntimeError(f"Expected exactly 1 camera model, but got {recon.num_cameras()}")
    cam_id, cam = next(iter(recon.cameras.items()))
    K = cam.calibration_matrix()

    camera_model = CameraModel(K=K, width=cam.width, height=cam.height)
    logger.info(f"Camera: {cam.model_name} {cam.width}x{cam.height}, f=({K[0,0]:.1f}, {K[1,1]:.1f}), c=({K[0,2]:.1f}, {K[1,2]:.1f})")

    # Keyframes
    keyframes: list[Keyframe] = []
    view_dirs: dict[int, np.ndarray] = {}

    for image_id, image in recon.images.items():
        if not image.has_pose:
            continue

        rigid = image.cam_from_world()
        R = rigid.rotation.matrix()
        t = rigid.translation.reshape(3, 1)
        pose = Pose(R=R, t=t)

        keyframes.append(Keyframe(
            image_id=image_id,
            frame_index=-1, # to be assigned later
            image_name=image.name,
            image_path=image_dir / image.name,
            pose=pose,
            P=pose.projection_matrix(K),
        ))

        view_dirs[image_id] = np.asarray(image.viewing_direction()).ravel()

    _assign_frame_indices(keyframes)

    # Point cloud filtering
    sparse_points: list[SparsePoint] = []
    obs_image_ids: list[list[int]] = []

    for point_id, point in recon.points3D.items():
        track = point.track
        if track.length() < min_track_length:
            continue
        if point.error > max_reproj_error:
            continue

        observations = [Observation(image_id=elem.image_id, point2D_index=elem.point2D_idx) for elem in track.elements]
        obs_image_ids.append([o.image_id for o in observations])

        sparse_points.append(SparsePoint(
            point_id=point_id,
            xyz=point.xyz.copy(),
            color=point.color.copy(),
            normal=np.zeros(3), # to be estimated later
            observations=observations,
            reproj_error=point.error,
            track_length=track.length(),
        ))
    
    # Surface normals
    if sparse_points:
        points_xyz = np.array([p.xyz for p in sparse_points])
        normals = _estimate_normals(points_xyz, obs_image_ids, view_dirs)
        for p, n in zip(sparse_points, normals):
            p.normal = n
    
    logger.info(f"Kept {len(sparse_points)} points after filtering by reproj error < {max_reproj_error}, track length >= {min_track_length}")

    return SfMResult(camera_model=camera_model, keyframes=keyframes, sparse_points=sparse_points)

def run_sfm(image_dir: str | Path, workspace_dir: str | Path = "", max_reproj_error=2.0, min_track_length=5) -> SfMResult:
    """Runs the full SfM pipeline and returns structured results."""

    image_dir = Path(image_dir)
    workspace = Path(workspace_dir)
    workspace.mkdir(parents=True, exist_ok=True)

    if not any(image_dir.iterdir()):
        raise FileNotFoundError(f"No images found in {image_dir}")

    recon, undist_image_dir = run_colmap(image_dir, workspace)
    return parse_reconstruction(recon, undist_image_dir, max_reproj_error, min_track_length)

if __name__ == "__main__":
    logging.basicConfig(
        format="[%(filename)s:%(lineno)d:%(funcName)s] %(message)s",
        level=logging.DEBUG,
    )
    if len(sys.argv) < 2:
        logger.error("Usage: python s1_sfm.py <image_dir>")
        sys.exit(1)
    result = run_sfm(sys.argv[1])
    logger.info(f"{len(result.keyframes)} keyframes, {len(result.sparse_points)} 3D points")
    logger.info("Done")
