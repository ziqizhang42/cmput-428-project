"""
Base mesh (Section 2.3)

Fits a smooth surface to the sparse point cloud from SfM
Provides dpeth-map rendering for view prediction

- Poisson reconstruction with depth smoothing
"""

import numpy as np
import open3d as o3d
import sys
import cv2
import pycolmap
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass
from pathlib import Path
import logging

from s1_sfm import SfMResult, CameraModel, Pose, Keyframe, parse_reconstruction

logger = logging.getLogger(__name__)

@dataclass
class BaseMesh:
    """From SfM"""
    vertices: np.ndarray # (V, 3) world-frame positions
    faces: np.ndarray # (F, 3) triangle vertex indices
    normals: np.ndarray # (V, 3) per-vertex normals

    # Raycasting scene, built once and reused
    _scene: o3d.t.geometry.RaycastingScene = None

# Currently poisson construction
def build_base_mesh(sfm_result: SfMResult, octree_depth: int, density_quantile: float) -> BaseMesh:
    """Build a base surface mesh from the sparse SfM point cloud."""
    points = np.array([p.xyz for p in sfm_result.sparse_points], dtype=np.float64)
    normals = np.array([p.normal for p in sfm_result.sparse_points], dtype=np.float64)

    # Sanitize: remove non-finite and zero-length normals
    mask = np.isfinite(points).all(axis=1) & np.isfinite(normals).all(axis=1)
    nrm = np.linalg.norm(normals, axis=1)
    mask &= nrm > 1e-8
    points = points[mask]
    normals = normals[mask]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    pcd, inlier_indices = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    points = np.asarray(pcd.points)  # keep points in sync for bbox computation later
    normals = np.asarray(pcd.normals)
    logger.info(f"After outlier removal: {len(points)} points remaining")

    logger.info(f"Input: {len(points)} oriented points (after sanitization)")

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=octree_depth, n_threads=1, scale =1.1, linear_fit=True
    )
    densities = np.asarray(densities)

    logger.info(f"Poisson: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")

    threshold = np.quantile(densities, density_quantile)
    mask = densities > threshold
    mesh.remove_vertices_by_mask(~mask)

    logger.info(f"After density trim ({density_quantile:.0%} quantile): {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")

    # Crop to padded SfM bounding box (remove hallucinated geometry)
    padding = 0.1 # 10% margin
    pts_min = points.min(axis=0)
    pts_max = points.max(axis=0)
    extent = pts_max - pts_min
    bbox_min = pts_min - padding * extent
    bbox_max = pts_max + padding * extent
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox_min, max_bound=bbox_max)
    mesh = mesh.crop(bbox)

    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    largest_cluster = cluster_n_triangles.argmax()
    mesh.remove_triangles_by_mask(triangle_clusters != largest_cluster)
    mesh.remove_unreferenced_vertices()
    logger.info(f"After component removal: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")

    logger.info(f"After bbox crop: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")

    # Reduces the jagged "octree" look without causing the object to shrink.
    mesh = mesh.filter_smooth_taubin(number_of_iterations=20)

    mesh.orient_triangles()

    mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.vertex_normals)

    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh()
    mesh_t.vertex.positions = o3d.core.Tensor(vertices.astype(np.float32))
    mesh_t.triangle.indices = o3d.core.Tensor(faces.astype(np.int32))
    scene.add_triangles(mesh_t)

    return BaseMesh(vertices=vertices, faces=faces, normals=normals, _scene=scene)

def render_depth(base_mesh: BaseMesh, pose: Pose, camera: CameraModel, return_z = False) -> np.ndarray:
    """
    Render the base mesh from a given viewpoint.
    Returns an (H, W) depth map where each pixel holds the distance along the viewing ray to the surface.
    Pixels with no hits are set to 0.
    """
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = pose.R
    extrinsic[:3, 3] = pose.t.ravel()

    K = camera.K.astype(np.float64)

    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intrinsic_matrix=o3d.core.Tensor(K),
        extrinsic_matrix=o3d.core.Tensor(extrinsic),
        width_px=camera.width,
        height_px=camera.height,
    )

    result = base_mesh._scene.cast_rays(rays)
    t_hit = result['t_hit'].numpy().astype(np.float64)

    # Compute metric Euclidean depth from hit points
    # t_hit scales along (possibly non-unit) ray directions
    rays_np = rays.numpy().astype(np.float64)
    origins = rays_np[..., :3]
    directions = rays_np[..., 3:6]

    # reconsturct hit positions and measure distance from camera center
    hit_points = origins + directions * t_hit[..., None]
    C = pose.camera_center().reshape(1, 1, 3)
    depth = np.linalg.norm(hit_points - C, axis=-1)
    depth[np.isinf(t_hit)] = 0.0

    if return_z == True:
        # Convert hit points to camera coordinates
        cam_points = np.einsum('ij,hwj->hwi', pose.R, hit_points - C)

        # get the value of Z
        Z_depth = cam_points[..., 2]

        Z_depth[np.isinf(t_hit)] = 0.0

        return depth, Z_depth

    return depth

def smooth_depth(depth: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth a depth map (Section 2.5.2)."""
    valid = depth > 0
    if not np.any(valid):
        return depth.copy()
    
    weights = valid.astype(np.float64)
    smoothed_depth = gaussian_filter(depth * weights, sigma=sigma)
    smoothed_weights = gaussian_filter(weights, sigma=sigma)

    safe = smoothed_weights > 1e-8
    result = np.zeros_like(depth)
    result[safe] = smoothed_depth[safe] / smoothed_weights[safe]

    result[~valid] = 0.0

    return result

def texture_mesh(mesh: BaseMesh, sfm_result: SfMResult) -> np.ndarray:
    """Computes per-vertex RGB colors by averaging samples from all visible keyframes."""
    num_verts = len(mesh.vertices)
    vertex_colors = np.zeros((num_verts, 3), dtype=np.float32)
    counts = np.zeros(num_verts, dtype=np.float32)

    K = sfm_result.camera_model.K
    h, w = sfm_result.camera_model.height, sfm_result.camera_model.width

    for kf in sfm_result.keyframes:
        img = cv2.imread(kf.image_path)
        if img is None: continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Project all vertices into this camera
        # x_cam = R @ X_world + t
        pts_cam = (kf.pose.R @ mesh.vertices.T + kf.pose.t).T
        
        # Only consider points in front of the camera
        valid_z = pts_cam[:, 2] > 0.1
        
        # Project to pixels: x_pix = K @ x_cam / z
        pts_pix = (K @ pts_cam.T).T
        pts_pix[:, 0] /= pts_pix[:, 2]
        pts_pix[:, 1] /= pts_pix[:, 2]

        # Check image boundaries
        u, v = pts_pix[:, 0], pts_pix[:, 1]
        mask = valid_z & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)

        # Basic visibility check: only color if vertex normal faces the camera
        # (dot product of world normal and view direction)
        view_dir = (kf.pose.camera_center() - mesh.vertices)
        view_dir /= np.linalg.norm(view_dir, axis=1, keepdims=True)
        cos_theta = np.sum(mesh.normals * view_dir, axis=1)
        mask &= (cos_theta > 0.3) # 0.3 approx 70 degrees

        if np.any(mask):
            # Sample colors using bilinear interpolation or simple rounding
            idx_u = u[mask].astype(int)
            idx_v = v[mask].astype(int)
            vertex_colors[mask] += img_rgb[idx_v, idx_u]
            counts[mask] += 1

    # Average the colors
    safe = counts > 0
    vertex_colors[safe] /= counts[safe, None]
    return vertex_colors

def save_textured_mesh(mesh, colors, path):
    """Saves the mesh with vertex colors using Open3D."""
    o3_mesh = o3d.geometry.TriangleMesh()
    o3_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3_mesh.vertex_normals = o3d.utility.Vector3dVector(mesh.normals)
    o3_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_triangle_mesh(str(path), o3_mesh)
    return 

if __name__ == "__main__":
    logging.basicConfig(
        format="[%(filename)s:%(lineno)d:%(funcName)s] %(message)s",
        level=logging.DEBUG,
    )

    workspace = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("workspace")
    recon_path = workspace / "dense" / "sparse"
    image_dir  = workspace / "dense" / "images"

    if not recon_path.exists():
        logger.error(f"Reconstruction not found at {recon_path}. Run s1_sfm.py first.")
        sys.exit(1)
    
    recon = pycolmap.Reconstruction(str(recon_path))
    sfm_result = parse_reconstruction(recon, image_dir, max_reproj_error=4.0, min_track_length=3)

    base_mesh = build_base_mesh(sfm_result, octree_depth=10, density_quantile=0.1)

    # Render first frame
    kf = sfm_result.keyframes[0]
    depth = render_depth(base_mesh, kf.pose, sfm_result.camera_model)
    depth = smooth_depth(depth, sigma=1.0)

    valid = depth > 0
    vis = np.zeros_like(depth, dtype=np.uint8)
    if valid.any():
        d_min, d_max = depth[valid].min(), depth[valid].max()
        vis[valid] = (255 * (depth[valid] - d_min) / (d_max - d_min + 1e-8)).astype(np.uint8)
    cv2.imwrite("depth_check.png", vis)

    logger.info(f"Saved depth_check.png: {valid.sum()} valid pixels, depth range [{depth[valid].min():.2f}, {depth[valid].max():.2f}]")
    logger.info("Done")
