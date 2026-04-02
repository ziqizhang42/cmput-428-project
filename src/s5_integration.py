"""
Integration (Section 2.6)

Triangulates refined depth maps into meshes and fuses them into a single global reconstruction.
"""

import numpy as np
import open3d as o3d
import logging
from dataclasses import dataclass, field
from s1_sfm import CameraModel, Pose

logger = logging.getLogger(__name__)

@dataclass
class TriangulatedMesh:
    """A triangle mesh with per-vertex metadata."""
    vertices: np.ndarray # (V, 3) world-frame positions
    faces: np.ndarray # (F, 3) triangle vertex indices
    normals: np.ndarray # (V, 3) per-vertex normals

@dataclass
class GlobalModel:
    """Accumulated global reconstruction from all bundles."""
    vertices: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    faces: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.int64))
    normals: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))

def triangulate_depth_map(depth: np.ndarray, pose: Pose, camera: CameraModel) -> TriangulatedMesh:
    """Triangulate a refined depth map into a 3D mesh. (Section 2.4.3, Equation 9)"""
    h, w = depth.shape
    K_inv = np.linalg.inv(camera.K)
    center = pose.camera_center()

    # Build per-pixel unit ray directions in world frame (Equation 4)
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    pixels = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3).T

    rays_cam = (K_inv @ pixels).T
    rays_world = (pose.R.T @ rays_cam.T).T
    norms = np.linalg.norm(rays_world, axis=-1, keepdims=True)
    unit_rays = rays_world / norms

    # Back-project
    D = depth.flatten()
    valid = D > 0
    vertices = unit_rays * D[:, None] + center

    # Grid triangulation
    valid_2d = valid.reshape(h, w)
    pixel_index = np.arange(h * w).reshape(h, w)

    # Quad corners
    tl = pixel_index[:-1, :-1] # top-left
    tr = pixel_index[:-1, 1:] # top-right
    bl = pixel_index[1:, :-1] # bottom-left
    br = pixel_index[1:, 1:] # bottom-right

    quad_valid = valid_2d[:-1, :-1] & valid_2d[:-1, 1:] & valid_2d[1:, :-1] & valid_2d[1:, 1:]

    # Two triangles per valid quad
    tl_f = tl[quad_valid].flatten()
    tr_f = tr[quad_valid].flatten()
    bl_f = bl[quad_valid].flatten()
    br_f = br[quad_valid].flatten()

    tri1 = np.stack([tl_f, bl_f, tr_f], axis=1) # upper-left triangle
    tri2 = np.stack([tr_f, bl_f, br_f], axis=1) # lower-right triangle
    faces = np.vstack([tri1, tri2])

    # Per-vertex normals
    verts_2d = vertices.reshape(h, w, 3)
    dvx = np.zeros_like(verts_2d)
    dvy = np.zeros_like(verts_2d)
    dvx[:, 1:-1] = verts_2d[:, 2:] - verts_2d[:, :-2]
    dvx[:, 0] = verts_2d[:, 1] - verts_2d[:, 0]
    dvx[:, -1] = verts_2d[:, -1] - verts_2d[:, -2]
    dvy[1:-1, :] = verts_2d[2:, :] - verts_2d[:-2, :]
    dvy[0, :] = verts_2d[1, :] - verts_2d[0, :]
    dvy[-1, :] = verts_2d[-1, :] - verts_2d[-2, :]

    normals = np.cross(dvx.reshape(-1, 3), dvy.reshape(-1, 3))
    n_norms = np.linalg.norm(normals, axis=1, keepdims=True)
    n_norms = np.maximum(n_norms, 1e-10)
    normals = normals / n_norms

    normals[~valid] = 0.0

    logging.info(f"Triangulated: {valid.sum()} vertices, {len(faces)} faces")
    return TriangulatedMesh(vertices=vertices, faces=faces, normals=normals)

def filter_by_error(mesh: TriangulatedMesh, E_s: np.ndarray, E_v: np.ndarray, ev_threshold: float = 0.9, es_threshold: float = 1e-3) -> TriangulatedMesh:
    """ Remove low-quality vertices based on reconstruction error measures. (Section 2.4.4)"""
    # Vertices to keep
    keep = ~((E_v < ev_threshold) & (E_s > es_threshold))

    # Build old -> new vertex index mapping
    new_index = np.full(len(mesh.vertices), -1, dtype=np.int32)
    new_index[keep] = np.arange(keep.sum())

    # Keep only faces where all three vertices survive
    face_valid = np.all(keep[mesh.faces], axis=1)
    new_faces = new_index[mesh.faces[face_valid]]
 
    new_verts = mesh.vertices[keep]
    new_normals = mesh.normals[keep]

    logging.info(f"After error filter: {len(new_verts)} vertices, {len(new_faces)} faces (removed {(~keep).sum()})")

    return TriangulatedMesh(vertices=new_verts, faces=new_faces, normals=new_normals)

def _render_global_depth(global_model: GlobalModel, pose: Pose, camera: CameraModel) -> np.ndarray:
    """Render the current global model into a reference viewpoint. Returns (H, W) depth map (ray-hit distance, 0 = no hit)."""
    if len(global_model.vertices) == 0 or len(global_model.faces) == 0:
        return np.zeros((camera.height, camera.width), dtype=np.float64)

    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh()
    mesh_t.vertex.positions = o3d.core.Tensor(global_model.vertices.astype(np.float32))
    mesh_t.triangle.indices = o3d.core.Tensor(global_model.faces.astype(np.int32))
    scene.add_triangles(mesh_t)

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
    depth = result['t_hit'].numpy().astype(np.float64)
    depth[depth == np.inf] = 0.0
    return depth

def fuse_into_global(global_model: GlobalModel, new_mesh: TriangulatedMesh, pose_ref: Pose, camera: CameraModel, dist_threshold: float = 0.01) -> GlobalModel:
    """Fuse a new per-bundle mesh into the global model. (Section 2.6)"""
    # Render existing global model into reference view
    global_depth = _render_global_depth(global_model, pose_ref, camera)

    h, w = camera.height, camera.width
    K = camera.K

    # Project each new vertex into the reference camera
    cam_coords = pose_ref.world_to_camera(new_mesh.vertices)
    z_vals = cam_coords[:, 2]

    proj = (K @ cam_coords.T).T
    px = proj[:, 0] / (z_vals + 1e-8)
    py = proj[:, 1] / (z_vals + 1e-8)

    # For each new vertex, check overlap with existing geometry.
    # create_rays_pinhole uses unnormalized ray directions (z-component about 1), so t_hit is about z-depth in camera frame.
    keep = np.ones(len(new_mesh.vertices), dtype=bool)

    # Remove vertices behind camera
    keep[z_vals <= 0] = False

    # Bounds check and overlap detection
    px_i = np.round(px).astype(np.int32)
    py_i = np.round(py).astype(np.int32)
    in_bounds = (px_i >= 0) & (px_i < w) & (py_i >= 0) & (py_i < h) & (z_vals > 0)
    ib_index = np.where(in_bounds)[0]

    if len(ib_index) > 0:
        existing_z = global_depth[py_i[ib_index], px_i[ib_index]]
        overlaps = (existing_z > 0) & (np.abs(z_vals[ib_index] - existing_z) < dist_threshold)
        keep[ib_index[overlaps]] = False
    
    # Remap faces
    new_index = np.full(len(new_mesh.vertices), -1, dtype=np.int32)
    new_index[keep] = np.arange(keep.sum())

    face_valid = np.all(keep[new_mesh.faces], axis=1)
    surviving_faces = new_index[new_mesh.faces[face_valid]]

    surviving_verts = new_mesh.vertices[keep]
    surviving_normals = new_mesh.normals[keep]

    n_removed = (~keep).sum()
    logging.info(f"Fusion: kept {keep.sum()} / {len(keep)} vertices ({n_removed} overlapping or invalid)")

    # Offset face indices and append to global
    offset = len(global_model.vertices)
    if len(surviving_faces) > 0:
        surviving_faces = surviving_faces + offset

    new_global_verts = np.vstack([global_model.vertices, surviving_verts]) if len(global_model.vertices) > 0 else surviving_verts
    new_global_faces = np.vstack([global_model.faces, surviving_faces]) if len(global_model.faces) > 0 else surviving_faces
    new_global_normals = np.vstack([global_model.normals, surviving_normals]) if len(global_model.normals) > 0 else surviving_normals

    return GlobalModel(vertices=new_global_verts, faces=new_global_faces, normals=new_global_normals)

def integrate_bundle(depth: np.ndarray, pose_ref: Pose, camera: CameraModel, global_model: GlobalModel,
                     E_s: np.ndarray | None = None, E_v: np.ndarray | None = None, dist_threshold: float = 0.01) -> GlobalModel:
    """Full stage 5 for one bundle: triangulate, filter, fuse"""
    mesh = triangulate_depth_map(depth, pose_ref, camera)
    if E_s is not None and E_v is not None:
        mesh = filter_by_error(mesh, E_s, E_v)
    return fuse_into_global(global_model, mesh, pose_ref, camera, dist_threshold)

def export_ply(global_model: GlobalModel, path: str) -> None:
    """Save the global model as a PLY file."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(global_model.vertices)
    mesh.triangles = o3d.utility.Vector3iVector(global_model.faces)
    mesh.vertex_normals = o3d.utility.Vector3dVector(global_model.normals)
    o3d.io.write_triangle_mesh(path, mesh)
    logger.info(f"Exported {path}: {len(global_model.vertices)} vertices, {len(global_model.faces)} faces")

if __name__ == "__main__":
    logging.basicConfig(
        format="[%(filename)s:%(lineno)d:%(funcName)s] %(message)s",
        level=logging.DEBUG,
    )
    # TODO
