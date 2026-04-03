"""
Runs Dense Reconstruction Pipeline
Assumes s1_sfm.py has already been run (workspace/dense/ exists).
"""

import cv2
import numpy as np
import pycolmap
import sys
import logging
import open3d as o3d
from pathlib import Path

from depth_denoise import denoise_depth_map_tvl1
from opticFlow import tv_l1
from s0_bundle_selection import Bundle, select_comparisons_from_buffer, save_bundles, _compute_overlap
from s1_sfm import parse_reconstruction
from s2_base_mesh import BaseMesh, build_base_mesh, render_depth, smooth_depth
from s3_view_prediction import get_view_prediction
from s4_sceneFlow import constrained_scene_flow
from s5_integration import GlobalModel, integrate_bundle, export_ply, triangulate_depth_map, render_global_depth

logger = logging.getLogger(__name__)

N_COMPARISONS = 4 # comparison frames per bundle (Section 3)
N_ITERATIONS = 2 # Total iterations for vertex updates (Section 2.5.3)

POISSON_DEPTH = 8 # base surface resolution
SIGMA = 1.0 # depth smoothing sigma
OVERLAP_THRESH = 0.3 # V_r for reference selection (Section 2.7)
BUNDLE_WINDOW = 30 # temporal window for comparison selection
DIST_THRESHOLD = 0.01 # overlap threshold for mesh fusion
OUTPUT_PATH = "reconstruction.ply"

DEBUG_DIR: Path = Path("debug")

def load_grayscale(path: Path) -> np.ndarray:
    """Load image as float64 grayscale in [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img.astype(np.float64) / 255.0

def save_depth_vis(depth: np.ndarray, path: Path, label: str = ""):
    """Save depth map as a color-mapped PNG."""
    valid = depth > 0
    if not np.any(valid):
        logger.warning(f"[{label}] all-zero depth, skipping save")
        return
    d_min, d_max = depth[valid].min(), depth[valid].max()
    logger.info(f"[{label}] depth range: {d_min:.4f} - {d_max:.4f}, valid: {valid.sum()}/{depth.size}")
    norm = np.zeros_like(depth, dtype=np.uint8)
    norm[valid] = ((depth[valid] - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    cv2.imwrite(str(path), colored)

def save_flow_vis(flow: np.ndarray, path: Path, label: str = ""):
    """Save optical flow as HSV color-mapped PNG."""
    mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    ang = np.arctan2(flow[..., 1], flow[..., 0])
    logger.info(f"[{label}] flow mag range: {mag.min():.2f} - {mag.max():.2f}, mean: {mag.mean():.2f}")
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / (mag.max() + 1e-8) * 255, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(str(path), bgr)

def save_mesh_ply(depth: np.ndarray, pose, camera, path: Path, label: str = ""):
    """Triangulate depth and save as PLY."""
    mesh = triangulate_depth_map(depth, pose, camera)
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    m.triangles = o3d.utility.Vector3iVector(mesh.faces)
    m.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(path), m)
    logger.info(f"[{label}] saved mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

def process_bundle(bundle, camera, mesh, bundle_index: int):
    """Stages 3-4 for one bundle: view prediction -> flow -> scene flow."""
    ref = bundle.reference
    comps = bundle.comparisons
    bdir = DEBUG_DIR / f"bundle_{bundle_index:03d}"
    bdir.mkdir(parents=True, exist_ok=True)

    img_ref = load_grayscale(ref.image_path)
    imgs_comp = [load_grayscale(c.image_path) for c in comps]

    # Save reference and comparison images
    cv2.imwrite(str(bdir / "ref.png"), (img_ref * 255).astype(np.uint8))
    for j, img_c in enumerate(imgs_comp):
        cv2.imwrite(str(bdir / f"comp_{j}.png"), (img_c * 255).astype(np.uint8))

    # Initial depth from base mesh
    base_depth = render_depth(mesh, ref.pose, camera)
    save_depth_vis(base_depth, bdir / "depth_0_base_raw.png", "base raw")

    depth = smooth_depth(base_depth, sigma=SIGMA)
    save_depth_vis(depth, bdir / "depth_0_base_smooth.png", "base smooth")

    if np.sum(depth > 0) == 0:
        logger.warning(f"Warning: no depth for {ref.image_name}, skipping")
        return depth, None, None

    # Current mesh for view prediction (starts as base mesh)
    current_mesh = mesh

    for it in range(N_ITERATIONS):
        it_dir = bdir / f"iter_{it}"
        it_dir.mkdir(exist_ok=True)

        flows = []
        for j, (comp_kf, img_comp) in enumerate(zip(comps, imgs_comp)):
            # Render current mesh from comparison viewpoint (Section 2.5.2)
            depth_comp = render_depth(current_mesh, comp_kf.pose, camera)
            save_depth_vis(depth_comp, it_dir / f"depth_comp_{j}.png", f"comp {j} depth")

            predicted = get_view_prediction(img_ref, depth_comp, ref.pose, comp_kf.pose, camera)
            cv2.imwrite(str(it_dir / f"predicted_{j}.png"), (np.clip(predicted, 0, 1) * 255).astype(np.uint8))

            # Calculate optical flow
            u1, u2 = tv_l1(predicted, img_comp)
            flow = np.stack([u1, u2], axis=-1).astype(np.float32)
            save_flow_vis(flow, it_dir / f"flow_{j}.png", f"flow {j}")
            flows.append(flow)

        comp_poses = [c.pose for c in comps]

        depth, E_s, E_v = constrained_scene_flow(depth, ref.pose, camera, comp_poses, flows)
        depth = depth.astype(np.float64)
        save_depth_vis(depth, it_dir / "depth_sceneflow.png", "sceneflow")

        # Denoise before next iteration's view prediction
        logger.info(f"Applying TV-L1 depth map denoising...")
        depth = denoise_depth_map_tvl1(D=depth, I_ref=img_ref, alpha=10.0, beta=1.0, lambda_data=1.0, num_iters=100)
        save_depth_vis(depth, it_dir / "depth_denoised.png", "denoised")

        # Reject outlier depth: zero out pixels that deviate too far from base mesh
        valid_base = base_depth > 0
        outlier = valid_base & (np.abs(depth - base_depth) > 0.5 * base_depth)
        depth[outlier] = 0.0
        logger.info(f"Iteration {it + 1}/{N_ITERATIONS}: {np.sum(depth > 0)} valid pixels ({outlier.sum()} outliers removed)")

        # Retriangulate depth into mesh for next iteration's view prediction (Section 2.5.2)
        if it < N_ITERATIONS - 1:
            local_mesh = triangulate_depth_map(depth, ref.pose, camera)
            scene = o3d.t.geometry.RaycastingScene()
            mesh_t = o3d.t.geometry.TriangleMesh()
            mesh_t.vertex.positions = o3d.core.Tensor(local_mesh.vertices.astype(np.float32))
            mesh_t.triangle.indices = o3d.core.Tensor(local_mesh.faces.astype(np.int32))
            scene.add_triangles(mesh_t)
            current_mesh = BaseMesh(
                vertices=local_mesh.vertices,
                faces=local_mesh.faces,
                normals=local_mesh.normals,
                _scene=scene,
            )

    # Save final per-bundle mesh
    save_mesh_ply(depth, ref.pose, camera, bdir / "bundle_mesh.ply", "final bundle mesh")

    return depth, E_s, E_v

if __name__ == "__main__":
    logging.basicConfig(
        format="[%(filename)s:%(lineno)d:%(funcName)s] %(message)s",
        level=logging.DEBUG,
    )
    if len(sys.argv) < 2:
        logger.error("Usage: python run_pipeline.py <workspace>")
        sys.exit(1)

    workspace = Path(sys.argv[1])
    recon_path = workspace / "dense" / "sparse"
    image_dir  = workspace / "dense" / "images"

    if not recon_path.exists():
        logger.error(f"No reconstruction at {recon_path} - run s1_sfm.py first")
        sys.exit(1)
    
    logger.info("[1/5] Loading SfM result...")
    recon = pycolmap.Reconstruction(str(recon_path))
    sfm_result = parse_reconstruction(recon, image_dir)
    logger.info(f"{len(sfm_result.keyframes)} keyframes, {len(sfm_result.sparse_points)} points")

    logger.info("[2/4] Building base mesh...")
    base_mesh = build_base_mesh(sfm_result, octree_depth=POISSON_DEPTH, density_quantile=0.1)

    logger.info("[3/4] Interleaved bundle selection + processing...")
    keyframes = sfm_result.keyframes
    camera = sfm_result.camera_model
    global_model = GlobalModel()
    bundles = []
    bundle_index = 0

    # Compute adaptive max baseline: median consecutive keyframe distance * 3
    # This approximates paper's constraint in scene-scale units
    consecutive_dists = [
        np.linalg.norm(keyframes[i+1].pose.camera_center() - keyframes[i].pose.camera_center())
        for i in range(len(keyframes) - 1)
    ]
    median_step = np.median(consecutive_dists)
    max_baseline = median_step * 3
    logger.info(f"Adaptive max baseline: {max_baseline:.4f} (median step {median_step:.4f} * 3)")

    # Section 2.7: walk through frames, maintain a buffer of co-visible frames.
    # When co-visibility V_c drops or buffer is full, trigger a new bundle.
    COVIS_THRESH = 0.7
    MAX_BUFFER = 60
    MIN_COVERAGE = 0.1

    ref_kf = None
    buffer = []

    for i, kf in enumerate(keyframes):
        # Find first reference
        if ref_kf is None:
            cov = _compute_overlap(base_mesh, kf.pose, kf.pose, camera)
            if cov >= MIN_COVERAGE:
                ref_kf = kf
                logger.info(f"Frame {i}: initial reference ({kf.image_name})")
            continue

        # Compute co-visibility with current reference via base mesh
        covis = _compute_overlap(base_mesh, ref_kf.pose, kf.pose, camera)

        # Trigger bundle when V_c drops or buffer is full
        trigger = covis < COVIS_THRESH or len(buffer) >= MAX_BUFFER

        if not trigger:
            buffer.append(kf)
            continue

        reason = f"V_c={covis:.2f}" if covis < COVIS_THRESH else f"buffer full ({len(buffer)})"
        logger.info(f"Frame {i}: {reason} -> bundle {bundle_index} (ref={ref_kf.image_name}, buffer={len(buffer)} frames)")

        # Process bundle
        if buffer:
            comparisons = select_comparisons_from_buffer(ref_kf, buffer, n=N_COMPARISONS, max_baseline=max_baseline)
            if comparisons:
                bundle = Bundle(reference=ref_kf, comparisons=comparisons)
                bundles.append(bundle)
                depth, E_s, E_v = process_bundle(bundle, camera, base_mesh, bundle_index)
                global_model = integrate_bundle(depth, ref_kf.pose, camera, global_model, E_s=E_s, E_v=E_v, dist_threshold=DIST_THRESHOLD)
                bundle_index += 1
                logger.info(f"Global model: {len(global_model.vertices)} vertices")

        # This frame becomes the new reference, clear buffer
        ref_kf = kf
        buffer = []

    # Process final bundle if buffer is non-empty
    if ref_kf is not None and buffer:
        logger.info(f"Final bundle {bundle_index} (ref={ref_kf.image_name}, buffer={len(buffer)} frames)")
        comparisons = select_comparisons_from_buffer(ref_kf, buffer, n=N_COMPARISONS, max_baseline=max_baseline)
        if comparisons:
            bundle = Bundle(reference=ref_kf, comparisons=comparisons)
            bundles.append(bundle)
            depth, E_s, E_v = process_bundle(bundle, camera, base_mesh, bundle_index)
            global_model = integrate_bundle(depth, ref_kf.pose, camera, global_model, E_s=E_s, E_v=E_v, dist_threshold=DIST_THRESHOLD)
            bundle_index += 1

    if bundles:
        bundle_path = workspace / "bundles.json"
        save_bundles(bundles, bundle_path)

    logger.info(f"[4/4] Saving final model to {OUTPUT_PATH}...")
    export_ply(global_model, OUTPUT_PATH)
    logger.info(f"Done: {bundle_index} bundles, {len(global_model.vertices)} vertices")
