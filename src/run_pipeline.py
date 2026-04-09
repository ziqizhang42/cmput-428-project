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
from opticFlow import structure_texture_decomposition_skimage, get_forward_backward_mask
from s0_bundle_selection import Bundle, select_comparisons_from_buffer, save_bundles, _compute_overlap
from s1_sfm import parse_reconstruction
from s2_base_mesh import BaseMesh, build_base_mesh, render_depth, smooth_depth, save_textured_mesh, texture_mesh
from s3_view_prediction import get_view_prediction
from s4_sceneFlow import constrained_scene_flow
from s5_integration import GlobalModel, integrate_bundle, export_ply, triangulate_depth_map, render_global_depth

logger = logging.getLogger(__name__)

N_COMPARISONS = 4 # comparison frames per bundle (Section 3)
N_ITERATIONS = 3 # Total iterations for vertex updates (Section 2.5.3)
EPSILON = 1e-4

POISSON_DEPTH = 8 # base surface resolution
SIGMA = 5.0 # depth smoothing sigma
OVERLAP_THRESH = 0.3 # V_r for reference selection (Section 2.7)
BUNDLE_WINDOW = 30 # temporal window for comparison selection
OUTPUT_PATH = ""  # set from workspace arg
DEBUG_DIR = Path(".")  # set from workspace arg

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

    # structure-texture decomposition
    _, img_ref_tex = structure_texture_decomposition_skimage(img_ref)

    # Convert back to float 64
    img_ref_tex = img_ref_tex.astype(np.float64)

    imgs_comp_tex = []
    for img in imgs_comp:
        _, t = structure_texture_decomposition_skimage(img)
        t = t.astype(np.float64)
        imgs_comp_tex.append(t)

    # Save reference and comparison images
    cv2.imwrite(str(bdir / "ref.png"), (img_ref * 255).astype(np.uint8))
    for j, img_c in enumerate(imgs_comp):
        cv2.imwrite(str(bdir / f"comp_{j}.png"), (img_c * 255).astype(np.uint8))

    # Initial depth from base mesh
    base_depth = render_depth(mesh, ref.pose, camera)
    save_depth_vis(base_depth, bdir / "depth_0_base_raw.png", "base raw")

    depth = smooth_depth(base_depth, sigma=SIGMA)
    save_depth_vis(depth, bdir / "depth_0_base_smooth.png", "base smooth")

    before_mesh = triangulate_depth_map(base_depth, ref.pose, camera)  # ref, not ref_kf
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(before_mesh.vertices)
    m.triangles = o3d.utility.Vector3iVector(before_mesh.faces)
    m.compute_vertex_normals()

    output_path = DEBUG_DIR / f"before_sceneflow_bundle_{bundle_index}.ply"
    o3d.io.write_triangle_mesh(str(output_path), m)

    if np.sum(depth > 0) == 0:
        logger.warning(f"Warning: no depth for {ref.image_name}, skipping")
        return depth, None, None

    # Current mesh for view prediction (starts as base mesh)
    current_mesh = mesh
    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
    tvl1.setScalesNumber(5)     # Default is 5, but can be increased if needed
    tvl1.setWarpingsNumber(5)   # Increases inner warping iterations to resolve complex flows
    tvl1.setEpsilon(0.01)

    for it in range(N_ITERATIONS):
        it_dir = bdir / f"iter_{it}"
        it_dir.mkdir(exist_ok=True)

        flows = []
        valid_masks = []
        depth_z = []
        for j, (comp_kf, img_comp, img_comp_tex) in enumerate(zip(comps, imgs_comp, imgs_comp_tex)):
            # Render current mesh from comparison viewpoint (Section 2.5.2)
            depth_comp, depth_z_comp = render_depth(current_mesh, comp_kf.pose, camera, True)
            save_depth_vis(depth_comp, it_dir / f"depth_comp_{j}.png", f"comp {j} depth")

            depth_z.append(depth_z_comp)

            # valid mask tells us which pixels of the image prediction are valid
            predicted, valid_mask = get_view_prediction(img_ref_tex, depth_comp, ref.pose, comp_kf.pose, camera)
            predicted_example, _ = get_view_prediction(img_ref, depth_comp, ref.pose, comp_kf.pose, camera)

            cv2.imwrite(str(it_dir / f"predicted_{j}.png"), (np.clip(predicted_example, 0, 1) * 255).astype(np.uint8))
            
            # Convert to correct format
            predicted = predicted.astype(np.float32)

            # Calculate optical flow (expects 8 byte unsigned integer)
            img_comp_u8 = (np.clip(img_comp_tex, 0, 1) * 255).astype(np.uint8)
            predicted_u8 = (np.clip(predicted, 0, 1) * 255).astype(np.uint8)

            # Forward flow: predicted -> img_comp
            flow_fw = tvl1.calc(predicted_u8, img_comp_u8, None).astype(np.float32)

            # Backward flow: img_comp -> predicted  
            flow_bw = tvl1.calc(img_comp_u8, predicted_u8, None).astype(np.float32)

            # Check if forward and backward flow ~the same
            fb_mask = get_forward_backward_mask(flow_fw, flow_bw)

            # Zero out inconsistent flow vectors
            flow_fw[~fb_mask] = 0.0

            # Combine with existing valid mask
            valid_mask = valid_mask & fb_mask
            valid_masks.append(valid_mask)

            save_flow_vis(flow_fw, it_dir / f"flow_{j}.png", f"flow {j}")
            flows.append(flow_fw)

        comp_poses = [c.pose for c in comps]

        depth, E_s, E_v = constrained_scene_flow(depth, ref.pose, camera, comp_poses, flows, valid_masks, depth_z)
        depth = depth.astype(np.float64)

        # Outlier rejection every inner iteration, not just after
        valid_base = base_depth > 0
        outlier = valid_base & (np.abs(depth - base_depth) > 0.3 * base_depth)
        depth[outlier] = base_depth[outlier]
        depth[depth <= 0] = base_depth[depth <= 0]
        
        save_depth_vis(depth, it_dir / "depth_sceneflow.png", "sceneflow")

        # Denoise before next iteration's view prediction
        logger.info(f"Applying TV-L1 depth map denoising...")
        depth = denoise_depth_map_tvl1(D=depth, I_ref=img_ref, alpha=10.0, beta=1.0, lambda_data=2.0, num_iters=100)
        save_depth_vis(depth, it_dir / "depth_denoised.png", "denoised")
        
        # Also catch any other pixels that somehow became 0 or negative
        invalid = depth <= 0
        depth[invalid] = base_depth[invalid]

        logger.info(f"Iteration {it + 1}/{N_ITERATIONS}: {np.sum(depth > 0)} valid pixels ({outlier.sum()} outliers removed)")

        if E_s is not None:
            valid = E_s > 0
            if np.any(valid):
                avg_error = np.mean(E_s[valid])
                logger.info(f"Iteration {it}: avg reprojection error = {avg_error:.6f}")

                if avg_error < EPSILON:
                    logger.info(f"Converged at iteration {it}")
                    break

        # Retriangulate depth into mesh for next iteration's view prediction (Section 2.5.2)
        if it < N_ITERATIONS - 1:
            depth_viewPred = smooth_depth(depth, sigma=SIGMA)

            local_mesh = triangulate_depth_map(depth_viewPred, ref.pose, camera)
            scene = o3d.t.geometry.RaycastingScene()
            mesh_t = o3d.t.geometry.TriangleMesh()
            mesh_t.vertex.positions = o3d.core.Tensor(local_mesh.vertices.astype(np.float32))
            mesh_t.triangle.indices = o3d.core.Tensor(local_mesh.faces.astype(np.int32))
            scene.add_triangles(mesh_t)
            current_mesh = BaseMesh(vertices=local_mesh.vertices, faces=local_mesh.faces, normals=local_mesh.normals, _scene=scene)

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
    DEBUG_DIR = workspace / "debug"
    OUTPUT_PATH = str(workspace / "reconstruction.ply")
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
    base_mesh = build_base_mesh(sfm_result, octree_depth=POISSON_DEPTH, density_quantile=0.15)

    logger.info("Texturing and saving initial Poisson base mesh...")
    base_colors = texture_mesh(base_mesh, sfm_result)
    save_textured_mesh(base_mesh, base_colors, workspace / "initial_textured_poisson.ply")

    logger.info("[3/4] Interleaved bundle selection + processing...")
    keyframes = sfm_result.keyframes
    camera = sfm_result.camera_model
    global_model = GlobalModel()
    bundles = []
    bundle_index = 0

    # Compute adaptive max baseline: median consecutive keyframe distance * 3
    # This approximates paper's constraint in scene-scale units
    depth_sample = render_depth(base_mesh, keyframes[0].pose, camera)
    valid_depths = depth_sample[depth_sample > 0]
    scene_depth = np.median(valid_depths) if len(valid_depths) > 0 else 1.0
    max_baseline = scene_depth * 0.06  # 6% of scene depth
    logger.info(f"Scene-depth-relative max baseline: {max_baseline:.4f}m (scene depth {scene_depth:.4f}m)")

    # Use scene_depth to compute the DIST_THRESHOLD for fusion
    DIST_THRESHOLD = scene_depth * 0.005
    logger.info(f"Fusion dist threshold: {DIST_THRESHOLD:.4f}m")

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

            if len(comparisons) < 2:  # need at least 2 for reliable scene flow
                logger.warning(f"Bundle {bundle_index}: only {len(comparisons)} comparison(s), skipping")
                ref_kf = kf
                buffer = []
                continue

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

    # Post-process global model
    logger.info("Post-processing global model...")

    final_o3d = o3d.geometry.TriangleMesh()
    final_o3d.vertices = o3d.utility.Vector3dVector(global_model.vertices)
    final_o3d.triangles = o3d.utility.Vector3iVector(global_model.faces.astype(np.int32))

    # Clean mesh
    final_o3d.remove_degenerate_triangles()
    final_o3d.remove_unreferenced_vertices()

    # Taubin smoothing
    final_o3d = final_o3d.filter_smooth_taubin(number_of_iterations=10)
    final_o3d.compute_vertex_normals()

    # Write back
    global_model = GlobalModel(vertices=np.asarray(final_o3d.vertices), faces=np.asarray(final_o3d.triangles), normals=np.asarray(final_o3d.vertex_normals))

    export_ply(global_model, OUTPUT_PATH)

    logger.info("Texturing and saving final reconstruction mesh...")
    final_mesh = BaseMesh(vertices=global_model.vertices, faces=global_model.faces, normals=global_model.normals)
    final_colors = texture_mesh(final_mesh, sfm_result)
    save_textured_mesh(final_mesh, final_colors, workspace / "reconstruction_textured.ply")

    logger.info(f"Done: {bundle_index} bundles, {len(global_model.vertices)} vertices")
