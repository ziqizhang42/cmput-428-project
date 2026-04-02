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
from s0_bundle_selection import select_bundles, save_bundles
from s1_sfm import parse_reconstruction
from s2_base_mesh import BaseMesh, build_base_mesh, render_depth, smooth_depth
from s3_view_prediction import get_view_prediction
from s4_sceneFlow import constrained_scene_flow
from s5_integration import GlobalModel, integrate_bundle, export_ply, triangulate_depth_map

logger = logging.getLogger(__name__)

N_COMPARISONS = 4 # comparison frames per bundle (Section 3)
SCENE_ITERATIONS = 3 # Number of scene flow iterations (2.4.2)
N_ITERATIONS = 2 # Total iterations for vertex updates (Section 2.5.3)

POISSON_DEPTH = 8 # base surface resolution
SIGMA = 1.0 # depth smoothing sigma
OVERLAP_THRESH = 0.3 # V_r for reference selection (Section 2.7)
BUNDLE_WINDOW = 30 # temporal window for comparison selection
DIST_THRESHOLD = 0.01 # overlap threshold for mesh fusion
OUTPUT_PATH = "reconstruction.ply"

def load_grayscale(path: Path) -> np.ndarray:
    """Load image as float64 grayscale in [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img.astype(np.float64) / 255.0

def process_bundle(bundle, camera, mesh):
    """Stages 3-4 for one bundle: view prediction -> flow -> scene flow."""
    ref = bundle.reference
    comps = bundle.comparisons

    img_ref = load_grayscale(ref.image_path)
    imgs_comp = [load_grayscale(c.image_path) for c in comps]

    # Initial depth from base mesh
    depth = render_depth(mesh, ref.pose, camera)
    depth = smooth_depth(depth, sigma=SIGMA)

    if np.sum(depth > 0) == 0:
        logger.warning(f"Warning: no depth for {ref.image_name}, skipping")
        return depth

    # Current mesh for view prediction (starts as base mesh)
    current_mesh = mesh

    for it in range(N_ITERATIONS):
        flows = []
        for comp_kf, img_comp in zip(comps, imgs_comp):
            # Render current mesh from comparision viewpoint (Section 2.5.2)
            depth_comp = render_depth(current_mesh, comp_kf.pose, camera)
            predicted = get_view_prediction(img_ref, depth_comp, ref.pose, comp_kf.pose, camera)

            # Calculate optical flow
            u1, u2 = tv_l1(predicted, img_comp)
            flows.append(np.stack([u1, u2], axis=-1).astype(np.float32))

        comp_poses = [c.pose for c in comps]

        for s_it in range(SCENE_ITERATIONS):
            # Update 3D points based on the optical flow data
            depth = constrained_scene_flow(depth, ref.pose, camera, comp_poses, flows).astype(np.float64)
            logger.debug(f"Scene Flow Sub-Iteration {s_it + 1}/{SCENE_ITERATIONS}")

        # Denoise before next iteration's view prediction
        logger.info(f"Applying TV-L1 depth map denoising...")
        depth = denoise_depth_map_tvl1(D=depth, I_ref=img_ref, alpha=10.0, beta=1.0, lambda_data=1.0, num_iters=100)

        logger.info(f"Iteration {it + 1}/{N_ITERATIONS}: {np.sum(depth > 0)} valid pixels after denoising")

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

    return depth

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

    logger.info("[2/5] Building base mesh...")
    mesh = build_base_mesh(sfm_result, octree_depth=POISSON_DEPTH, density_quantile=0.1)

    logger.info("[3/5] Selecting bundles...")
    bundles = select_bundles(sfm_result, mesh, n_comparisons=N_COMPARISONS, overlap_threshold=OVERLAP_THRESH, window=BUNDLE_WINDOW)

    if not bundles:
        logger.warning("No bundles selected - check settings")
        sys.exit(1)

    bundle_path = workspace / "bundles.json"
    save_bundles(bundles, bundle_path)

    logger.info(f"[4/5] Processing {len(bundles)} bundles...")
    global_model = GlobalModel()

    for i, bundle in enumerate(bundles):
        logger.info(f"Bundle {i + 1}/{len(bundles)}: ref={bundle.reference.image_name}")

        depth = process_bundle(bundle, sfm_result.camera_model, mesh)

        # TODO: need E_s and E_v
        global_model = integrate_bundle(depth, bundle.reference.pose, sfm_result.camera_model, global_model, E_s, E_v, dist_threshold=DIST_THRESHOLD)

    logger.info(f"[5/5] Saving final model to {OUTPUT_PATH}...")
    export_ply(global_model, OUTPUT_PATH)
    logger.info(f"Done, {len(global_model.vertices)} vertices")
