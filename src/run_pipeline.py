"""
Runs Dense Reconstruction Pipeline
Assumes s1_sfm.py has already been run (workspace/dense/ exists).
"""

import cv2
import numpy as np
import pycolmap
import sys
import logging
from pathlib import Path

from opticFlow import tv_l1
from s0_bundle_selection import select_bundles, save_bundles
from s1_sfm import parse_reconstruction
from s2_base_mesh import build_base_mesh, render_depth, smooth_depth
from s3_view_prediction import get_view_prediction
from s4_sceneFlow import constrained_scene_flow
from s5_integration import GlobalModel, integrate_bundle, export_ply

logger = logging.getLogger(__name__)

N_COMPARISONS = 3 # comparison frames per bundle
N_ITERATIONS = 2 # scene flow iterations per bundle (Section 2.5.3)
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

def process_bundle(bundle, camera, surface):
    """Stages 3-4 for one bundle: view prediction -> flow -> scene flow."""
    ref = bundle.reference
    comps = bundle.comparisons

    img_ref = load_grayscale(ref.image_path)
    imgs_comp = [load_grayscale(c.image_path) for c in comps]

    # Initial depth from base surface
    depth = render_depth(surface, ref.pose, camera)
    depth = smooth_depth(depth, sigma=SIGMA)

    if np.sum(depth > 0) == 0:
        logger.warning(f"Warning: no depth for {ref.image_name}, skipping")
        return depth
    
    for it in range(N_ITERATIONS):
        # For each comparison: predict view, compute optical flow
        flows = []
        for comp_kf, img_comp in zip(comps, imgs_comp):
            predicted = get_view_prediction(img_ref, depth, ref.pose, comp_kf.pose, camera)
            u1, u2 = tv_l1(predicted, img_comp)
            flows.append(np.stack([u1, u2], axis=-1).astype(np.float32))

        # Scene flow update
        comp_poses = [c.pose for c in comps]
        depth = constrained_scene_flow(depth, ref.pose, camera, comp_poses, flows).astype(np.float64)
        logger.info(f"Iteration {it + 1}/{N_ITERATIONS}: {np.sum(depth > 0)} valid pixels")

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
