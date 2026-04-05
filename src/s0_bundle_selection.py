"""
Camera Bundle Selection (Section 2.7)
"""

import numpy as np
import logging
import json
from pathlib import Path
from dataclasses import dataclass

from s1_sfm import SfMResult, CameraModel, Pose, Keyframe
from s2_base_mesh import BaseMesh, render_depth

logger = logging.getLogger(__name__)

@dataclass
class Bundle:
    """A camera bundle (one reference frame and n comparison frames)."""
    reference: Keyframe
    comparisons: list[Keyframe]

def _compute_coverage(mesh: BaseMesh, pose: Pose, camera: CameraModel) -> float:
    """Fraction of pixels in the frame that see the base mesh."""
    depth = render_depth(mesh, pose, camera)
    total_pixels = camera.width * camera.height
    valid_pixels = np.sum(depth > 0)
    return valid_pixels / total_pixels

def _compute_overlap(mesh: BaseMesh, pose_ref: Pose, pose_candidate: Pose, camera: CameraModel) -> float:
    """Fraction of the reference frame's visible surface that is also visible from the candidate frame."""
    depth_ref = render_depth(mesh, pose_ref, camera)
    depth_cand = render_depth(mesh, pose_candidate, camera)
    ref_valid = depth_ref > 0
    cand_valid = depth_cand > 0

    # overlap = pixels valid in both views / pixels valid in reference
    both_valid = ref_valid & cand_valid
    ref_count = np.sum(ref_valid)
    if ref_count == 0:
        return 0.0
    return np.sum(both_valid) / ref_count

def select_reference_frames(keyframes: list[Keyframe], mesh: BaseMesh, camera: CameraModel, overlap_threshold: float = 0.3, min_coverage: float = 0.1) -> list[int]:
    """
    Walk through keyframes and select reference frames where the view has changed enough from the last reference.
    Returns list of indices into the keyframes list.
    """
    if len(keyframes) == 0:
        return []

    # First reference is always the first keyframe with enough coverage
    ref_indices = []
    for i, kf in enumerate(keyframes):
        cov = _compute_coverage(mesh, kf.pose, camera)
        if cov >= min_coverage:
            ref_indices.append(i)
            break

    if not ref_indices:
        logger.warning("Warning: no keyframe sees the surface")
        return []

    last_ref_index = ref_indices[0]
    last_ref_pose = keyframes[last_ref_index].pose

    for i in range(last_ref_index + 1, len(keyframes)):
        kf = keyframes[i]

        # check coverage
        cov = _compute_coverage(mesh, kf.pose, camera)
        if cov < min_coverage:
            continue

        # check overlap with last reference
        overlap = _compute_overlap(
            mesh, keyframes[last_ref_index].pose, kf.pose, camera
        )

        # low overlap, so new reference
        if overlap < overlap_threshold:
            ref_indices.append(i)
            last_ref_index = i

    logging.info(f"Selected {len(ref_indices)} reference frames from {len(keyframes)} keyframes (V_r < {overlap_threshold})")
    return ref_indices

def _translation_from_ref(kf: Keyframe, ref: Keyframe) -> np.ndarray:
    """Translation vector from reference camera center to kf center."""
    return kf.pose.camera_center() - ref.pose.camera_center()

def _perpendicular_angle(translation: np.ndarray, view_dir: np.ndarray) -> float:
    """
    Angle of the translation vector projected perpendicular to the viewing direction.
    Returns angle in [0, 2pi).
    """
    # Build a local frame: z = view_dir, x and y perpendicular
    z = view_dir / (np.linalg.norm(view_dir) + 1e-8)

    # Pick an arbitrary vector not parallel to z to build x-axis
    up = np.array([0, 1, 0]) if abs(z[1]) < 0.9 else np.array([1, 0, 0])
    x = np.cross(z, up)
    x = x / (np.linalg.norm(x) + 1e-8)
    y = np.cross(z, x)

    # Project translation onto the perpendicular plane
    tx = np.dot(translation, x)
    ty = np.dot(translation, y)

    return np.arctan2(ty, tx) % (2 * np.pi)

def select_comparisons_from_buffer(ref: Keyframe, buffer: list[Keyframe], n: int = 4,
                                   min_baseline: float = 0.005, max_baseline: float = float('inf')) -> list[Keyframe]:
    """
    Select n comparison frames from a buffer of co-visible frames (Section 2.7).
    All frames in the buffer already have V_c >= 0.7 with the reference by construction.
    Selection maximizes angular diversity of translation directions.
    """
    ref_view_dir = ref.pose.R.T @ np.array([0, 0, 1])

    # Compute baseline and angle for each buffer frame
    candidates = []
    for kf in buffer:
        trans = _translation_from_ref(kf, ref)
        baseline = np.linalg.norm(trans)
        if baseline < min_baseline or baseline > max_baseline:
            continue

        # Check view direction alignment
        kf_view_dir = kf.pose.R.T @ np.array([0, 0, 1])
        cos_sim = np.dot(ref_view_dir, kf_view_dir)
        view_angle = np.arccos(np.clip(cos_sim, -1.0, 1.0)) * 180 / np.pi
        
        if view_angle > 20.0:  # Ignore frames rotated by more than 20 degrees
            continue

        angle = _perpendicular_angle(trans, ref_view_dir)
        candidates.append((kf, baseline, angle))

    if len(candidates) == 0:
        return []

    if len(candidates) <= n:
        result = [c[0] for c in candidates]
        angles_deg = [c[2] * 180 / np.pi for c in candidates]
        logger.info(f"Ref {ref.image_name}: {len(result)} comparisons (all buffer), angles={[f'{a:.0f}°' for a in angles_deg]}")
        return result

    # Greedy angular diversity: start with largest baseline
    candidates.sort(key=lambda c: c[1], reverse=True)
    selected = [candidates[0]]
    remaining = candidates[1:]

    for _ in range(n - 1):
        if not remaining:
            break

        selected_angles = [s[2] for s in selected]
        best_score = -1
        best_index = 0

        for j, cand in enumerate(remaining):
            min_dist = min(min(abs(cand[2] - sa), 2 * np.pi - abs(cand[2] - sa)) for sa in selected_angles)
            if min_dist > best_score:
                best_score = min_dist
                best_index = j

        selected.append(remaining[best_index])
        remaining.pop(best_index)

    result = [s[0] for s in selected]
    angles_deg = [s[2] * 180 / np.pi for s in selected]
    logger.info(f"Ref {ref.image_name}: {len(result)} comparisons, angles={[f'{a:.0f}°' for a in angles_deg]}")
    return result

def select_bundles(sfm_result: SfMResult, mesh: BaseMesh, n_comparisons: int = 3, overlap_threshold: float = 0.3, window: int = 30) -> list[Bundle]:
    """Select all bundles for dense reconstruction."""
    keyframes = sfm_result.keyframes
    camera = sfm_result.camera_model

    # Select reference frames
    ref_indices = select_reference_frames(keyframes, mesh, camera, overlap_threshold)

    # Select comparison frames for each reference
    bundles = []
    for ref_index in ref_indices:
        comp_indices = select_comparison_frames(ref_index, keyframes, camera, mesh, n=n_comparisons, window=window)

        if len(comp_indices) == 0:
            continue

        bundle = Bundle(reference=keyframes[ref_index], comparisons=[keyframes[i] for i in comp_indices])
        bundles.append(bundle)

    logger.info(f"Created {len(bundles)} bundles ({n_comparisons} comparisons each)")
    return bundles

def _serialize_keyframe(kf: Keyframe) -> dict:
    return {
        "image_id": kf.image_id,
        "frame_index": kf.frame_index,
        "image_name": kf.image_name,
        "image_path": str(kf.image_path),
        "R": kf.pose.R.tolist(),
        "t": kf.pose.t.tolist(),
    }

def save_bundles(bundles: list[Bundle], path: str | Path) -> None:
    data = []
    for b in bundles:
        data.append({
            "reference": _serialize_keyframe(b.reference),
            "comparisons": [_serialize_keyframe(c) for c in b.comparisons],
        })
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logging.info(f"Saved {len(bundles)} bundles to {path}")

def load_bundles(path: str | Path, keyframes: list[Keyframe]) -> list[Bundle]:
    with open(path) as f:
        data = json.load(f)
    kf_map = {kf.image_id: kf for kf in keyframes} if keyframes else None

    def _deserialize(entry: dict) -> Keyframe:
        if kf_map and entry["image_id"] in kf_map:
            return kf_map[entry["image_id"]]
        R = np.array(entry["R"])
        t = np.array(entry["t"])
        pose = Pose(R=R, t=t)
        return Keyframe(
            image_id=entry["image_id"],
            frame_index=entry["frame_index"],
            image_name=entry["image_name"],
            image_path=str(entry["image_path"]),
            pose=pose,
            P=pose.projection_matrix(np.eye(3)),
        )
    
    bundles = []
    for entry in data:
        ref = _deserialize(entry["reference"])
        comps = [_deserialize(c) for c in entry["comparisons"]]
        if comps:
            bundles.append(Bundle(reference=ref, comparisons=comps))

    logging.info(f"Loaded {len(bundles)} bundles from {path}")
    return bundles

if __name__ == "__main__":
    logging.basicConfig(
        format="[%(filename)s:%(lineno)d:%(funcName)s] %(message)s",
        level=logging.DEBUG,
    )
    # TODO
