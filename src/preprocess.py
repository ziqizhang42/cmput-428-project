"""
Frame extraction

- One-shot with no feedback (we will do adaptive frame/bundle selection later)
- Now includes automatic downscaling to a target resolution while preserving aspect ratio.
"""

import cv2
from pathlib import Path
import sys
import logging

logger = logging.getLogger(__name__)

def extract_frames(video_path: str | Path, frame_step: int, output_dir: str | Path, target_height: int | None = None) -> list[Path]:
    """Samples every 'frame_step'-th frame from a video, and downsamples to desired resolution"""
    
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Video: {video_path.name} ({total_frames} frames, {fps:.1f} fps)")

    paths = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % frame_step == 0:
            if target_height is not None and frame.shape[0] > target_height:
                h, w = frame.shape[:2]
                scale = target_height / float(h)
                new_width = int(w * scale)
                frame = cv2.resize(frame, (new_width, target_height), interpolation=cv2.INTER_AREA)

            p = output_dir / f"frame_{i:06d}.png"
            cv2.imwrite(str(p), frame)
            paths.append(p)
        i += 1
    
    cap.release()
    logger.info(f"Extracted {len(paths)} frames to {output_dir}")
    return paths

if __name__ == "__main__":
    logging.basicConfig(
        format="[%(filename)s:%(lineno)d:%(funcName)s] %(message)s",
        level=logging.DEBUG,
    )

    if len(sys.argv) < 2:
        logger.error("Usage: python s0_preprocess.py <video_path> [frame_step] [output_dir] [target_height=None]")
        sys.exit(1)

    video_path = sys.argv[1]
    frame_step = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    output_dir = sys.argv[3] if len(sys.argv) > 3 else Path("frames")
    target_height = int(sys.argv[4]) if len(sys.argv) > 4 else None

    frames = extract_frames(video_path, frame_step, output_dir, target_height)
    logger.info("Done")
