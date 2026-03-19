"""
Frame extraction

- One-shot with no feedback (we will do adaptive frame/bundle selection later)
"""

import cv2
from pathlib import Path
import sys
import logging

logger = logging.getLogger(__name__)

def extract_frames(video_path: str | Path, frame_step: int, output_dir: str | Path) -> list[Path]:
    """Samples every 'frame_step'-th frame from a video"""
    
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
        logger.error("Usage: python preprocess.py <video_path> [frame_step] [output_dir]")
        sys.exit(1)

    video_path = sys.argv[1]
    frame_step = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    output_dir = sys.argv[3] if len(sys.argv) > 3 else Path("frames")

    frames = extract_frames(video_path, frame_step, output_dir)
    logger.info("Done")
