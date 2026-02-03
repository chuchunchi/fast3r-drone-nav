"""
Simple script to extract the first N frames from a video file, optionally
with an interval (e.g. every 5th frame).

Usage:
    # first 10 consecutive frames
    python extract_first_n_frames.py /path/to/video.mp4 10

    # 10 frames, one every 5 frames (i.e., frames 0, 5, 10, ...)
    python extract_first_n_frames.py /path/to/video.mp4 10 --interval 5

This will write frames as:
    frame_0001.png, frame_0002.png, ...
in a folder next to the video called `<video_name>_frames`.
"""

import argparse
import os
from pathlib import Path

import cv2


def extract_first_n_frames(
    video_path: Path,
    n: int,
    output_dir: Path | None = None,
    interval: int = 1,
) -> None:
    """Extract the first `n` frames from `video_path` into `output_dir`.

    If `interval` > 1, save frames spaced by that many frames, e.g.
    interval=5 saves frames 0, 5, 10, ...
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_dir is None:
        output_dir = video_path.with_suffix("")  # remove extension
        output_dir = output_dir.parent / f"{output_dir.name}_frames"

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    saved_count = 0
    frame_index = 0
    interval = max(1, interval)
    try:
        while saved_count < n:
            ret, frame = cap.read()
            if not ret:
                # End of video
                break

            if frame_index % interval == 0:
                saved_count += 1
                out_path = output_dir / f"frame_{saved_count:04d}.png"
                cv2.imwrite(str(out_path), frame)

            frame_index += 1
    finally:
        cap.release()

    print(f"Saved {saved_count} frame(s) to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the first N frames from a video, optionally with an interval."
    )
    parser.add_argument("video", type=str, help="Path to the input video file.")
    parser.add_argument("n", type=int, help="Number of frames to extract.")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output directory. Defaults to `<video_name>_frames` next to the video.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="Frame interval. 1 = every frame, 5 = every 5th frame, etc.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    n = max(0, args.n)
    output_dir = Path(args.out) if args.out is not None else None
    interval = max(1, args.interval)

    if n == 0:
        print("Requested 0 frames; nothing to do.")
        return

    extract_first_n_frames(video_path, n, output_dir, interval=interval)


if __name__ == "__main__":
    main()

