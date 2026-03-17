"""
Simple script to extract the first N frames from a video file, optionally
with an interval (e.g. every 5th frame).

Usage:
    # first 10 consecutive frames
    python extract_first_n_frames.py /path/to/video.mp4 10

    # 10 frames, one every 5 frames (i.e., frames 0, 5, 10, ...)
    python extract_first_n_frames.py /path/to/video.mp4 10 --interval 5

    # 50 frames evenly distributed across entire video (auto interval)
    python extract_first_n_frames.py /path/to/video.mp4 50 --interval -1

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

    If `interval` == -1, automatically calculate interval to evenly distribute
    n frames across the entire video (including first and last frames).
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

    # Auto-calculate interval if requested
    if interval == -1:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            raise RuntimeError(f"Could not determine frame count for: {video_path}")

        # Calculate evenly spaced frame indices including first and last
        if n == 1:
            frame_indices = [0]
        elif n >= total_frames:
            # Extract all frames if n >= total frames
            frame_indices = list(range(total_frames))
        else:
            # Evenly distribute n frames across total_frames, including first and last
            frame_indices = [
                int(round(i * (total_frames - 1) / (n - 1)))
                for i in range(n)
            ]

        # Extract specific frame indices
        saved_count = 0
        frame_index = 0
        indices_set = set(frame_indices)

        try:
            while frame_index < max(frame_indices) + 1 if frame_indices else 0:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_index in indices_set:
                    saved_count += 1
                    out_path = output_dir / f"frame_{saved_count:04d}.png"
                    cv2.imwrite(str(out_path), frame)

                frame_index += 1
        finally:
            cap.release()

        print(f"Saved {saved_count} frame(s) to {output_dir}")
        print(f"Auto-calculated interval: extracted frames at indices {frame_indices[:5]}{'...' if len(frame_indices) > 5 else ''}")

    else:
        # Original interval-based extraction
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
        help="Frame interval. 1 = every frame, 5 = every 5th frame, -1 = auto-calculate to evenly distribute N frames across entire video (including first and last).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    n = max(0, args.n)
    output_dir = Path(args.out) if args.out is not None else None
    interval = args.interval

    if n == 0:
        print("Requested 0 frames; nothing to do.")
        return

    extract_first_n_frames(video_path, n, output_dir, interval=interval)


if __name__ == "__main__":
    main()

