#!/usr/bin/env python3
"""Benchmark: pairwise cached vs. uncached Fast3R inference.

Covers spec sections 8.5 (wall-clock latency) and 8.7 (per-stage profiling
breakdown).  Outputs a results table suitable for thesis inclusion.

Usage:
    python visual_homing/scripts/benchmark_cached_inference.py [--N 50] [--warmup 10]
"""

import argparse
import io
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fast3r.models.fast3r import Fast3R


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@contextmanager
def _suppress_stdout():
    """Suppress stdout to hide profiling prints from Fast3R.forward()."""
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old

def _make_views(device: str = "cuda", dtype=torch.float32):
    """Create two random 512x384 view dicts (matching deployed resolution)."""
    views = []
    for _ in range(2):
        img = torch.randn(1, 3, 384, 512, device=device, dtype=dtype)
        view = {
            "img": img,
            "true_shape": torch.tensor([[384, 512]], device=device),
        }
        views.append(view)
    return views


def _load_model():
    model = Fast3R.from_pretrained("jedyang97/Fast3R_ViT_Large_512")
    model = model.to("cuda")
    model.eval()
    return model


def _gpu_info() -> str:
    name = torch.cuda.get_device_name(0)
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"{name} ({mem_gb:.1f} GB)"


# ---------------------------------------------------------------------------
# Section 8.5 — Wall-clock latency comparison
# ---------------------------------------------------------------------------

def benchmark_wall_clock(model, N: int, warmup: int):
    """Measure end-to-end latency for uncached and cached paths."""
    view_live, view_target = _make_views()

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        # Warmup both paths
        for _ in range(warmup):
            model.forward([view_live, view_target])
        cached = model.encode_image(view_target)
        for _ in range(warmup):
            model.forward_pair_cached(view_live, cached)

        # Benchmark: uncached
        times_uncached = []
        for _ in range(N):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.forward([view_live, view_target])
            torch.cuda.synchronize()
            times_uncached.append(time.perf_counter() - t0)

        # Benchmark: cached
        cached = model.encode_image(view_target)
        times_cached = []
        for _ in range(N):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.forward_pair_cached(view_live, cached)
            torch.cuda.synchronize()
            times_cached.append(time.perf_counter() - t0)

        # Benchmark: encode_image alone (one-time cost)
        times_encode = []
        for _ in range(N):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.encode_image(view_target)
            torch.cuda.synchronize()
            times_encode.append(time.perf_counter() - t0)

    return {
        "uncached_ms": np.array(times_uncached) * 1000,
        "cached_ms": np.array(times_cached) * 1000,
        "encode_ms": np.array(times_encode) * 1000,
    }


# ---------------------------------------------------------------------------
# Section 8.7 — Per-stage profiling breakdown
# ---------------------------------------------------------------------------

def benchmark_profiling(model, N: int, warmup: int):
    """Collect per-stage timings using the profiling=True flag."""
    view_live, view_target = _make_views()

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for _ in range(warmup):
            model.forward([view_live, view_target])
        cached = model.encode_image(view_target)
        for _ in range(warmup):
            model.forward_pair_cached(view_live, cached)

        # Profiling: uncached (suppress forward()'s debug prints)
        uncached_stages = []
        for _ in range(N):
            with _suppress_stdout():
                _, info = model.forward([view_live, view_target], profiling=True)
            uncached_stages.append(info)

        # Profiling: cached
        cached = model.encode_image(view_target)
        cached_stages = []
        for _ in range(N):
            with _suppress_stdout():
                _, info = model.forward_pair_cached(view_live, cached, profiling=True)
            cached_stages.append(info)

    def _aggregate(records, key):
        vals = [r[key] * 1000 for r in records if key in r]
        if not vals:
            return None
        return {"mean": np.mean(vals), "std": np.std(vals)}

    stage_keys = ["encode_images_time", "decoder_time", "head_forward_time", "total_time"]
    result = {}
    for key in stage_keys:
        result[key] = {
            "uncached": _aggregate(uncached_stages, key),
            "cached": _aggregate(cached_stages, key),
        }
    return result


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_wall_clock(results: dict):
    u = results["uncached_ms"]
    c = results["cached_ms"]
    e = results["encode_ms"]
    speedup = (1 - np.mean(c) / np.mean(u)) * 100

    print("\n" + "=" * 64)
    print("  WALL-CLOCK LATENCY (Spec §8.5)")
    print("=" * 64)
    print(f"  {'Metric':<30} {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print(f"  {'-'*30} {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for label, arr in [
        ("Uncached forward (ms)", u),
        ("Cached forward (ms)", c),
        ("encode_image only (ms)", e),
    ]:
        print(
            f"  {label:<30} {np.mean(arr):8.2f}  {np.std(arr):8.2f}"
            f"  {np.min(arr):8.2f}  {np.max(arr):8.2f}"
        )
    print(f"  {'-'*30} {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    print(f"  {'Speedup':<30} {speedup:7.1f}%")
    print(f"  {'Saved per frame':<30} {np.mean(u) - np.mean(c):7.2f} ms")
    print()


def print_profiling(stages: dict):
    print("=" * 64)
    print("  PER-STAGE PROFILING BREAKDOWN (Spec §8.7)")
    print("=" * 64)
    print(
        f"  {'Stage':<25} {'Uncached':>10} {'Cached':>10}"
        f" {'Delta':>10} {'Δ%':>8}"
    )
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    for key in ["encode_images_time", "decoder_time", "head_forward_time", "total_time"]:
        u_stats = stages[key]["uncached"]
        c_stats = stages[key]["cached"]
        if u_stats is None or c_stats is None:
            continue
        u_ms = u_stats["mean"]
        c_ms = c_stats["mean"]
        delta = c_ms - u_ms
        delta_pct = (delta / u_ms) * 100 if u_ms > 0 else 0

        label = key.replace("_time", "").replace("_", " ")
        print(
            f"  {label:<25} {u_ms:9.2f}ms {c_ms:9.2f}ms"
            f" {delta:+9.2f}ms {delta_pct:+7.1f}%"
        )
    print()


def print_markdown_table(stages: dict):
    """Print a Markdown table for direct thesis/paper inclusion."""
    print("=" * 64)
    print("  MARKDOWN TABLE (copy-paste into thesis)")
    print("=" * 64)
    print()
    print("| Stage | Uncached (ms) | Cached (ms) | Δ (ms) | Δ (%) |")
    print("|-------|--------------|-------------|--------|-------|")

    for key in ["encode_images_time", "decoder_time", "head_forward_time", "total_time"]:
        u_stats = stages[key]["uncached"]
        c_stats = stages[key]["cached"]
        if u_stats is None or c_stats is None:
            continue
        u_ms = u_stats["mean"]
        c_ms = c_stats["mean"]
        delta = c_ms - u_ms
        delta_pct = (delta / u_ms) * 100 if u_ms > 0 else 0

        label = key.replace("_time", "").replace("_", " ")
        if key == "total_time":
            label = "**total**"
        print(
            f"| {label} | {u_ms:.2f} ± {u_stats['std']:.2f} "
            f"| {c_ms:.2f} ± {c_stats['std']:.2f} "
            f"| {delta:+.2f} | {delta_pct:+.1f}% |"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark cached vs. uncached Fast3R pairwise inference"
    )
    parser.add_argument(
        "--N", type=int, default=50,
        help="Number of iterations per benchmark (default: 50)",
    )
    parser.add_argument(
        "--warmup", type=int, default=10,
        help="Number of warmup iterations (default: 10)",
    )
    args = parser.parse_args()

    print(f"\nGPU: {_gpu_info()}")
    print(f"Model: jedyang97/Fast3R_ViT_Large_512")
    print(f"Resolution: 512 × 384 (4:3)")
    print(f"Iterations: {args.N} (warmup: {args.warmup})")
    print(f"Autocast: float16")

    print("\nLoading model...")
    model = _load_model()

    print("Running wall-clock benchmark...")
    wc = benchmark_wall_clock(model, args.N, args.warmup)
    print_wall_clock(wc)

    print("Running per-stage profiling...")
    stages = benchmark_profiling(model, args.N, args.warmup)
    print_profiling(stages)
    print_markdown_table(stages)


if __name__ == "__main__":
    main()
