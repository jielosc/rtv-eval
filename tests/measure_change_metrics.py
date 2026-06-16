"""
Measure the empirical distributions of frame difference (D) and SSIM change (S)
on real MotionBench videos, plus their correlation.

Goal: decide whether 0.5/0.5 weighting is meaningful, whether D and S are
redundant, and what calibration constants (mean/std, percentiles) the adaptive
strategy should use.

Usage: python tests/measure_change_metrics.py [num_videos]
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rtv_eval.strategy.change_score import frame_difference, ssim

VIDEO_ROOT = Path("/Users/jielosc/Research/MotionBench/hf_download/MotionBench")
ANALYSIS_W, ANALYSIS_H = 224, 126
ANALYSIS_FPS = 8.0


def analysis_frames(video_path: Path) -> list[np.ndarray]:
    """Decode the low-res grayscale analysis stream at ANALYSIS_FPS."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total <= 0:
        cap.release()
        return []
    step = max(1, int(round(src_fps / ANALYSIS_FPS)))
    want = set(range(0, total, step))
    grays: list[np.ndarray] = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i in want:
            small = cv2.resize(frame, (ANALYSIS_W, ANALYSIS_H), interpolation=cv2.INTER_AREA)
            grays.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        i += 1
    cap.release()
    return grays


def pct(arr: np.ndarray, p: float) -> float:
    return float(np.percentile(arr, p))


def describe(name: str, arr: np.ndarray) -> None:
    print(f"  {name}:")
    print(f"    mean={arr.mean():.4f}  std={arr.std():.4f}  cv(std/mean)={arr.std() / (arr.mean() + 1e-9):.2f}")
    print(f"    min={arr.min():.4f}  p5={pct(arr, 5):.4f}  p25={pct(arr, 25):.4f}  "
          f"median={pct(arr, 50):.4f}  p75={pct(arr, 75):.4f}  p95={pct(arr, 95):.4f}  max={arr.max():.4f}")


def main() -> None:
    n_videos = int(sys.argv[1]) if len(sys.argv) > 1 else 60

    all_videos = sorted(VIDEO_ROOT.rglob("*.mp4"))
    rng = random.Random(0)
    sample = rng.sample(all_videos, min(n_videos, len(all_videos)))

    d_vals: list[float] = []
    s_vals: list[float] = []
    used = 0

    for vp in sample:
        grays = analysis_frames(vp)
        if len(grays) < 2:
            continue
        used += 1
        for a, b in zip(grays[:-1], grays[1:]):
            d_vals.append(frame_difference(a, b))
            s_vals.append(1.0 - ssim(a, b))

    d = np.array(d_vals)
    s = np.array(s_vals)

    print(f"\n=== Change metric distributions ===")
    print(f"Videos used: {used}/{len(sample)} sampled  |  {len(d)} frame-pairs\n")
    describe("D  (frame difference)", d)
    describe("S  (1 - SSIM)", s)

    # Correlation
    pearson = float(np.corrcoef(d, s)[0, 1])
    # Spearman via rank
    dr = np.argsort(np.argsort(d))
    sr = np.argsort(np.argsort(s))
    spearman = float(np.corrcoef(dr, sr)[0, 1])
    print(f"\n  Correlation D vs S:  Pearson={pearson:.3f}  Spearman={spearman:.3f}")

    # How much does each drive the equal-weight sum C = 0.5D + 0.5S?
    print(f"\n  Variance contribution to C = 0.5*D + 0.5*S:")
    var_d = (0.5 ** 2) * d.var()
    var_s = (0.5 ** 2) * s.var()
    print(f"    0.5*D var share = {var_d / (var_d + var_s) * 100:.1f}%")
    print(f"    0.5*S var share = {var_s / (var_d + var_s) * 100:.1f}%")

    # Calibration constants for z-score normalization (option A)
    print(f"\n  Calibration constants (option A, z-score):")
    print(f"    D: mu={d.mean():.4f} sigma={d.std():.4f}")
    print(f"    S: mu={s.mean():.4f} sigma={s.std():.4f}")
    print(f"  Robust min-max (p5..p95):")
    print(f"    D: lo={pct(d, 5):.4f} hi={pct(d, 95):.4f}")
    print(f"    S: lo={pct(s, 5):.4f} hi={pct(s, 95):.4f}")

    # Persist raw values for plotting.
    out = Path(__file__).resolve().parent.parent / "results" / "change_metrics.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, d=d, s=s, used=used, n_videos=len(sample))
    print(f"\n  Raw values saved to: {out}")


if __name__ == "__main__":
    main()
