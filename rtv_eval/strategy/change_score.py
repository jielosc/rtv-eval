from __future__ import annotations

import cv2
import numpy as np

# SSIM stabilization constants for 8-bit dynamic range (L = 255), per Wang et al.
_C1 = (0.01 * 255) ** 2
_C2 = (0.03 * 255) ** 2


def frame_difference(prev_gray: np.ndarray, cur_gray: np.ndarray) -> float:
    """Mean absolute pixel difference between two grayscale frames, normalized to [0, 1].

    Cheap on-device proxy for temporal change. Inputs are uint8 grayscale of the
    same shape; the 255 division puts the result on the same [0, 1] scale as the
    SSIM-change term so the two can be weighted directly.
    """
    diff = cv2.absdiff(prev_gray, cur_gray)
    return float(diff.mean()) / 255.0


def ssim(prev_gray: np.ndarray, cur_gray: np.ndarray) -> float:
    """Structural similarity index between two grayscale frames, in [-1, 1] (≈[0, 1] in practice).

    Standard single-scale SSIM (Wang et al. 2004) with an 11x11 Gaussian window,
    sigma 1.5. Implemented with OpenCV/numpy only — no scikit-image dependency —
    to stay light enough to mirror an on-device computation.
    """
    a = prev_gray.astype(np.float64)
    b = cur_gray.astype(np.float64)

    kernel = (11, 11)
    sigma = 1.5
    mu_a = cv2.GaussianBlur(a, kernel, sigma)
    mu_b = cv2.GaussianBlur(b, kernel, sigma)

    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = cv2.GaussianBlur(a * a, kernel, sigma) - mu_a_sq
    sigma_b_sq = cv2.GaussianBlur(b * b, kernel, sigma) - mu_b_sq
    sigma_ab = cv2.GaussianBlur(a * b, kernel, sigma) - mu_ab

    numerator = (2 * mu_ab + _C1) * (2 * sigma_ab + _C2)
    denominator = (mu_a_sq + mu_b_sq + _C1) * (sigma_a_sq + sigma_b_sq + _C2)
    ssim_map = numerator / denominator
    return float(ssim_map.mean())


def change_score(
    prev_gray: np.ndarray,
    cur_gray: np.ndarray,
    w_frame_diff: float = 0.5,
    w_ssim: float = 0.5,
) -> float:
    """Weighted change score C = w_frame_diff * D + w_ssim * (1 - SSIM), clamped to [0, 1].

    D is the normalized frame difference; (1 - SSIM) is the SSIM change. Both
    terms live on [0, 1] in range, but NOT in distribution — see `change_score_norm`
    for the calibrated variant used by the adaptive strategy. This raw form is kept
    for direct metric inspection and tests.
    """
    d = frame_difference(prev_gray, cur_gray)
    s_change = 1.0 - ssim(prev_gray, cur_gray)
    score = w_frame_diff * d + w_ssim * s_change
    return float(min(1.0, max(0.0, score)))


def _robust_scale(value: float, lo: float, hi: float) -> float:
    """Map `value` from [lo, hi] onto [0, 1] (robust min-max), clamped."""
    if hi <= lo:
        return 0.0
    return float(min(1.0, max(0.0, (value - lo) / (hi - lo))))


def change_score_norm(
    prev_gray: np.ndarray,
    cur_gray: np.ndarray,
    w_frame_diff: float,
    w_ssim: float,
    d_lo: float,
    d_hi: float,
    s_lo: float,
    s_hi: float,
) -> float:
    """Calibrated change score for tier decisions.

    Each metric is robust-min-max normalized to [0, 1] using calibration bounds
    (measured percentiles) BEFORE weighting, so weights are meaningful despite the
    two metrics' very different empirical spreads. The weighted sum is renormalized
    by total weight so C stays in [0, 1] and comparable across weight choices.

    When w_ssim == 0, SSIM is not computed at all (the expensive term is skipped),
    which is the default on-device path: frame difference alone carries nearly all
    the tier-ordering signal (Spearman 0.97 vs the combined metric).
    """
    total_w = w_frame_diff + w_ssim
    if total_w <= 0:
        return 0.0

    d = frame_difference(prev_gray, cur_gray)
    d_n = _robust_scale(d, d_lo, d_hi)
    acc = w_frame_diff * d_n

    if w_ssim > 0:
        s_change = 1.0 - ssim(prev_gray, cur_gray)
        s_n = _robust_scale(s_change, s_lo, s_hi)
        acc += w_ssim * s_n

    return float(min(1.0, max(0.0, acc / total_w)))
