from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from rtv_eval.strategy.change_score import (
    change_score,
    change_score_norm,
    frame_difference,
    ssim,
)


class ChangeScoreTest(unittest.TestCase):
    def test_identical_frames_have_zero_change(self) -> None:
        gray = np.full((126, 224), 128, dtype=np.uint8)
        self.assertEqual(frame_difference(gray, gray), 0.0)
        self.assertAlmostEqual(ssim(gray, gray), 1.0, places=5)
        self.assertAlmostEqual(change_score(gray, gray), 0.0, places=5)

    def test_frame_difference_is_normalized(self) -> None:
        black = np.zeros((126, 224), dtype=np.uint8)
        white = np.full((126, 224), 255, dtype=np.uint8)
        # Max possible mean abs diff is 255 -> normalized to 1.0
        self.assertAlmostEqual(frame_difference(black, white), 1.0, places=5)

    def test_change_score_bounded(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.integers(0, 256, size=(126, 224), dtype=np.uint8)
        b = rng.integers(0, 256, size=(126, 224), dtype=np.uint8)
        c = change_score(a, b)
        self.assertGreaterEqual(c, 0.0)
        self.assertLessEqual(c, 1.0)

    def test_weights_select_metric(self) -> None:
        black = np.zeros((126, 224), dtype=np.uint8)
        white = np.full((126, 224), 255, dtype=np.uint8)
        # All weight on frame diff -> equals normalized frame difference (1.0)
        only_diff = change_score(black, white, w_frame_diff=1.0, w_ssim=0.0)
        self.assertAlmostEqual(only_diff, 1.0, places=5)


class ChangeScoreNormTest(unittest.TestCase):
    BOUNDS = dict(d_lo=0.0013, d_hi=0.0904, s_lo=0.0040, s_hi=0.5890)

    def test_pure_d_skips_ssim(self) -> None:
        # With w_ssim=0, SSIM must not be consulted; patching it to explode
        # proves it is never called on the default path.
        a = np.full((126, 224), 50, dtype=np.uint8)
        b = np.full((126, 224), 60, dtype=np.uint8)
        with mock.patch("rtv_eval.strategy.change_score.ssim", side_effect=AssertionError):
            c = change_score_norm(a, b, w_frame_diff=1.0, w_ssim=0.0, **self.BOUNDS)
        self.assertGreaterEqual(c, 0.0)
        self.assertLessEqual(c, 1.0)

    def test_normalized_output_in_unit_range(self) -> None:
        black = np.zeros((126, 224), dtype=np.uint8)
        white = np.full((126, 224), 255, dtype=np.uint8)
        c = change_score_norm(black, white, w_frame_diff=0.5, w_ssim=0.5, **self.BOUNDS)
        self.assertGreaterEqual(c, 0.0)
        self.assertLessEqual(c, 1.0)

    def test_robust_scale_clamps_above_hi(self) -> None:
        # A frame diff well above d_hi must clamp to 1.0, not exceed it.
        black = np.zeros((126, 224), dtype=np.uint8)
        white = np.full((126, 224), 255, dtype=np.uint8)  # D = 1.0 >> d_hi
        c = change_score_norm(black, white, w_frame_diff=1.0, w_ssim=0.0, **self.BOUNDS)
        self.assertAlmostEqual(c, 1.0, places=5)

    def test_zero_weights_return_zero(self) -> None:
        a = np.zeros((126, 224), dtype=np.uint8)
        b = np.full((126, 224), 128, dtype=np.uint8)
        self.assertEqual(
            change_score_norm(a, b, w_frame_diff=0.0, w_ssim=0.0, **self.BOUNDS), 0.0
        )


if __name__ == "__main__":
    unittest.main()
