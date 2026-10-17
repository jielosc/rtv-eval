from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConfigModel(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)


class FrameSamplingConfig(ConfigModel):
    mode: Literal["fps", "count"]
    fps: float | None = Field(default=None, gt=0)
    count: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _check_mode(self):
        if getattr(self, self.mode) is None:
            raise ValueError(f"Frame sampling mode '{self.mode}' requires {self.mode}")
        return self


class ResolutionConfig(ConfigModel):
    mode: Literal["preset", "exact"]
    preset: Literal["360p", "480p", "720p", "1080p"] | None = None
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _check_mode(self):
        if self.mode == "preset" and self.preset is None:
            raise ValueError("Preset resolution requires preset")
        if self.mode == "exact" and (self.width is None or self.height is None):
            raise ValueError("Exact resolution requires width and height")
        return self


class AdaptiveTier(ConfigModel):
    """One upload tier: how a window is encoded once its change score lands here.

    A tier couples upload frame rate, resolution, and JPEG quality. The mapping
    is fully configurable so the resolution/quality coupling direction (e.g.
    high-motion -> low-quality) can be reversed for experiments.
    """

    name: str                        # label, e.g. "low" / "mid" / "high"
    min_score: float = Field(default=0.0, ge=0, le=1)           # inclusive lower bound on window change score C
    fps: float = Field(gt=0)         # upload frame rate for windows in this tier
    resolution: ResolutionConfig     # resolution applied to this tier's frames
    quality: int = Field(default=85, ge=1, le=100)                # JPEG quality (1-100) for this tier's frames


class AdaptiveConfig(ConfigModel):
    """Content-adaptive upload policy driven by frame-difference + SSIM change.

    On-device simulation: the analysis stream is decoded at low resolution and a
    low analysis fps. For each adjacent analysis-frame pair we compute a change
    score C = w_frame_diff * D + w_ssim * (1 - SSIM), both normalized to [0, 1].
    The video is split into fixed-length windows; each window's mean C is mapped
    to a tier whose lower bound it clears, and the window's frames are uploaded
    at that tier's fps / resolution / quality.
    """

    analysis_width: int = Field(default=224, gt=0)        # low-res analysis stream width
    analysis_height: int = Field(default=126, gt=0)       # low-res analysis stream height
    analysis_fps: float = Field(default=8.0, gt=0)        # frames per second sampled for analysis (5-10 typical)
    window_seconds: float = Field(default=1.0, gt=0)      # window length for tier decisions
    # D (frame difference) alone carries almost all the tier-ordering signal:
    # on 300 MotionBench videos (16802 frame-pairs) D vs (1-SSIM) ranked at
    # Spearman 0.97, so SSIM is redundant for tier assignment. Default is pure D
    # (w_ssim=0), which also skips the SSIM compute. Raw D and raw (1-SSIM) live
    # on very different spreads (std(S) ~= 5.7x std(D)), so a raw 0.5/0.5 sum is
    # ~97% driven by S. When w_ssim>0 each metric is robust-min-max normalized to
    # [0,1] using the calibration bounds below *before* weighting, so the weights
    # are actually meaningful and C stays comparable across both modes.
    w_frame_diff: float = Field(default=1.0, ge=0)        # weight on (normalized) frame difference D
    w_ssim: float = Field(default=0.0, ge=0)              # weight on (normalized) SSIM change; 0 => skip SSIM
    smoothing: float = Field(default=0.0, ge=0, lt=1)           # EMA factor in [0, 1) for C; 0 disables
    # Robust-min-max calibration bounds (p5..p95 from the 300-video measurement).
    # Each metric is scaled (value - lo) / (hi - lo) and clamped to [0,1].
    d_norm_lo: float = 0.0013        # D p5
    d_norm_hi: float = 0.0904        # D p95
    s_norm_lo: float = 0.0040        # (1-SSIM) p5
    s_norm_hi: float = 0.5890        # (1-SSIM) p95
    # Tiers ordered low -> high change. `min_score` is the inclusive lower bound
    # on the window's normalized change score C in [0,1]; the highest-bound tier
    # whose min_score <= C wins. The first tier should use min_score 0.0.
    tiers: list[AdaptiveTier] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_policy(self):
        if self.w_frame_diff + self.w_ssim <= 0:
            raise ValueError("At least one change-score weight must be positive")
        if self.d_norm_hi <= self.d_norm_lo or self.s_norm_hi <= self.s_norm_lo:
            raise ValueError("Normalization upper bounds must exceed lower bounds")
        scores = [tier.min_score for tier in self.tiers]
        if scores[0] != 0 or any(a >= b for a, b in zip(scores, scores[1:])):
            raise ValueError("Tier thresholds must start at 0 and strictly increase")
        return self


class StrategyConfig(ConfigModel):
    name: str
    frame_sampling: FrameSamplingConfig | None = None
    resolution: ResolutionConfig | None = None
    adaptive: AdaptiveConfig | None = None

    @model_validator(mode="after")
    def _check_exactly_one_path(self) -> "StrategyConfig":
        is_adaptive = self.adaptive is not None
        is_static = self.frame_sampling is not None or self.resolution is not None
        if is_adaptive and is_static:
            raise ValueError(
                f"Strategy '{self.name}': set either 'adaptive' or "
                "'frame_sampling'+'resolution', not both."
            )
        if not is_adaptive:
            if self.frame_sampling is None or self.resolution is None:
                raise ValueError(
                    f"Strategy '{self.name}': static strategy requires both "
                    "'frame_sampling' and 'resolution'."
                )
        return self


class LocalServingConfig(ConfigModel):
    python_executable: str | None = None
    device: str = "cuda:0"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    attn_implementation: str = "sdpa"
    cache_dir: str | None = None
    startup_timeout: float = Field(default=600.0, gt=0)
    shutdown_timeout: float = Field(default=15.0, gt=0)


class ModelConfig(ConfigModel):
    name: str
    backend: Literal["openai", "local"] = "openai"
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"  # unused for the temporary local server
    local: LocalServingConfig = Field(default_factory=LocalServingConfig)
    model_name: str                 # model string sent to the API
    max_tokens: int = Field(default=256, gt=0)
    temperature: float = 0.0
    system_prompt: str | None = None
    timeout: float = Field(default=120.0, gt=0)
    detail: str = "low"             # image detail level: "low", "high", "auto"

    @model_validator(mode="after")
    def _check_backend(self) -> "ModelConfig":
        if self.backend == "openai" and not self.base_url:
            raise ValueError("OpenAI backend requires base_url")
        if self.backend == "local" and self.base_url:
            raise ValueError("Local backend allocates its own endpoint; omit base_url")
        return self


class BenchmarkConfig(ConfigModel):
    name: str
    type: str                       # adapter class selector, e.g. "motionbench"
    data_dir: str                   # path to metadata jsonl
    video_root: str                 # root dir to resolve video_path
    split: Literal["dev", "test"] = "dev"              # "dev" or "test"
    min_duration: float | None = None   # seconds, inclusive
    max_duration: float | None = None   # seconds, inclusive


class ExperimentConfig(ConfigModel):
    name: str
    benchmark: BenchmarkConfig
    models: list[ModelConfig] = Field(min_length=1)
    strategies: list[StrategyConfig] = Field(min_length=1)
    concurrency: int = Field(default=5, gt=0)
    output_dir: str = "results"

    @model_validator(mode="after")
    def _check_names(self):
        for label, items in (("models", self.models), ("strategies", self.strategies)):
            names = [item.name for item in items]
            if len(names) != len(set(names)):
                raise ValueError(f"Duplicate names in {label}")
        return self


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig.model_validate(raw)
