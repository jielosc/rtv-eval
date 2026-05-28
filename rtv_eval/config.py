from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class FrameSamplingConfig(BaseModel):
    mode: Literal["fps", "count"]
    fps: float | None = None
    count: int | None = None


class ResolutionConfig(BaseModel):
    mode: Literal["preset", "exact"]
    preset: str | None = None       # "360p", "480p", "720p", "1080p"
    width: int | None = None
    height: int | None = None


class StrategyConfig(BaseModel):
    name: str
    frame_sampling: FrameSamplingConfig
    resolution: ResolutionConfig


class ModelConfig(BaseModel):
    name: str
    base_url: str
    api_key_env: str                # env var name holding the API key
    model_name: str                 # model string sent to the API
    max_tokens: int = 256
    temperature: float = 0.0
    system_prompt: str | None = None
    timeout: float = 120.0
    detail: str = "low"             # image detail level: "low", "high", "auto"


class BenchmarkConfig(BaseModel):
    name: str
    type: str                       # adapter class selector, e.g. "motionbench"
    data_dir: str                   # path to metadata jsonl
    video_root: str                 # root dir to resolve video_path
    split: str = "dev"              # "dev" or "test"
    min_duration: float | None = None   # seconds, inclusive
    max_duration: float | None = None   # seconds, inclusive


class ExperimentConfig(BaseModel):
    name: str
    benchmark: BenchmarkConfig
    models: list[ModelConfig] = Field(min_length=1)
    strategies: list[StrategyConfig] = Field(min_length=1)
    concurrency: int = 5
    output_dir: str = "results"


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig(**raw)
