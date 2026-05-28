from __future__ import annotations

import json
import logging
from pathlib import Path

from rtv_eval.benchmarks.base import Benchmark, QAEntry
from rtv_eval.config import BenchmarkConfig

logger = logging.getLogger(__name__)


class MotionBench(Benchmark):
    def name(self) -> str:
        return "motionbench"

    def load(self, config: BenchmarkConfig) -> list[QAEntry]:
        meta_path = Path(config.data_dir) / "video_info.meta.jsonl"
        entries: list[QAEntry] = []

        with open(meta_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                qa = item["qa"][0]

                # Filter by split
                if config.split == "dev" and qa["answer"] == "NA":
                    continue
                if config.split == "test" and qa["answer"] != "NA":
                    continue

                entries.append(QAEntry(
                    uid=qa["uid"],
                    video_path=item["video_path"],
                    question=qa["question"],
                    answer=qa["answer"] if qa["answer"] != "NA" else None,
                    metadata={
                        "question_type": item.get("question_type"),
                        "video_type": item.get("video_type"),
                        "video_info": item.get("video_info"),
                        "key": item.get("key"),
                    },
                ))

        logger.info("Loaded %d entries from MotionBench (split=%s)", len(entries), config.split)
        return entries

    def resolve_video_path(self, video_path: str) -> Path:
        raise NotImplementedError(
            "Use resolve_video_path_cached() with a pre-built index instead."
        )

    @staticmethod
    def build_video_index(video_root: Path) -> dict[str, Path]:
        """Build filename -> full path index for O(1) lookup."""
        index: dict[str, Path] = {}
        for subdir in ["self-collected", "public-dataset"]:
            d = video_root / subdir
            if not d.exists():
                logger.warning("Video subdirectory not found: %s", d)
                continue
            for f in d.iterdir():
                if f.suffix == ".mp4":
                    index[f.name] = f
        logger.info("Built video index with %d entries", len(index))
        return index
