from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

import openai
from tqdm import tqdm

from rtv_eval.benchmarks.base import Benchmark, QAEntry
from rtv_eval.benchmarks.motionbench import MotionBench
from rtv_eval.config import ExperimentConfig, ModelConfig, StrategyConfig
from rtv_eval.models.base import ModelCaller
from rtv_eval.models.openai_compat import OpenAICompatCaller
from rtv_eval.models.managed import model_session
from rtv_eval.scoring.exact_match import ExactMatchScorer
from rtv_eval.storage.db import Database
from rtv_eval.strategy.preprocessor import VideoPreprocessor

logger = logging.getLogger(__name__)


def _load_benchmark(config: ExperimentConfig) -> Benchmark:
    if config.benchmark.type == "motionbench":
        return MotionBench()
    raise ValueError(f"Unknown benchmark type: {config.benchmark.type}")


def _load_caller(config: ModelConfig) -> ModelCaller:
    return OpenAICompatCaller(config)


class Runner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.db = Database(Path(config.output_dir) / f"{config.name}.db")
        self.benchmark = _load_benchmark(config)

    async def run(self) -> None:
        entries = self.benchmark.load(self.config.benchmark)
        if not entries:
            logger.info("No evaluation entries; no model will be started.")
            return
        video_root = Path(self.config.benchmark.video_root)
        video_index = self.benchmark.build_video_index(video_root)

        total_combos = len(self.config.models) * len(self.config.strategies)
        logger.info(
            "Experiment '%s': %d models x %d strategies = %d combinations, %d QA entries",
            self.config.name,
            len(self.config.models),
            len(self.config.strategies),
            total_combos,
            len(entries),
        )

        for model_cfg in self.config.models:
            run_ids = [self._get_run_id(model_cfg, strategy) for strategy in self.config.strategies]
            expected = {entry.uid for entry in entries}
            if all(expected <= self.db.get_completed_uids(run_id) for run_id in run_ids):
                logger.info("[%s] All entries already recorded; skipping model startup.", model_cfg.name)
                continue
            async with model_session(model_cfg, Path(self.config.output_dir)) as caller_config:
                for strategy_cfg in self.config.strategies:
                    await self._run_combo(
                        model_cfg, strategy_cfg, entries, video_index, caller_config=caller_config,
                    )

    def _get_run_id(self, model_cfg: ModelConfig, strategy_cfg: StrategyConfig) -> int:
        config_snapshot = json.dumps({
            # Changing sampling or prompt/scoring semantics must not resume legacy results.
            "evaluation_protocol": "mcq-letter-v3",
            "model": model_cfg.model_dump(),
            "strategy": strategy_cfg.model_dump(),
            "benchmark": self.config.benchmark.model_dump(),
        })
        return self.db.get_or_create_run(
            self.config.name,
            model_cfg.name,
            strategy_cfg.name,
            self.config.benchmark.name,
            config_snapshot,
        )

    async def _run_combo(
        self,
        model_cfg: ModelConfig,
        strategy_cfg: StrategyConfig,
        entries: list[QAEntry],
        video_index: dict[str, Path],
        *, caller_config: ModelConfig | None = None,
    ) -> None:
        run_id = self._get_run_id(model_cfg, strategy_cfg)
        completed = self.db.get_completed_uids(run_id)
        pending = [e for e in entries if e.uid not in completed]

        label = f"{model_cfg.name} x {strategy_cfg.name}"
        if not pending:
            logger.info("[%s] All %d entries already completed, skipping.", label, len(entries))
            return

        logger.info("[%s] %d pending / %d total", label, len(pending), len(entries))

        preprocessor = VideoPreprocessor(strategy_cfg)
        caller = _load_caller(caller_config or model_cfg)
        scorer = ExactMatchScorer()
        semaphore = asyncio.Semaphore(self.config.concurrency)

        # Update run status back to running
        self.db.mark_run_running(run_id)

        pbar = tqdm(total=len(pending), desc=label)
        fatal_connection_error = asyncio.Event()

        async def process_one(entry: QAEntry) -> None:
            async with semaphore:
                if fatal_connection_error.is_set():
                    pbar.update(1)
                    return

                video_path = video_index.get(entry.video_path)
                if video_path is None:
                    logger.warning("Video not found: %s, skipping", entry.video_path)
                    pbar.update(1)
                    return

                try:
                    frames = preprocessor.process(video_path)
                    if not frames:
                        logger.warning("No frames from %s, skipping", entry.video_path)
                        return

                    t0 = time.monotonic()
                    response = await caller.call(entry.question, frames)
                    latency = (time.monotonic() - t0) * 1000

                    valid_options = entry.metadata.get("valid_options", ("A", "B", "C", "D"))
                    if entry.answer is not None:
                        extracted, correct = scorer.score(
                            response, entry.answer, valid_options=valid_options,
                        )
                    else:
                        # test split: no ground truth, only extract the predicted letter
                        extracted = ExactMatchScorer._extract_letter(
                            response, valid_options=valid_options,
                        )
                        correct = None

                    self.db.save_result(
                        run_id=run_id,
                        qa_uid=entry.uid,
                        question_type=entry.metadata.get("question_type"),
                        video_path=str(video_path),
                        raw_response=response,
                        extracted_answer=extracted,
                        ground_truth=entry.answer,
                        is_correct=correct,
                        latency_ms=latency,
                        num_frames=len(frames),
                        answer_status="valid" if extracted else "invalid_format",
                    )
                except openai.APIConnectionError:
                    fatal_connection_error.set()
                    logger.exception("Connection error processing %s", entry.uid)
                except Exception:
                    logger.exception("Error processing %s", entry.uid)
                finally:
                    pbar.update(1)

        # Run with concurrency control
        tasks = [asyncio.create_task(process_one(e)) for e in pending]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.db.mark_run_failed(run_id)
            raise
        finally:
            pbar.close()
            close = getattr(caller, "close", None)
            if close is not None:
                await close()

        missing = {entry.uid for entry in entries} - self.db.get_completed_uids(run_id)
        if missing:
            self.db.mark_run_failed(run_id)
            logger.error("[%s] Incomplete: %d entries missing; rerun to retry.", label, len(missing))
            return

        self.db.mark_run_completed(run_id)
        summary = self.db.get_accuracy_summary(run_id)
        logger.info("[%s] Completed. Accuracy: %.1f%% (%d/%d)", label, summary["accuracy"] * 100, summary["correct"], summary["total"])
