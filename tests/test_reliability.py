import asyncio
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from rtv_eval.cli import cli
from rtv_eval.config import AdaptiveConfig, ExperimentConfig, FrameSamplingConfig, ResolutionConfig
from rtv_eval.runner import Runner
from rtv_eval.strategy.frame_sampler import _sample_indices, sample_frames
from rtv_eval.strategy.resizer import resize_frame
from tests.test_adaptive_sampler import _tiers
from tests.test_runner_status import _build_config, _FakeBenchmark


@pytest.mark.parametrize('data', [dict(mode='fps'), dict(mode='count'), dict(mode='fps', fps=0),
                                  dict(mode='fps', fps=float('inf')), dict(mode='count', count=-1)])
def test_invalid_sampling_rejected(data):
    with pytest.raises(ValidationError):
        FrameSamplingConfig(**data)


@pytest.mark.parametrize('data', [dict(mode='exact', width=0, height=2), dict(mode='exact', width=2),
                                  dict(mode='preset'), dict(mode='preset', preset='bad')])
def test_invalid_resolution_rejected(data):
    with pytest.raises(ValidationError):
        ResolutionConfig(**data)


@pytest.mark.parametrize('changes', [dict(concurrency=0), dict(concurrency=-1), dict(duplicate=True)])
def test_invalid_experiment_rejected(tmp_path, changes):
    data = _build_config(tmp_path).model_dump()
    if changes.pop('duplicate', False):
        data['models'] *= 2
    data.update(changes)
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(data)


@pytest.mark.parametrize('changes', [dict(window_seconds=0), dict(smoothing=1),
    dict(w_frame_diff=0, w_ssim=0), dict(d_norm_hi=0), dict(analysis_fps=-1)])
def test_invalid_adaptive_policy_rejected(changes):
    with pytest.raises(ValidationError):
        AdaptiveConfig(tiers=_tiers(), **changes)


def test_invalid_tier_order_rejected():
    with pytest.raises(ValidationError):
        AdaptiveConfig(tiers=list(reversed(_tiers())))


def test_fractional_fps_and_source_cap():
    indices = _sample_indices(300, 30, FrameSamplingConfig(mode='fps', fps=8))
    assert len(indices) == 80
    assert len(set(indices)) == 80
    assert indices[:4] == [0, 3, 7, 11]
    assert _sample_indices(3, 30, FrameSamplingConfig(mode='fps', fps=60)) == [0, 1, 2]


def test_unknown_frame_count_still_respects_sampling():
    cap = mock.Mock()
    cap.get.side_effect = [0, 30]
    with mock.patch('rtv_eval.strategy.frame_sampler.cv2.VideoCapture', return_value=cap), \
         mock.patch('rtv_eval.strategy.frame_sampler._read_all_frames', return_value=list(range(300))):
        frames = sample_frames(Path('clip.mp4'), FrameSamplingConfig(mode='count', count=8))
    assert len(frames) == 8
    assert frames[0] == 0 and frames[-1] == 299
    cap.release.assert_called_once()


def test_extreme_aspect_ratio_remains_nonempty():
    frame = np.zeros((1000, 1, 3), dtype=np.uint8)
    result = resize_frame(frame, ResolutionConfig(mode='exact', width=1, height=1))
    assert result.shape == (1, 1, 3)


@pytest.mark.parametrize('failure', ['missing_video', 'empty_frames', 'api_error'])
def test_incomplete_run_fails_then_resumes(tmp_path, failure):
    benchmark = _FakeBenchmark()
    caller = mock.AsyncMock()
    caller.call.return_value = 'A'
    processor = mock.Mock()
    processor.process.return_value = ['frame']
    if failure == 'missing_video':
        benchmark.build_video_index = lambda _: {}
    elif failure == 'empty_frames':
        processor.process.return_value = []
    else:
        caller.call.side_effect = ValueError('bad response')
    with mock.patch('rtv_eval.runner._load_benchmark', return_value=benchmark), \
         mock.patch('rtv_eval.runner._load_caller', return_value=caller), \
         mock.patch('rtv_eval.runner.VideoPreprocessor', return_value=processor):
        runner = Runner(_build_config(tmp_path))
        try:
            asyncio.run(runner.run())
            assert runner.db.get_all_runs()[0]['status'] == 'failed'
            benchmark.build_video_index = _FakeBenchmark.build_video_index
            processor.process.return_value = ['frame']
            caller.call.side_effect = None
            asyncio.run(runner.run())
            assert runner.db.get_all_runs()[0]['status'] == 'completed'
            assert len(runner.db.get_completed_uids(1)) == 1
        finally:
            runner.db.close()


def test_cli_incomplete_run_exits_nonzero(tmp_path):
    runner = mock.Mock()
    runner.run = mock.AsyncMock()
    runner.db.get_all_runs.return_value = [{'status': 'failed'}]
    config_path = tmp_path / 'config.yaml'
    config_path.write_text('')
    with mock.patch('rtv_eval.cli.load_config', return_value=_build_config(tmp_path)), \
         mock.patch('rtv_eval.runner.Runner', return_value=runner):
        result = CliRunner().invoke(cli, ['run', str(config_path)])
    assert result.exit_code == 1
    assert 'incomplete' in result.output
    assert 'Done!' not in result.output
    runner.db.close.assert_called_once()


def test_cancellation_closes_caller_after_tasks_stop(tmp_path):
    caller = mock.AsyncMock()
    caller.call.side_effect = asyncio.CancelledError
    with mock.patch('rtv_eval.runner._load_benchmark', return_value=_FakeBenchmark()), \
         mock.patch('rtv_eval.runner._load_caller', return_value=caller), \
         mock.patch('rtv_eval.runner.VideoPreprocessor') as processor:
        processor.return_value.process.return_value = ['frame']
        runner = Runner(_build_config(tmp_path))
        try:
            with pytest.raises(asyncio.CancelledError):
                asyncio.run(runner.run())
            assert runner.db.get_all_runs()[0]['status'] == 'failed'
            caller.close.assert_awaited_once()
        finally:
            runner.db.close()
