# Development

Use Python 3.10+ from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
rtv-eval run configs/example.yaml --dry-run
python -m build
```

The automated suite uses synthetic videos, mocked model inference, and short-lived loopback HTTP subprocesses. It needs permission to bind local ports but does not need API keys, model weights, GPUs, or MotionBench downloads. Real GPU inference and dataset evaluation must be validated separately. CI runs on Python 3.10 and 3.12.

The manual capacity probe is `python scripts/measure_frame_capacity.py`. It reads `RTV_EVAL_CONFIG` (default `configs/local.yaml`), requires a running model endpoint and real videos, and can incur API costs. It is deliberately outside automatic test discovery. Calibration utilities remain in `tests/measure_change_metrics.py` and `tests/plot_change_metrics.py`.

When changing sampling, prompt construction, or scoring, add a regression that demonstrates the old failure and update the evaluation protocol if results become incomparable. Do not commit API keys, local dataset paths, model weights, or result databases. Bug reports should include the command, sanitized configuration, Python version, and traceback.
