# RTV-Eval

Real-Time Video Understanding Evaluation Platform. Test different models with different video preprocessing strategies (frame rate × resolution) on video understanding benchmarks.

## Install

```bash
pip install -e .
```

## Quick Start

```bash
# 1. Copy and edit config
cp configs/example.yaml configs/local.yaml
# Edit configs/local.yaml: set your models, strategies, benchmark paths

# 2. Set API key
export OPENAI_API_KEY="sk-..."

# 3. Run
rtv-eval run configs/local.yaml

# 4. View results
rtv-eval report results/my_experiment.db
rtv-eval status results/my_experiment.db
```

## Config

Config file has three sections: **benchmark**, **models**, **strategies**. The runner evaluates every model × strategy combination.

```yaml
name: "my_experiment"
concurrency: 5
output_dir: "results"

benchmark:
  name: "motionbench"
  type: "motionbench"
  data_dir: "/path/to/MotionBench/data"
  video_root: "/path/to/MotionBench/hf_download/MotionBench"
  split: "dev"              # "dev" for local scoring, "test" for leaderboard

models:
  - name: "gpt-4o"
    base_url: "https://api.openai.com/v1"
    api_key_env: "OPENAI_API_KEY"    # env var name, NOT the key itself
    model_name: "gpt-4o-2024-08-06"
    max_tokens: 256
    temperature: 0.0
    detail: "low"                    # "low" / "high" / "auto"

strategies:
  - name: "8frames_480p"
    frame_sampling:
      mode: "count"         # "count": fixed N frames evenly spaced
      count: 8
    resolution:
      mode: "preset"        # "preset": 360p / 480p / 720p / 1080p
      preset: "480p"

  - name: "1fps_720p"
    frame_sampling:
      mode: "fps"           # "fps": sample at given frame rate
      fps: 1.0
    resolution:
      mode: "preset"
      preset: "720p"

  - name: "16frames_256x256"
    frame_sampling:
      mode: "count"
      count: 16
    resolution:
      mode: "exact"         # "exact": specific pixel dimensions
      width: 256
      height: 256
```

## Strategies

The strategy layer simulates real-time video streaming constraints:

| Mode | Description | Use Case |
|------|-------------|----------|
| `count` | N evenly-spaced frames across the video | Consistent input size, good for short clips |
| `fps` | Extract at a target frame rate | Simulates real streaming (e.g., 1fps, 2fps) |

Resolution presets scale proportionally (preserve aspect ratio):

| Preset | Max Height |
|--------|------------|
| 360p | 360px |
| 480p | 480px |
| 720p | 720px |
| 1080p | 1080px |

## CLI

```bash
rtv-eval run <config.yaml>              # Run experiment
rtv-eval run <config.yaml> --dry-run    # Preview without running
rtv-eval report <results.db>            # Print comparison table
rtv-eval report <results.db> -f json    # Export as JSON
rtv-eval report <results.db> -f csv -o out.csv  # Export as CSV
rtv-eval status <results.db>            # Show run status
```

Use `-v` for debug logging: `rtv-eval -v run ...`

## Results

Results are stored in SQLite (`results/<experiment>.db`). Each run tracks:

- Per-question: raw response, extracted answer, correctness, latency, frame count
- Summary: overall accuracy + per question-type breakdown (6 types in MotionBench)

Interrupted runs resume automatically — completed entries are skipped on re-run.

## Project Structure

```
rtv_eval/
├── config.py           # Pydantic config models + YAML loader
├── runner.py           # Orchestrator: model × strategy × benchmark
├── cli.py              # CLI entry point
├── benchmarks/         # Dataset adapters (MotionBench)
├── strategy/           # Frame sampling + resolution preprocessing
├── models/             # API callers (OpenAI-compatible)
├── scoring/            # Answer extraction + matching
└── storage/            # SQLite DB + result export
```
