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
| `adaptive` | Content-adaptive fps + resolution per window | Simulates an on-device upload policy |

Resolution presets scale proportionally (preserve aspect ratio):

| Preset | Max Height |
|--------|------------|
| 360p | 360px |
| 480p | 480px |
| 720p | 720px |
| 1080p | 1080px |

### Adaptive strategy

The `adaptive` strategy simulates an on-device upload policy. The camera decodes a
cheap low-resolution analysis stream (default `224×126`, 8 fps). For each adjacent
analysis-frame pair it computes a change score:

```
D = normalized frame difference          (mean abs pixel diff / 255, in [0,1])
S = SSIM change = 1 - SSIM               (in [0,1])
C = w_frame_diff · D̂ + w_ssim · Ŝ        (each metric robust-min-max normalized first)
```

**Default is pure frame difference (`w_ssim: 0.0`).** On 300 MotionBench videos
(16802 frame-pairs), `D` and `S` rank windows almost identically (Spearman 0.97), so
SSIM adds little tier-ordering signal while costing the most compute — it is skipped
entirely when its weight is 0. The two metrics also have very different spreads
(`std(S) ≈ 5.7 × std(D)`), so a naive `0.5·D + 0.5·S` is ~97% driven by `S`. To avoid
that, when `w_ssim > 0` each metric is **robust-min-max normalized** to `[0,1]` using
calibration bounds (`*_norm_lo/hi`, measured p5..p95) *before* weighting, and the sum is
renormalized by total weight — so `C` stays in `[0,1]` and the weights are meaningful.

The video is split into fixed windows (`window_seconds`); each window's mean `C` is
mapped to the highest tier whose `min_score` it clears, and that window's frames are
uploaded at the tier's `fps`, `resolution`, and JPEG `quality`. With the default bounds
and thresholds, windows split roughly 48% / 27% / 25% across low / mid / high.

Tiers are fully configurable — including the resolution/quality coupling direction — so
you can run the default "high motion → low quality, high fps" policy or invert it. A
config uses `adaptive` *instead of* `frame_sampling` + `resolution`:

```yaml
strategies:
  - name: "adaptive_3tier"
    adaptive:
      analysis_width: 224
      analysis_height: 126
      analysis_fps: 8.0
      window_seconds: 1.0
      w_frame_diff: 1.0        # pure D by default
      w_ssim: 0.0              # >0 to add SSIM (normalized via *_norm bounds below)
      smoothing: 0.0           # EMA in [0,1) to damp tier flicker
      d_norm_lo: 0.0013        # calibration bounds (300-video p5..p95)
      d_norm_hi: 0.0904
      s_norm_lo: 0.0040
      s_norm_hi: 0.5890
      tiers:                   # ordered low -> high change; first min_score = 0.0
        - {name: low,  min_score: 0.0,  fps: 1.0, resolution: {mode: preset, preset: 720p}, quality: 90}
        - {name: mid,  min_score: 0.15, fps: 2.0, resolution: {mode: preset, preset: 480p}, quality: 75}
        - {name: high, min_score: 0.40, fps: 4.0, resolution: {mode: preset, preset: 360p}, quality: 60}
```

The decision is causal: a window's tier comes only from change observed up to that
window, mirroring what a device could compute online. The calibration constants and
thresholds above come from `tests/measure_change_metrics.py` (re-run it on your own
data to recalibrate; `tests/plot_change_metrics.py` visualizes the distributions).

## CLI

```bash
rtv-eval run <config.yaml>              # Run experiment
rtv-eval run <config.yaml> --dry-run    # Preview without running
rtv-eval report <results.db>            # Print comparison table
rtv-eval report <results.db> -f json    # Export as JSON
rtv-eval report <results.db> -f csv -o out.csv  # Export as CSV
rtv-eval export <results.db> --view summary      # Export run-level CSV for plotting
rtv-eval export <results.db> --view results      # Export per-question CSV for plotting
rtv-eval status <results.db>            # Show run status
```

Use `-v` for debug logging: `rtv-eval -v run ...`

## Results

Results are stored in SQLite (`results/<experiment>.db`). Each run tracks:

- Per-question: raw response, extracted answer, correctness, latency, frame count
- Summary: overall accuracy + per question-type breakdown (6 types in MotionBench)

Interrupted runs resume automatically — completed entries are skipped on re-run.

For visualization, use `rtv-eval export`:

- `--view summary`: one row per run with accuracy, counts, average latency, average frame count
- `--view results`: one row per question with correctness, latency, frame count, and raw model output

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
