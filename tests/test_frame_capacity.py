"""
Stress test: find the max frames the model can handle at various resolutions.
Reads video paths, model endpoint, and credentials from an experiment config YAML.
Default: configs/local.yaml
"""
import asyncio
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import openai
from PIL import Image

# Allow running from repo root even if rtv_eval is not installed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rtv_eval.config import load_config, ModelConfig

# --- Load from experiment config ---
CONFIG_PATH = Path(os.environ.get("RTV_EVAL_CONFIG", "configs/internvl3_only.yaml"))

_cfg = load_config(CONFIG_PATH)
_model: ModelConfig = _cfg.models[0]  # use the first model in the config

BASE_URL = _model.base_url
MODEL_NAME = _model.model_name
API_KEY = os.environ.get(_model.api_key_env, "dummy")
VIDEO_ROOT = Path(_cfg.benchmark.video_root)
META_PATH = Path(_cfg.benchmark.data_dir) / "video_info.meta.jsonl"

# Test grid: (frame_count, resolution_label, max_height)
TEST_GRID = [
    # Frame counts to test
    (24, "480p", 480),
]

QUESTION = "What is happening in this video? Answer in one sentence."


def find_test_video() -> tuple[Path, dict]:
    """Find a video of medium length (~10-15s) for testing."""
    with open(META_PATH) as f:
        for line in f:
            item = json.loads(line.strip())
            vi = item.get("video_info", {})
            dur = vi.get("duration", 0)
            if 10 < dur < 15:
                vp = item["video_path"]
                for sub in ["public-dataset", "self-collected"]:
                    p = VIDEO_ROOT / sub / vp
                    if p.exists():
                        return p, vi
    # Fallback: first available
    for sub in ["public-dataset", "self-collected"]:
        d = VIDEO_ROOT / sub
        if d.exists():
            for f in sorted(d.iterdir()):
                if f.suffix == ".mp4":
                    cap = cv2.VideoCapture(str(f))
                    vi = {
                        "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
                        "fps": cap.get(cv2.CAP_PROP_FPS),
                        "resolution": {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                       "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
                    }
                    cap.release()
                    return f, vi
    raise FileNotFoundError("No video found")


def extract_frames(video_path: Path, n_frames: int) -> list[np.ndarray]:
    """Extract n evenly-spaced frames from video."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(n_frames, total)
    indices = np.linspace(0, total - 1, n, dtype=int).tolist()
    idx_set = set(indices)
    frames = []
    frame_no = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_no in idx_set:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_no += 1
        if len(frames) == n:
            break
    cap.release()
    return frames


def resize_frame(frame: np.ndarray, max_height: int) -> np.ndarray:
    """Resize preserving aspect ratio, constrain by height."""
    h, w = frame.shape[:2]
    if h <= max_height:
        return frame
    scale = max_height / h
    return cv2.resize(frame, (int(w * scale), max_height), interpolation=cv2.INTER_AREA)


def encode_jpeg(frame: np.ndarray, quality: int = 85) -> str:
    pil_img = Image.fromarray(frame)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


async def test_call(client: openai.AsyncOpenAI, frames_b64: list[str],
                     n_frames: int, res_label: str) -> dict:
    """Make one API call and report result."""
    content = []
    for b64 in frames_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
        })
    content.append({"type": "text", "text": QUESTION})

    # Estimate total payload size
    total_b64_bytes = sum(len(b) for b in frames_b64)
    approx_mb = total_b64_bytes * 3 / 4 / 1024 / 1024  # base64 -> bytes -> MB

    t0 = time.time()
    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": content}],
                max_tokens=128,
                temperature=0.0,
            ),
            timeout=120,
        )
        elapsed = time.time() - t0
        answer = resp.choices[0].message.content or ""
        usage = resp.usage
        return {
            "frames": n_frames,
            "resolution": res_label,
            "status": "OK",
            "elapsed_s": round(elapsed, 2),
            "response": answer[:100],
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "payload_mb": round(approx_mb, 2),
        }
    except asyncio.TimeoutError:
        elapsed = time.time() - t0
        return {
            "frames": n_frames,
            "resolution": res_label,
            "status": "TIMEOUT",
            "elapsed_s": round(elapsed, 2),
            "payload_mb": round(approx_mb, 2),
        }
    except Exception as e:
        elapsed = time.time() - t0
        err = str(e)[:200]
        return {
            "frames": n_frames,
            "resolution": res_label,
            "status": f"ERROR: {err}",
            "elapsed_s": round(elapsed, 2),
            "payload_mb": round(approx_mb, 2),
        }


async def main():
    print("=== Frame Capacity Stress Test ===\n")

    # Find test video
    video_path, video_info = find_test_video()
    print(f"Video: {video_path.name}")
    print(f"  Duration: {video_info.get('duration', '?'):.1f}s")
    print(f"  FPS: {video_info.get('fps', '?')}")
    res = video_info.get("resolution", {})
    print(f"  Resolution: {res.get('width', '?')}x{res.get('height', '?')}")
    print()

    client = openai.AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=120)

    # Check server connectivity
    try:
        models = await client.models.list()
        print(f"Server OK. Models: {[m.id for m in models.data][:3]}...\n")
    except Exception as e:
        print(f"ERROR: Cannot connect to server at {BASE_URL}")
        print(f"  {e}")
        return

    # Group tests by resolution for cleaner output
    from collections import defaultdict
    grouped = defaultdict(list)
    for n_frames, res_label, max_h in TEST_GRID:
        grouped[res_label].append((n_frames, max_h))

    results = []
    for res_label in ["360p", "480p", "720p", "1080p"]:
        if res_label not in grouped:
            continue
        print(f"--- {res_label} ---")
        for n_frames, max_h in grouped[res_label]:
            # Extract & resize frames
            raw_frames = extract_frames(video_path, n_frames)
            resized = [resize_frame(f, max_h) for f in raw_frames]
            frames_b64 = [encode_jpeg(f) for f in resized]

            h, w = resized[0].shape[:2]
            print(f"  {n_frames} frames @ {w}x{h} ... ", end="", flush=True)

            result = await test_call(client, frames_b64, n_frames, res_label)
            results.append(result)

            status = result["status"]
            elapsed = result["elapsed_s"]
            payload = result["payload_mb"]
            tokens = result.get("prompt_tokens", "?")
            print(f"{status} | {elapsed}s | ~{payload}MB | {tokens} tokens")

            if status != "OK":
                print(f"    -> Stopping {res_label} tests (model failed at {n_frames} frames)")
                break
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Resolution':<12} {'Max OK':<10} {'Last Fail':<12} {'Tokens@Max':<12}")
    print("-" * 46)
    for res_label in ["360p", "480p", "720p", "1080p"]:
        ok_results = [r for r in results if r["resolution"] == res_label and r["status"] == "OK"]
        fail_results = [r for r in results if r["resolution"] == res_label and r["status"] != "OK"]
        max_ok = max((r["frames"] for r in ok_results), default=0)
        last_fail = min((r["frames"] for r in fail_results), default=None)
        tokens_at_max = "?"
        for r in ok_results:
            if r["frames"] == max_ok:
                tokens_at_max = r.get("prompt_tokens", "?")
                break
        fail_str = str(last_fail) if last_fail else "N/A"
        print(f"{res_label:<12} {max_ok:<10} {fail_str:<12} {tokens_at_max:<12}")

    # Save full results
    out_path = Path(__file__).parent.parent / "results" / "frame_capacity_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
