from __future__ import annotations

# Model loading and multi-image inference adapted from local-model-serving.
import gc
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path


def resolve_dtype(name: str):
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[name]


def move_tensors(batch, device):
    import torch

    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def maybe_warn_about_fp8_model(model_id: str, device: str) -> None:
    import torch

    if "FP8" not in model_id.upper():
        return
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return

    major, minor = torch.cuda.get_device_capability(0)
    if (major, minor) < (8, 9):
        print(
            "Warning: this GPU does not support native FP8 execution for the official "
            "Qwen3-VL FP8 checkpoint. Transformers may dequantize it back to BF16.",
            file=sys.stderr,
        )


def resolve_model_source(model: str, cache_dir: str) -> str:
    model_path = Path(model).expanduser()
    if model_path.exists():
        return str(model_path.resolve())

    cache_root = Path(cache_dir).expanduser()
    repo_cache_dir = cache_root / f"models--{model.replace('/', '--')}"
    if not repo_cache_dir.is_dir():
        return model

    ref_path = repo_cache_dir / "refs" / "main"
    if ref_path.is_file():
        snapshot_name = ref_path.read_text(encoding="utf-8").strip()
        snapshot_dir = repo_cache_dir / "snapshots" / snapshot_name
        if snapshot_dir.is_dir():
            return str(snapshot_dir.resolve())

    snapshots = repo_cache_dir / "snapshots"
    if not snapshots.is_dir():
        return model
    snapshot_dirs = sorted(path for path in snapshots.iterdir() if path.is_dir())
    if len(snapshot_dirs) == 1:
        return str(snapshot_dirs[0].resolve())

    return model


@dataclass
class ServiceConfig:
    model: str = "Qwen/Qwen3-VL-8B-Instruct"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    cache_dir: str = str(Path.cwd())


class Qwen3VLService:
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.model = None
        self.processor = None
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._last_used_at = time.monotonic()
        self._active_requests = 0
        self._stop_event = threading.Event()

    def _touch_last_used(self) -> None:
        with self._state_lock:
            self._last_used_at = time.monotonic()

    def _begin_request(self) -> None:
        with self._state_lock:
            self._active_requests += 1
            self._last_used_at = time.monotonic()

    def _end_request(self) -> None:
        with self._state_lock:
            if self._active_requests > 0:
                self._active_requests -= 1
            self._last_used_at = time.monotonic()

    def unload(self) -> None:
        with self._load_lock:
            with self._generate_lock:
                with self._state_lock:
                    if self._active_requests != 0:
                        return
                    model = self.model
                    processor = self.processor
                    self.model = None
                    self.processor = None
                    self._last_used_at = time.monotonic()

                if model is None and processor is None:
                    return

                del model
                del processor
                gc.collect()

                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        if hasattr(torch.cuda, "ipc_collect"):
                            torch.cuda.ipc_collect()
                except Exception:
                    pass

    def close(self) -> None:
        self._stop_event.set()
        self.unload()

    def load(self) -> None:
        import torch
        from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

        if self.model is not None and self.processor is not None:
            return

        with self._load_lock:
            if self.model is not None and self.processor is not None:
                return

            if self.config.device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA device requested but torch.cuda.is_available() is False."
                )

            maybe_warn_about_fp8_model(self.config.model, self.config.device)
            model_source = resolve_model_source(self.config.model, self.config.cache_dir)
            model_config = AutoConfig.from_pretrained(model_source, cache_dir=self.config.cache_dir)
            if model_config.model_type != "qwen3_vl":
                raise ValueError(
                    "The integrated local loader supports Qwen3-VL only; "
                    f"received model type {model_config.model_type!r}."
                )
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_source,
                torch_dtype=resolve_dtype(self.config.dtype),
                attn_implementation=self.config.attn_implementation,
                device_map="auto" if self.config.device.startswith("cuda") else None,
                cache_dir=self.config.cache_dir,
            )
            if not self.config.device.startswith("cuda"):
                model = model.to(self.config.device)

            processor = AutoProcessor.from_pretrained(
                model_source,
                cache_dir=self.config.cache_dir,
            )

            self.model = model
            self.processor = processor
            self._touch_last_used()

    def infer_messages(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        images=None,
        videos=None,
        temperature: float = 0.0,
    ) -> str:
        return self._generate_messages(
            messages=messages,
            max_new_tokens=max_new_tokens,
            images=images,
            videos=videos,
            temperature=temperature,
        )

    def _prepare_message_inputs(
        self,
        messages: list[dict],
        images=None,
        videos=None,
    ):
        self.load()
        assert self.model is not None
        assert self.processor is not None

        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        processor_kwargs = {
            "text": [prompt_text],
            "padding": True,
            "return_tensors": "pt",
        }
        if images:
            processor_kwargs["images"] = images
        if videos:
            processor_kwargs["videos"] = videos

        inputs = self.processor(**processor_kwargs)
        inputs.pop("token_type_ids", None)
        return move_tensors(inputs, self.model.device)

    def _generate_messages(
        self,
        messages: list[dict],
        max_new_tokens: int,
        images=None,
        videos=None,
        temperature: float = 0.0,
    ) -> str:
        import torch

        self._begin_request()
        with self._generate_lock:
            try:
                inputs = self._prepare_message_inputs(
                    messages=messages,
                    images=images,
                    videos=videos,
                )

                with torch.inference_mode():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=temperature > 0,
                        **({"temperature": temperature} if temperature > 0 else {}),
                    )

                generated_ids = [
                    output_ids[len(input_ids):]
                    for input_ids, output_ids in zip(inputs["input_ids"], output_ids)
                ]
                decoded = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            finally:
                self._end_request()

        text = decoded[0].strip() if decoded else ""
        if not text:
            raise RuntimeError("Model returned an empty response.")
        return text
