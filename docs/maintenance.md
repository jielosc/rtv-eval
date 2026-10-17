# Reliability maintenance

## Fixed

- Missing videos, empty decoding results, and non-connection inference failures previously allowed a run to be marked completed. Completion now requires every expected question to have a persisted result. CLI failures exit nonzero; missing questions can be retried.
- Cancellation now cancels and awaits sibling tasks before closing the caller, and marks the run failed.
- Zero concurrency could wait forever. Invalid sampling rates/counts, missing resolution fields, zero dimensions, non-finite configuration values, invalid adaptive weights/thresholds/bounds, and duplicate model/strategy names now fail validation.
- Integer-truncated FPS intervals could over-sample (30 → 8 FPS previously produced 10 FPS). Fractional index spacing now respects target rate; unknown-frame-count fallback also applies the requested sampling policy.
- Extremely narrow/tall images could resize to a zero-pixel dimension. Resizing now keeps both dimensions at least one pixel.
- Automatic test collection imported a manual capacity probe that required a private config. It now lives in `scripts/measure_frame_capacity.py`.
- Documented `report -f` commands now work. Local example paths are generic. README explains window buffering and what latency/accuracy actually measure.

## Compatibility

Protocol `mcq-letter-v3` isolates the sampling changes from legacy results. Start a new experiment name after upgrading; old databases remain readable. Existing configurations with invalid values now fail early instead of producing runtime failures.

## Validation scope

Regression tests cover failure/resume behavior, CLI exit status, validation, fractional sampling, unknown frame counts, and extreme aspect ratios. The existing suite exercises scoring, export, adaptive sampling, model lifecycle, and the local API using synthetic/mocked inputs. Actual MotionBench performance and GPU inference require datasets, weights, credentials, and suitable hardware that are not supplied with this repository.

Remaining limitations: preprocessing runs synchronously and decodes videos per question; the unknown-frame-count fallback retains decoded frames in memory. Adaptive sampling requires a known frame count and buffers full windows. The dataset adapter reads the first QA entry per metadata row and indexes flat video directories by filename; use the supported MotionBench layout. These are not claims of a complete security or production-readiness audit.
