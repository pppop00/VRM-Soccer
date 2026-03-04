# CLAUDE.md

## Project Goal
Build a reliable data pipeline that transforms soccer tracking streams into VBVR-aligned BEV clip artifacts for large-scale dataset generation.

## Canonical Output Contract
For each clip folder, export exactly three visual files:
- `video.mp4`
- `first_frame.png`
- `last_frame.png`

No extra per-frame images should be emitted by default.

## Engineering Priorities
1. Determinism at scale (seeded sampling and shard-safe indexing)
2. Strict parser validation with explicit error messages
3. Modularity (parser, sampler, realism filter, renderer, exporter)
4. Tactical plausibility gates configurable from CLI
5. Stable output schema for downstream training pipelines

## Main Entry
- `soccer_bev_pipeline.py`

## Typical Workflow
1. Validate parser compatibility with source data format.
2. Run single-clip smoke test.
3. Run batched deterministic generation.
4. Verify output directories contain exactly required artifacts.

## Guardrails
- Keep defaults conservative and reproducible.
- Do not silently duplicate clips unless user explicitly allows it.
- Keep large raw datasets and generated outputs out of git history.

## Dependency Baseline
- Python 3.10+
- `opencv-python`
- `numpy`
- `pandas`
