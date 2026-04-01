# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## TOP PRIORITY — Subagent & Parallelism Strategy

**核心原则：每一步都尽可能使用最多数量的 subagents 并行工作，数量不设上限（as many as needed）。宁可多开不要少开。**

Before starting **any** task:
1. **More than 3 lines of code to change?** → Delegate to subagents.
2. **Web research needed?** → Launch research subagent in parallel.
3. **Independent subtasks visible?** → Run as parallel subagents in a single message.

### Maximize Parallelism
- 拆分粒度尽可能小：每个独立文件、每个独立组件 = 独立 subagent
- 研究与实现并行；测试与文档并行
- **永远不要因为 subagent 太多而合并任务**

### Context Passing Rules
Every subagent prompt **must** include: file paths + line numbers, current state, exact scope, expected output format.

---

## Project Goal
Build a reliable data pipeline that transforms soccer tracking streams into VBVR-aligned BEV clip artifacts for large-scale dataset generation.

## Commands

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run all tests:**
```bash
python -m pytest tests/
```

**Run a single test:**
```bash
python -m pytest tests/test_adapters.py::AdapterTests::test_metrica_adapter_produces_canonical_columns_and_ball
```

**Single-clip smoke test (Metrica):**
```bash
python soccer_bev_pipeline.py \
  --dataset metrica \
  --home_csv sample_data/metrica_official/data/Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv \
  --away_csv sample_data/metrica_official/data/Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv \
  --start_frame 1 --disable_realism_filter --fps 16 --seconds 5
```

**Batched deterministic generation:**
```bash
python soccer_bev_pipeline.py --dataset <name> --num_clips N --seed S --fps 16 --seconds 5
```

**EC2-to-S3 batch runner:**
```bash
scripts/run_pipeline_to_s3.sh \
  --dataset metrica \
  --home-csv <path> --away-csv <path> \
  --s3-bucket <bucket> --s3-prefix <prefix> \
  -- --num_clips 100 --fps 25 --seconds 10 --seed 20260306
```

## Canonical Output Contract (VBVR-aligned)
For each clip folder, export exactly these files (matching [VBVR-DataFactory](https://github.com/Video-Reason/VBVR-DataFactory) convention):
- `ground_truth.mp4`, `first_frame.png`, `final_frame.png`, `prompt.txt`

Instance folder naming: `{prefix}_{8-digit-index}`, e.g. `soccer_bev_00000000`.

## Architecture

Single-file pipeline (`soccer_bev_pipeline.py`). Key data flow:

```
Raw data → Adapter (canonical long_df) → Parser (densify/interpolate/normalize/clip) → Sampler → Realism filter → BEVRenderer → Exporter
```

**Adapter layer** (`BaseTrackingAdapter`, `MetricaAdapter`, `SkillCornerAdapter`, `SkillCornerV2Adapter`)
- Convert raw source formats into one canonical long-form tracking table
- Required columns: `frame_id`, `agent_id`, `agent_type`, `x`, `y`
- Optional: `period`, `timestamp`, `team_id`, `player_id`, `possession_team`, `source_dataset`, `source_match_id`

**Parser layer** (`DataParser`, `AdapterBackedParser`, source-specific subclasses)
- Shared: densification, interpolation, coordinate normalization, clip extraction, possession reindexing
- Source-specific parsers are thin wrappers over adapters

**Coordinate normalization** (`_detect_axis_mode`, `normalize_coordinates_inplace`)
- Percentile-based range detection (p0.5/p99.5) — robust against outliers
- Supports unit [0,1], centered [-dim/2, dim/2], and metric [0, dim] modes
- After normalization, `x`/`y` are always in metric meters [0, pitch_length_m] × [0, pitch_width_m]

**Sampling** (`sample_clip_specs`)
- Deterministic: global seed + logical clip index → derived child seed
- Shard-safe with `--clip_index_offset`

**Dataset registry**
- Primary CLI selector: `--dataset` (`metrica`, `skillcorner_v1`, `skillcorner_v2`)
- Legacy `--mode` is compatibility-only

**Tests** (`tests/test_adapters.py`)
- Covers canonical adapter contract, deterministic ordering, interpolation, strict schema failures

## Key CLI Flags

| Flag | Default | Notes |
|------|---------|-------|
| `--dataset` | required | `metrica`, `skillcorner_v1`, `skillcorner_v2` |
| `--num_clips` | 1 | Clips to generate |
| `--seed` | 42 | Global RNG seed |
| `--clip_index_offset` | 0 | For sharded runs |
| `--start_frame` | None | Pin start frame (skips sampler) |
| `--disable_realism_filter` | off | Bypass all realism gates |
| `--allow_duplicate_starts` | off | Allow repeated clip windows |
| `--fps` | 16 | Output video FPS (Wan2.1 native FPS) |
| `--seconds` | 5 | Clip duration (81 frames = N×4+1 constraint for Wan2.1) |

## Engineering Priorities
1. Determinism at scale (seeded sampling and shard-safe indexing)
2. Strict parser validation with explicit error messages
3. Stable output schema for downstream training pipelines

## Known Pitfalls
- Real tracking data can have coordinates slightly outside [0,1] or [0, dim] — percentile-based axis detection handles this, but extreme outliers (>p99.5) are ignored for mode classification.
- Metrica sample-data CSVs use a 3-row header format; the multiline header parser must be tried before the generic fallback parsers.
- Some Metrica frames have NaN for substituted players — the pipeline interpolates and forward/back-fills these gaps.
- SkillCorner support is intentionally strict and versioned. Unsupported schemas must fail fast with explicit errors.

## Guardrails
- Do not silently duplicate clips unless user explicitly allows it (`--allow_duplicate_starts`).
- Keep large raw datasets and generated outputs out of git history.
- Do not rewrite raw dataset files. Any cleanup/cache layer must live outside the raw source tree.
