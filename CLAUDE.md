# CLAUDE.md

## Project Goal
Build a reliable data pipeline that transforms soccer tracking streams into VBVR-aligned BEV clip artifacts for large-scale dataset generation.

## Canonical Output Contract (VBVR-aligned)
For each clip folder, export exactly these files (matching [VBVR-DataFactory](https://github.com/Video-Reason/VBVR-DataFactory) convention):
- `ground_truth.mp4`
- `first_frame.png`
- `final_frame.png`
- `prompt.txt`

Instance folder naming: `{prefix}_{8-digit-index}`, e.g. `soccer_bev_00000000`.

## Architecture

Single-file pipeline (`soccer_bev_pipeline.py`) with these components:

1. **Canonical adapter layer** (`BaseTrackingAdapter`, `MetricaAdapter`, `SkillCornerAdapter`, `SkillCornerV2Adapter`)
   - Adapters convert raw source formats into one canonical long-form tracking table
   - Required canonical columns: `frame_id`, `agent_id`, `agent_type`, `x`, `y`
   - Optional canonical columns: `period`, `timestamp`, `team_id`, `player_id`, `possession_team`, `source_dataset`, `source_match_id`
2. **Shared parser layer** (`DataParser`, `AdapterBackedParser`, `MetricaParser`, `SkillCornerParser`, `SkillCornerV2Parser`)
   - Shared responsibilities: densification, interpolation, coordinate normalization, clip extraction, possession reindexing
   - Source-specific parser classes are thin wrappers over adapters
3. **Coordinate normalization** (`_detect_axis_mode`, `normalize_coordinates_inplace`)
   - Uses percentile-based range detection (p0.5/p99.5) to be robust against off-pitch outliers
   - Supports unit [0,1], centered [-dim/2, dim/2], and metric [0, dim] modes
4. **Orientation normalization** (`normalize_attack_direction`)
   - Possession-label-aware direction inference, centroid-drift fallback
5. **Realism filter** (`evaluate_clip_realism`, `RealismConfig`)
   - Ball in-bounds, attack progress, support ratio, defense shape, possession coherence
6. **Sampling** (`sample_clip_specs`)
   - Deterministic: global seed + logical clip index → derived child seed
   - Shard-safe with `--clip_index_offset`
7. **BEV Renderer** (`BEVRenderer`)
   - Full pitch with FIFA-standard markings (penalty areas, goal areas, center circle, arcs, goals)
   - Green striped grass background
   - Team colors: red home (#E63946), blue away (#457B9D), yellow ball (#F5F500)
   - Black outlines on agents for contrast
   - Cached pitch background for efficient frame rendering
8. **Prompt Generator** (`analyze_clip_events`, `generate_clip_prompt`)
   - Extracts possession, passes, ball movement from tracking data
   - Generates English tactical description as `prompt.txt`
9. **Exporter** (`export_vbvr_clip`)
   - Tries avc1 codec first, falls back to mp4v
   - Cleans stale visual files before writing
   - File names follow VBVR convention: `ground_truth.mp4`, `first_frame.png`, `final_frame.png`
10. **Dataset Registry**
   - Primary CLI selector is `--dataset`
   - Supported datasets: `metrica`, `skillcorner_v1`, `skillcorner_v2`
   - Legacy `--mode` is compatibility-only
11. **Regression tests**
   - `tests/test_adapters.py` covers canonical adapter contract, deterministic ordering, interpolation, and strict schema failures

## Engineering Priorities
1. Determinism at scale (seeded sampling and shard-safe indexing)
2. Strict parser validation with explicit error messages
3. Modularity (parser, sampler, realism filter, renderer, exporter)
4. Tactical plausibility gates configurable from CLI
5. Stable output schema for downstream training pipelines

## Main Entry
- `soccer_bev_pipeline.py`

## Typical Workflow
1. Clone Metrica sample data into `sample_data/metrica_official/`.
2. Run single-clip smoke test with `--dataset metrica --start_frame 1 --disable_realism_filter`.
3. Run batched deterministic generation with `--dataset <name> --num_clips N --seed S`.
4. Verify output directories contain exactly the 4 required artifacts (`ground_truth.mp4`, `first_frame.png`, `final_frame.png`, `prompt.txt`).

## Known Pitfalls
- Real tracking data can have coordinates slightly outside [0,1] or [0, dim] — the percentile-based axis detection handles this, but extreme outliers (>p99.5) are ignored for mode classification.
- Metrica sample-data CSVs use a 3-row header format; the multiline header parser must be tried before the generic fallback parsers.
- Some Metrica frames have NaN for substituted players — the pipeline interpolates and forward/back-fills these gaps.
- SkillCorner support is intentionally strict and versioned. `skillcorner_tracking_v1` and the official Open Data JSONL schema (`skillcorner_tracking_v2`) are supported; unsupported schemas should fail fast with explicit errors.

## Guardrails
- Keep defaults conservative and reproducible.
- Do not silently duplicate clips unless user explicitly allows it (`--allow_duplicate_starts`).
- Keep large raw datasets and generated outputs out of git history.
- Do not rewrite raw dataset files. Any future cleanup/cache layer must live outside the raw source tree.

## Dependency Baseline
- Python 3.10+
- `opencv-python`
- `numpy`
- `pandas`
