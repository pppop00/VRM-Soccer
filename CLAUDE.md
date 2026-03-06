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

1. **Parsers** (`MetricaParser`, `SkillCornerParser`)
   - Metrica: handles both simple CSV headers and 3-row multiline headers (team/jersey/field rows)
   - SkillCorner: handles list-of-frames or dict-with-frames-key JSON structures
2. **Coordinate normalization** (`_detect_axis_mode`, `normalize_coordinates_inplace`)
   - Uses percentile-based range detection (p0.5/p99.5) to be robust against off-pitch outliers
   - Supports unit [0,1], centered [-dim/2, dim/2], and metric [0, dim] modes
3. **Orientation normalization** (`normalize_attack_direction`)
   - Possession-label-aware direction inference, centroid-drift fallback
4. **Realism filter** (`evaluate_clip_realism`, `RealismConfig`)
   - Ball in-bounds, attack progress, support ratio, defense shape, possession coherence
5. **Sampling** (`sample_clip_specs`)
   - Deterministic: global seed + logical clip index → derived child seed
   - Shard-safe with `--clip_index_offset`
6. **BEV Renderer** (`BEVRenderer`)
   - Full pitch with FIFA-standard markings (penalty areas, goal areas, center circle, arcs, goals)
   - Green striped grass background
   - Team colors: red home (#E63946), blue away (#457B9D), yellow ball (#F5F500)
   - Black outlines on agents for contrast
   - Cached pitch background for efficient frame rendering
7. **Prompt Generator** (`analyze_clip_events`, `generate_clip_prompt`)
   - Extracts possession, passes, ball movement from tracking data
   - Generates English tactical description as `prompt.txt`
8. **Exporter** (`export_vbvr_clip`)
   - Tries avc1 codec first, falls back to mp4v
   - Cleans stale visual files before writing
   - File names follow VBVR convention: `ground_truth.mp4`, `first_frame.png`, `final_frame.png`

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
2. Run single-clip smoke test with `--start_frame 1 --disable_realism_filter`.
3. Run batched deterministic generation with `--num_clips N --seed S`.
4. Verify output directories contain exactly the 4 required artifacts (`ground_truth.mp4`, `first_frame.png`, `final_frame.png`, `prompt.txt`).

## Known Pitfalls
- Real tracking data can have coordinates slightly outside [0,1] or [0, dim] — the percentile-based axis detection handles this, but extreme outliers (>p99.5) are ignored for mode classification.
- Metrica sample-data CSVs use a 3-row header format; the multiline header parser must be tried before the generic fallback parsers.
- Some Metrica frames have NaN for substituted players — the pipeline interpolates and forward/back-fills these gaps.

## Guardrails
- Keep defaults conservative and reproducible.
- Do not silently duplicate clips unless user explicitly allows it (`--allow_duplicate_starts`).
- Keep large raw datasets and generated outputs out of git history.

## Dependency Baseline
- Python 3.10+
- `opencv-python`
- `numpy`
- `pandas`
