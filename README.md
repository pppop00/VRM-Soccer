# VRM-Soccer

A production-oriented pipeline for converting soccer tracking data into VBVR-style BEV clips.

The project ingests tracking data (Metrica CSV pair or SkillCorner JSON), samples 10-second windows, applies coordinate normalization and attack-direction alignment, renders tactical BEV frames, and exports exactly:

- `video.mp4`
- `first_frame.png`
- `last_frame.png`

## What This Version Supports

- Multi-source parsers:
  - Metrica Sports tracking CSVs (home + away)
  - SkillCorner tracking JSON
- Robust coordinate mode detection:
  - unit `[0,1]`
  - centered `[-L/2, L/2]`, `[-W/2, W/2]`
  - metric `[0, L]`, `[0, W]`
- Orientation normalization:
  - possession-aware direction inference
  - centroid-drift fallback
- Deterministic large-scale sampling:
  - global seed + logical clip index derived child seeds
  - shard-friendly with `--clip_index_offset`
- Tactical realism filtering (configurable thresholds)
- Strict output contract per clip directory (only 3 visual files)

## Install

```bash
pip install -r requirements.txt
```

## Data

This repo does **not** bundle large datasets.

Recommended sources:
- Metrica sample data: https://github.com/metrica-sports/sample-data
- SkillCorner open data: https://github.com/SkillCorner/opendata

## Quick Start

### 1) Single clip (Metrica)

```bash
python soccer_bev_pipeline.py --mode metrica \
  --home_csv /path/to/home.csv \
  --away_csv /path/to/away.csv \
  --output_root output \
  --clip_id clip_0001 \
  --fps 25 \
  --seconds 10 \
  --seed 42
```

### 2) Batch generation (deterministic)

```bash
python soccer_bev_pipeline.py --mode metrica \
  --home_csv /path/to/home.csv \
  --away_csv /path/to/away.csv \
  --output_root output \
  --num_clips 1000 \
  --clip_id_prefix clip \
  --clip_index_offset 0 \
  --fps 25 \
  --seconds 10 \
  --seed 2026
```

### 3) SkillCorner mode

```bash
python soccer_bev_pipeline.py --mode skillcorner \
  --tracking_json /path/to/tracking.json \
  --output_root output \
  --clip_id clip_0001 \
  --fps 25 \
  --seconds 10 \
  --seed 42
```

## Core CLI Arguments

- Input:
  - `--mode {metrica,skillcorner}`
  - `--home_csv`, `--away_csv` (Metrica)
  - `--tracking_json` (SkillCorner)
- Sampling:
  - `--num_clips`
  - `--seed`
  - `--clip_index_offset`
  - `--start_frame` (fixed start, only when `num_clips=1`)
  - `--max_sampling_attempts`
  - `--allow_duplicate_starts`
- Clip/video:
  - `--fps` (default `25`)
  - `--seconds` (default `10`)
  - `--width`, `--height`
- Rendering colors:
  - `--home_color`, `--away_color`, `--ball_color`
- Pitch:
  - `--pitch_length_m`, `--pitch_width_m`
- Realism filter:
  - `--disable_realism_filter`
  - `--min_ball_in_bounds_ratio`
  - `--min_attack_progress_m`
  - `--support_distance_m`
  - `--min_support_ratio`
  - `--min_defense_ahead_ratio`
  - `--min_majority_possession_ratio`

## Output Layout

For each clip (for example `output/clip_0000123/`):

```text
clip_0000123/
├── video.mp4
├── first_frame.png
└── last_frame.png
```

## AWS / 1M Scale Notes

- Use many workers, each with the same `--seed` and different `--clip_index_offset`.
- Keep clip assignment deterministic by logical clip index range.
- Keep `--allow_duplicate_starts` disabled unless you explicitly accept repeated windows.
- Tune realism thresholds to balance quality vs throughput.

## Main File

- `soccer_bev_pipeline.py`: end-to-end parser + sampler + renderer + exporter CLI.

## License

See `LICENSE`.
