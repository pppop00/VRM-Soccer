# VRM-Soccer

A production-oriented pipeline for converting soccer tracking data into VBVR-style BEV clips with full tactical pitch rendering.

The project ingests tracking data through dataset adapters, samples 10-second windows, applies coordinate normalization and attack-direction alignment, renders tactical BEV frames on a realistic soccer pitch, and exports per clip:

- `ground_truth.mp4`
- `first_frame.png`
- `final_frame.png`
- `prompt.txt`

Output naming follows the [VBVR-DataFactory](https://github.com/Video-Reason/VBVR-DataFactory) convention.

## Features

- **Canonical multi-source adapter layer**
  - `MetricaAdapter` for Metrica Sports tracking CSV pairs
  - `SkillCornerAdapter` for strict `skillcorner_tracking_v1` JSON
  - `SkillCornerV2Adapter` for official SkillCorner Open Data (`*_tracking_extrapolated.jsonl` + `*_match.json`)
  - Shared parser stage for densification, interpolation, normalization, and clip extraction
- **Canonical ingestion contract**
  - Required columns: `frame_id`, `agent_id`, `agent_type`, `x`, `y`
  - Optional columns: `period`, `timestamp`, `team_id`, `player_id`, `possession_team`, `source_dataset`, `source_match_id`
- **Robust coordinate mode detection** (percentile-based, outlier-tolerant)
  - unit `[0,1]`
  - centered `[-L/2, L/2]`, `[-W/2, W/2]`
  - metric `[0, L]`, `[0, W]`
- **Tactical BEV renderer**
  - Green pitch with alternating grass stripes
  - Full FIFA-standard markings: boundary, halfway line, center circle, penalty areas, goal areas, penalty spots, penalty arcs, corner arcs, goals with net fill
  - Distinct team colors: red (home) vs blue (away), yellow ball
  - Black outlines on all agents for visibility
  - 800x520 default resolution (correct pitch aspect ratio)
- **Orientation normalization**
  - Possession-aware direction inference
  - Centroid-drift fallback
- **Deterministic large-scale sampling**
  - Global seed + logical clip index → derived child seeds
  - Shard-friendly with `--clip_index_offset`
- **Tactical realism filtering** (configurable thresholds)
  - Ball in-bounds ratio
  - Attack progress
  - Support distance / ratio
  - Defense shape
  - Majority possession coherence
- **VBVR-aligned output contract** per clip directory (4 files: `ground_truth.mp4`, `first_frame.png`, `final_frame.png`, `prompt.txt`)

## Install

```bash
pip install -r requirements.txt
```

Requires Python 3.10+, `opencv-python`, `numpy`, `pandas`.

## Data

This repo does **not** bundle large datasets.

Recommended sources:
- Metrica sample data: https://github.com/metrica-sports/sample-data
- SkillCorner open data: https://github.com/SkillCorner/opendata

Raw datasets are treated as immutable inputs. Dataset-specific cleanup should happen only in canonicalized cache artifacts, not by rewriting source files.

Clone Metrica sample data into `sample_data/`:

```bash
git clone https://github.com/metrica-sports/sample-data.git sample_data/metrica_official
```

## Quick Start

### 1) Single clip (Metrica)

```bash
python soccer_bev_pipeline.py --dataset metrica \
  --home_csv sample_data/metrica_official/data/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv \
  --away_csv sample_data/metrica_official/data/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv \
  --output_root output \
  --clip_id soccer_bev_00000000 \
  --fps 25 \
  --seconds 10 \
  --seed 42
```

### 2) Batch generation (deterministic)

```bash
python soccer_bev_pipeline.py --dataset metrica \
  --home_csv /path/to/home.csv \
  --away_csv /path/to/away.csv \
  --output_root output \
  --num_clips 1000 \
  --clip_id_prefix soccer_bev \
  --clip_index_offset 0 \
  --fps 25 \
  --seconds 10 \
  --seed 2026
```

### 3) SkillCorner Open Data (`skillcorner_v2`)

```bash
python soccer_bev_pipeline.py --dataset skillcorner_v2 \
  --tracking_json sample_data/skillcorner_opendata/data/matches/1886347/1886347_tracking_extrapolated.jsonl \
  --match_json sample_data/skillcorner_opendata/data/matches/1886347/1886347_match.json \
  --output_root output \
  --clip_id soccer_bev_00000000 \
  --fps 10 \
  --seconds 10 \
  --seed 42
```

Legacy `--mode` is still accepted as a compatibility alias, but new usage should prefer `--dataset`.

## CLI Arguments

### Input
| Argument | Description |
|---|---|
| `--dataset` | `metrica`, `skillcorner_v1`, or `skillcorner_v2` |
| `--home_csv` | Metrica home team CSV path |
| `--away_csv` | Metrica away team CSV path |
| `--tracking_json` | SkillCorner tracking path: JSON for v1, JSONL for official v2 |
| `--match_json` | Optional SkillCorner companion match metadata JSON path for v2 |

### Sampling
| Argument | Default | Description |
|---|---|---|
| `--num_clips` | `1` | Number of clips to sample |
| `--seed` | `42` | Global random seed |
| `--clip_index_offset` | `0` | Starting logical clip index (for sharding) |
| `--start_frame` | — | Fixed start frame (only when `num_clips=1`) |
| `--max_sampling_attempts` | `100` | Max retries per clip for realism filter |
| `--allow_duplicate_starts` | off | Allow same start frame across clips |

### Clip & Video
| Argument | Default | Description |
|---|---|---|
| `--fps` | `25` | Output video frame rate |
| `--seconds` | `10` | Clip duration |
| `--width` | `800` | Frame width in pixels |
| `--height` | `520` | Frame height in pixels |
| `--pitch_length_m` | `105.0` | Pitch length in meters |
| `--pitch_width_m` | `68.0` | Pitch width in meters |

### Rendering
| Argument | Default | Description |
|---|---|---|
| `--home_color` | `#E63946` | Home team color (hex) |
| `--away_color` | `#457B9D` | Away team color (hex) |
| `--ball_color` | `#F5F500` | Ball color (hex) |
| `--disable_orientation_normalization` | off | Skip attack-direction alignment |

### Realism Filter
| Argument | Default | Description |
|---|---|---|
| `--disable_realism_filter` | off | Skip all realism checks |
| `--min_ball_in_bounds_ratio` | `0.98` | Min fraction of frames with ball in bounds |
| `--min_attack_progress_m` | `3.0` | Min forward progress by attacking team (meters) |
| `--support_distance_m` | `20.0` | Max distance for attacker "support" check |
| `--min_support_ratio` | `0.55` | Min ratio of frames with nearby support |
| `--min_defense_ahead_ratio` | `0.0` | Min ratio of frames with defenders ahead of ball |
| `--min_majority_possession_ratio` | `0.50` | Min ratio of frames where one team holds possession |

## Output Layout

Follows [VBVR-DataFactory](https://github.com/Video-Reason/VBVR-DataFactory) naming convention:

```text
output/soccer_bev_00000000/
├── ground_truth.mp4   # 10s BEV video (25fps)
├── first_frame.png    # First frame snapshot
├── final_frame.png    # Last frame snapshot
└── prompt.txt         # Tactical description

output/soccer_bev_00000001/
├── ground_truth.mp4
├── first_frame.png
├── final_frame.png
└── prompt.txt
...
```

Instance folders use 8-digit zero-padded indices (`_00000000`, `_00000001`, ...).

## Architecture

```text
soccer_bev_pipeline.py
├── Canonical Adapters
│   ├── BaseTrackingAdapter       — dataset-specific raw -> canonical long-form mapping
│   ├── MetricaAdapter            — Metrica CSV pairs (incl. multiline headers)
│   ├── SkillCornerAdapter        — strict skillcorner_tracking_v1 JSON
│   └── SkillCornerV2Adapter      — official SkillCorner Open Data JSONL + match metadata
├── Shared Parser
│   ├── DataParser (ABC)          — shared pivot + interpolation contract
│   ├── AdapterBackedParser       — shared canonical parser implementation
│   ├── MetricaParser             — thin wrapper over MetricaAdapter
│   ├── SkillCornerParser         — thin wrapper over SkillCornerAdapter
│   └── SkillCornerV2Parser       — thin wrapper over SkillCornerV2Adapter
├── Coordinate Normalization
│   ├── _detect_axis_mode()       — percentile-based unit/centered/metric detection
│   └── normalize_coordinates_inplace()
├── Orientation
│   ├── normalize_attack_direction()
│   └── infer_possession_by_proximity()
├── Realism Filter
│   ├── RealismConfig             — configurable thresholds
│   └── evaluate_clip_realism()   — ball bounds, progress, support, defense shape
├── Sampling
│   └── sample_clip_specs()       — deterministic seed-derived clip selection
├── BEVRenderer
│   ├── _draw_pitch()             — full FIFA-standard pitch markings
│   └── render_frames()           — tactical dots on pitch background
├── Prompt Generator
│   ├── analyze_clip_events()     — possession, passes, ball movement from tracking
│   └── generate_clip_prompt()    — English tactical description
├── Dataset Registry
│   ├── _resolve_dataset_name()   — dataset / legacy mode resolution
│   └── _create_dataset_parser()  — adapter-backed parser factory
├── Tests
│   └── tests/test_adapters.py    — adapter contract and canonicalization regression tests
└── export_vbvr_clip()            — ground_truth.mp4 + first/final frame PNGs + prompt.txt
```

## Testing

```bash
python3 -m unittest discover -s tests -v
```

## AWS / 1M Scale Notes

- Use many workers, each with the same `--seed` and different `--clip_index_offset`.
- Keep clip assignment deterministic by logical clip index range.
- Keep `--allow_duplicate_starts` disabled unless you explicitly accept repeated windows.
- Tune realism thresholds to balance quality vs throughput.

## AWS / EC2 -> S3

Use [scripts/run_pipeline_to_s3.sh](/Users/pppop/Desktop/Projects/Video-Reason-Soccer/scripts/run_pipeline_to_s3.sh) for EC2 jobs. It runs the pipeline against local files on the instance, syncs the generated clip folders to S3, and deletes the local batch by default.

Metrica example:

```bash
scripts/run_pipeline_to_s3.sh \
  --dataset metrica \
  --home-csv sample_data/metrica_official/data/Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv \
  --away-csv sample_data/metrica_official/data/Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv \
  --s3-bucket videosoccer \
  --s3-prefix vrm-soccer/metrica/run_0001 \
  -- --num_clips 100 --fps 25 --seconds 10 --seed 20260306
```

SkillCorner example:

```bash
scripts/run_pipeline_to_s3.sh \
  --dataset skillcorner_v2 \
  --tracking-json sample_data/skillcorner_opendata/data/matches/1886347/1886347_tracking_extrapolated.jsonl \
  --match-json sample_data/skillcorner_opendata/data/matches/1886347/1886347_match.json \
  --s3-bucket videosoccer \
  --s3-prefix vrm-soccer/skillcorner/run_0001 \
  -- --num_clips 3 --fps 10 --seconds 10 --seed 20260306
```

Notes:
- The EC2 instance must already have an IAM role with S3 write access.
- S3 is the durable output store. EC2 local disk is only a temporary staging area.
- Use `--keep-local` if you want to inspect generated files on the instance before cleanup.

## License

Apache-2.0. See `LICENSE`.
