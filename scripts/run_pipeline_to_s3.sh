#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_pipeline_to_s3.sh \
    --dataset <metrica|skillcorner_v1|skillcorner_v2> \
    --s3-bucket <bucket> \
    --s3-prefix <prefix> \
    [dataset-specific args] \
    [aws-wrapper args] \
    -- [extra soccer_bev_pipeline.py args]

Dataset-specific args:
  Metrica:
    --home-csv <path>
    --away-csv <path>

  SkillCorner:
    --tracking-json <path>
    --match-json <path>   (optional for skillcorner_v2)

AWS wrapper args:
  --region <aws-region>          Default: us-east-2
  --repo-root <path>             Default: parent of this script
  --work-root <path>             Default: /tmp/vrm-soccer
  --run-name <name>              Default: <dataset>_<UTC timestamp>
  --python-bin <python>          Default: python3
  --keep-local                   Do not delete local output after upload

Everything after '--' is passed directly to soccer_bev_pipeline.py.

Examples:
  scripts/run_pipeline_to_s3.sh \
    --dataset metrica \
    --home-csv sample_data/metrica_official/data/Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv \
    --away-csv sample_data/metrica_official/data/Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv \
    --s3-bucket videosoccer \
    --s3-prefix vrm-soccer/metrica/run_0001 \
    -- --num_clips 100 --fps 25 --seconds 10 --seed 20260306

  scripts/run_pipeline_to_s3.sh \
    --dataset skillcorner_v2 \
    --tracking-json sample_data/skillcorner_opendata/data/matches/1886347/1886347_tracking_extrapolated.jsonl \
    --match-json sample_data/skillcorner_opendata/data/matches/1886347/1886347_match.json \
    --s3-bucket videosoccer \
    --s3-prefix vrm-soccer/skillcorner/run_0001 \
    -- --num_clips 3 --fps 10 --seconds 10 --seed 20260306
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

dataset=""
s3_bucket=""
s3_prefix=""
home_csv=""
away_csv=""
tracking_json=""
match_json=""
region="us-east-2"
repo_root="${DEFAULT_REPO_ROOT}"
work_root="/tmp/vrm-soccer"
run_name=""
python_bin="python3"
keep_local="false"

pipeline_extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      dataset="${2:-}"
      shift 2
      ;;
    --s3-bucket)
      s3_bucket="${2:-}"
      shift 2
      ;;
    --s3-prefix)
      s3_prefix="${2:-}"
      shift 2
      ;;
    --home-csv)
      home_csv="${2:-}"
      shift 2
      ;;
    --away-csv)
      away_csv="${2:-}"
      shift 2
      ;;
    --tracking-json)
      tracking_json="${2:-}"
      shift 2
      ;;
    --match-json)
      match_json="${2:-}"
      shift 2
      ;;
    --region)
      region="${2:-}"
      shift 2
      ;;
    --repo-root)
      repo_root="${2:-}"
      shift 2
      ;;
    --work-root)
      work_root="${2:-}"
      shift 2
      ;;
    --run-name)
      run_name="${2:-}"
      shift 2
      ;;
    --python-bin)
      python_bin="${2:-}"
      shift 2
      ;;
    --keep-local)
      keep_local="true"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      pipeline_extra_args=("$@")
      break
      ;;
    *)
      echo "Unknown wrapper argument: $1" >&2
      echo "Use '--' before soccer_bev_pipeline.py arguments." >&2
      exit 1
      ;;
  esac
done

if [[ -z "${dataset}" || -z "${s3_bucket}" || -z "${s3_prefix}" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 1
fi

case "${dataset}" in
  metrica)
    if [[ -z "${home_csv}" || -z "${away_csv}" ]]; then
      echo "--dataset metrica requires --home-csv and --away-csv." >&2
      exit 1
    fi
    ;;
  skillcorner_v1|skillcorner_v2)
    if [[ -z "${tracking_json}" ]]; then
      echo "--dataset ${dataset} requires --tracking-json." >&2
      exit 1
    fi
    ;;
  *)
    echo "Unsupported dataset for wrapper: ${dataset}" >&2
    exit 1
    ;;
esac

require_cmd "${python_bin}"
require_cmd aws

if [[ ! -f "${repo_root}/soccer_bev_pipeline.py" ]]; then
  echo "soccer_bev_pipeline.py not found under repo root: ${repo_root}" >&2
  exit 1
fi

if [[ -z "${run_name}" ]]; then
  run_name="${dataset}_$(date -u +%Y%m%dT%H%M%SZ)"
fi

local_output_root="${work_root%/}/${run_name}"
mkdir -p "${local_output_root}"

pipeline_cmd=(
  "${python_bin}"
  "${repo_root}/soccer_bev_pipeline.py"
  "--dataset" "${dataset}"
  "--output_root" "${local_output_root}"
)

if [[ -n "${home_csv}" ]]; then
  pipeline_cmd+=("--home_csv" "${home_csv}")
fi
if [[ -n "${away_csv}" ]]; then
  pipeline_cmd+=("--away_csv" "${away_csv}")
fi
if [[ -n "${tracking_json}" ]]; then
  pipeline_cmd+=("--tracking_json" "${tracking_json}")
fi
if [[ -n "${match_json}" ]]; then
  pipeline_cmd+=("--match_json" "${match_json}")
fi
if [[ ${#pipeline_extra_args[@]} -gt 0 ]]; then
  pipeline_cmd+=("${pipeline_extra_args[@]}")
fi

echo "Running pipeline locally on EC2:"
printf '  %q' "${pipeline_cmd[@]}"
printf '\n'

"${pipeline_cmd[@]}"

s3_uri="s3://${s3_bucket}/${s3_prefix%/}/"

echo "Syncing outputs to ${s3_uri}"
aws s3 sync "${local_output_root}/" "${s3_uri}" --region "${region}" --only-show-errors

echo "Upload complete: ${s3_uri}"

if [[ "${keep_local}" != "true" ]]; then
  rm -rf "${local_output_root}"
  echo "Removed local output root: ${local_output_root}"
else
  echo "Kept local output root: ${local_output_root}"
fi
