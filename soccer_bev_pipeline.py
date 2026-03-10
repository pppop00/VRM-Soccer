#!/usr/bin/env python3
"""
Cross-domain soccer tracking -> VBVR-aligned BEV clip generator.

Supports:
- Metrica-style tracking CSVs (home + away)
- SkillCorner-style tracking JSON

Outputs per clip directory (VBVR-aligned):
- ground_truth.mp4
- first_frame.png
- final_frame.png
- prompt.txt
"""

from __future__ import annotations

import argparse
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import cv2
import numpy as np
import pandas as pd


class AgentType(str, Enum):
    HOME = "home"
    AWAY = "away"
    BALL = "ball"


CANONICAL_REQUIRED_COLUMNS = ("frame_id", "agent_id", "agent_type", "x", "y")
CANONICAL_OPTIONAL_COLUMNS = (
    "period",
    "timestamp",
    "team_id",
    "player_id",
    "possession_team",
    "source_dataset",
    "source_match_id",
)


@dataclass
class ParsedTrackingClip:
    coords_xy: np.ndarray
    agent_ids: list[str]
    agent_types: np.ndarray
    frame_ids: np.ndarray
    fps: int
    pitch_length_m: float
    pitch_width_m: float
    possession_team_by_frame: Optional[np.ndarray] = None


@dataclass
class RealismConfig:
    enabled: bool = True
    min_ball_in_bounds_ratio: float = 0.98
    min_attack_progress_m: float = 3.0
    support_distance_m: float = 20.0
    min_support_ratio: float = 0.55
    min_defense_ahead_ratio: float = 0.0
    min_majority_possession_ratio: float = 0.50


@dataclass
class ClipRealismReport:
    passed: bool
    reasons: list[str]
    metrics: dict[str, float]


@dataclass
class SampledClipSpec:
    logical_clip_index: int
    start_frame: int
    seed: int
    realism_report: ClipRealismReport


@dataclass
class CanonicalTrackingSource:
    long_df: pd.DataFrame
    source_name: str
    possession_df: Optional[pd.DataFrame] = None
    source_dataset: Optional[str] = None
    source_match_id: Optional[str] = None
    schema_version: Optional[str] = None


def _natural_sort_key(value: str) -> list[Any]:
    chunks = re.split(r"(\d+)", value)
    out: list[Any] = []
    for c in chunks:
        if c.isdigit():
            out.append(int(c))
        else:
            out.append(c.lower())
    return out


def _normalize_team_label(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return None
    text = str(raw).strip().lower()
    if text in {"home", "h", "team_home", "home_team", "home team"}:
        return AgentType.HOME.value
    if text in {"away", "a", "team_away", "away_team", "away team"}:
        return AgentType.AWAY.value
    return None


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    color = hex_color.lstrip("#")
    if len(color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return (b, g, r)


def _is_close_range(min_v: float, max_v: float, lo: float, hi: float, tol: float) -> bool:
    return min_v >= lo - tol and max_v <= hi + tol


def _detect_axis_mode(values: np.ndarray, dim_m: float, axis_name: str, source_name: str) -> str:
    finite_vals = values[np.isfinite(values)]
    if finite_vals.size == 0:
        raise ValueError(f"{source_name}: no finite values found for axis '{axis_name}'.")

    min_v = float(np.percentile(finite_vals, 0.5))
    max_v = float(np.percentile(finite_vals, 99.5))

    tol_norm = 0.12
    tol_meter = max(1.0, dim_m * 0.02)

    if _is_close_range(min_v, max_v, 0.0, 1.0, tol_norm):
        return "unit"
    if _is_close_range(min_v, max_v, -dim_m / 2.0, dim_m / 2.0, tol_meter):
        return "centered"
    if _is_close_range(min_v, max_v, 0.0, dim_m, tol_meter):
        return "metric"

    abs_min = float(np.min(finite_vals))
    abs_max = float(np.max(finite_vals))
    raise ValueError(
        f"{source_name}: unknown coordinate range on axis '{axis_name}' "
        f"(p0.5={min_v:.4f}, p99.5={max_v:.4f}, abs_min={abs_min:.4f}, "
        f"abs_max={abs_max:.4f}, expected unit [0,1], centered "
        f"[-{dim_m/2:.2f},{dim_m/2:.2f}], or metric [0,{dim_m:.2f}])."
    )


def _normalize_axis(values: np.ndarray, mode: str, dim_m: float) -> np.ndarray:
    out = values.astype(np.float64, copy=True)
    if mode == "unit":
        out *= dim_m
    elif mode == "centered":
        out += dim_m / 2.0
    elif mode == "metric":
        pass
    else:
        raise ValueError(f"Unknown axis mode: {mode}")
    return out


def normalize_coordinates_inplace(
    df: pd.DataFrame,
    pitch_length_m: float,
    pitch_width_m: float,
    source_name: str,
) -> None:
    x_vals = df["x"].to_numpy(dtype=np.float64)
    y_vals = df["y"].to_numpy(dtype=np.float64)

    x_mode = _detect_axis_mode(x_vals, pitch_length_m, "x", source_name)
    y_mode = _detect_axis_mode(y_vals, pitch_width_m, "y", source_name)

    df["x"] = _normalize_axis(x_vals, x_mode, pitch_length_m)
    df["y"] = _normalize_axis(y_vals, y_mode, pitch_width_m)


def _infer_dx_from_possession_labels(
    possession: Optional[np.ndarray],
    home_centroid_x: np.ndarray,
    away_centroid_x: np.ndarray,
) -> Optional[float]:
    if possession is None or len(possession) != len(home_centroid_x):
        return None

    dx_vals: list[float] = []
    for t in range(1, len(possession)):
        team = _normalize_team_label(possession[t - 1])
        if team == AgentType.HOME.value:
            a = home_centroid_x[t - 1]
            b = home_centroid_x[t]
        elif team == AgentType.AWAY.value:
            a = away_centroid_x[t - 1]
            b = away_centroid_x[t]
        else:
            continue
        if np.isfinite(a) and np.isfinite(b):
            dx_vals.append(float(b - a))

    if len(dx_vals) < 3:
        return None
    return float(np.mean(dx_vals))


def _infer_dx_from_centroid_drift(home_centroid_x: np.ndarray, away_centroid_x: np.ndarray) -> Optional[float]:
    home_dx = np.diff(home_centroid_x)
    away_dx = np.diff(away_centroid_x)
    home_dx = home_dx[np.isfinite(home_dx)]
    away_dx = away_dx[np.isfinite(away_dx)]

    if home_dx.size == 0 and away_dx.size == 0:
        return None
    if home_dx.size == 0:
        return float(np.mean(away_dx))
    if away_dx.size == 0:
        return float(np.mean(home_dx))

    home_mean = float(np.mean(home_dx))
    away_mean = float(np.mean(away_dx))
    return home_mean if abs(home_mean) >= abs(away_mean) else away_mean


def normalize_attack_direction(
    coords_xy: np.ndarray,
    clip: ParsedTrackingClip,
) -> tuple[np.ndarray, bool, Optional[float]]:
    coords = coords_xy.astype(np.float32, copy=True)
    home_mask = np.array([a == AgentType.HOME for a in clip.agent_types], dtype=bool)
    away_mask = np.array([a == AgentType.AWAY for a in clip.agent_types], dtype=bool)
    if not np.any(home_mask) or not np.any(away_mask):
        return coords, False, None

    home_centroid_x = np.nanmean(coords[:, home_mask, 0], axis=1)
    away_centroid_x = np.nanmean(coords[:, away_mask, 0], axis=1)

    inferred_dx = _infer_dx_from_possession_labels(
        clip.possession_team_by_frame, home_centroid_x, away_centroid_x
    )
    if inferred_dx is None:
        inferred_dx = _infer_dx_from_centroid_drift(home_centroid_x, away_centroid_x)

    flipped = False
    if inferred_dx is not None and np.isfinite(inferred_dx) and inferred_dx < 0.0:
        coords[:, :, 0] = clip.pitch_length_m - coords[:, :, 0]
        flipped = True

    return coords, flipped, inferred_dx


def infer_possession_by_proximity(coords_xy: np.ndarray, agent_types: np.ndarray) -> np.ndarray:
    home_mask = np.array([a == AgentType.HOME for a in agent_types], dtype=bool)
    away_mask = np.array([a == AgentType.AWAY for a in agent_types], dtype=bool)
    ball_idx = np.where(np.array([a == AgentType.BALL for a in agent_types], dtype=bool))[0]
    if ball_idx.size == 0:
        return np.array([None] * coords_xy.shape[0], dtype=object)

    ball_xy = coords_xy[:, ball_idx[0], :]
    possession = np.array([None] * coords_xy.shape[0], dtype=object)
    for t in range(coords_xy.shape[0]):
        bx, by = float(ball_xy[t, 0]), float(ball_xy[t, 1])
        if not (np.isfinite(bx) and np.isfinite(by)):
            continue

        home_dist = np.inf
        away_dist = np.inf
        if np.any(home_mask):
            dx = coords_xy[t, home_mask, 0] - bx
            dy = coords_xy[t, home_mask, 1] - by
            d = np.sqrt(dx * dx + dy * dy)
            d = d[np.isfinite(d)]
            if d.size > 0:
                home_dist = float(np.min(d))
        if np.any(away_mask):
            dx = coords_xy[t, away_mask, 0] - bx
            dy = coords_xy[t, away_mask, 1] - by
            d = np.sqrt(dx * dx + dy * dy)
            d = d[np.isfinite(d)]
            if d.size > 0:
                away_dist = float(np.min(d))

        if np.isfinite(home_dist) and np.isfinite(away_dist):
            possession[t] = AgentType.HOME.value if home_dist <= away_dist else AgentType.AWAY.value
        elif np.isfinite(home_dist):
            possession[t] = AgentType.HOME.value
        elif np.isfinite(away_dist):
            possession[t] = AgentType.AWAY.value
    return possession


def evaluate_clip_realism(clip: ParsedTrackingClip, config: RealismConfig) -> ClipRealismReport:
    if not config.enabled:
        return ClipRealismReport(passed=True, reasons=[], metrics={})

    coords, _, inferred_dx = normalize_attack_direction(clip.coords_xy, clip)
    agent_types = clip.agent_types
    ball_idx_arr = np.where(np.array([a == AgentType.BALL for a in agent_types], dtype=bool))[0]
    if ball_idx_arr.size == 0:
        return ClipRealismReport(
            passed=False,
            reasons=["ball_missing"],
            metrics={"inferred_dx": float("nan")},
        )
    ball_idx = int(ball_idx_arr[0])
    ball_xy = coords[:, ball_idx, :]
    ball_valid = np.isfinite(ball_xy[:, 0]) & np.isfinite(ball_xy[:, 1])
    in_bounds = (
        ball_valid
        & (ball_xy[:, 0] >= 0.0)
        & (ball_xy[:, 0] <= clip.pitch_length_m)
        & (ball_xy[:, 1] >= 0.0)
        & (ball_xy[:, 1] <= clip.pitch_width_m)
    )
    ball_in_bounds_ratio = float(np.mean(in_bounds.astype(np.float32)))

    pos = clip.possession_team_by_frame
    if pos is None or len(pos) != coords.shape[0]:
        pos = infer_possession_by_proximity(coords, agent_types)
    norm_pos = np.array([_normalize_team_label(x) for x in pos], dtype=object)

    home_count = int(np.sum(norm_pos == AgentType.HOME.value))
    away_count = int(np.sum(norm_pos == AgentType.AWAY.value))
    total_pos = home_count + away_count
    if total_pos == 0:
        return ClipRealismReport(
            passed=False,
            reasons=["possession_unresolved"],
            metrics={"ball_in_bounds_ratio": ball_in_bounds_ratio},
        )

    attacking_team = AgentType.HOME.value if home_count >= away_count else AgentType.AWAY.value
    majority_ratio = float(max(home_count, away_count) / total_pos)
    attack_mask = np.array([a == AgentType(attacking_team) for a in agent_types], dtype=bool)
    defend_mask = np.array([a != AgentType.BALL and a != AgentType(attacking_team) for a in agent_types], dtype=bool)
    if not np.any(attack_mask) or not np.any(defend_mask):
        return ClipRealismReport(
            passed=False,
            reasons=["team_partition_invalid"],
            metrics={"ball_in_bounds_ratio": ball_in_bounds_ratio},
        )

    attack_centroid_x = np.nanmean(coords[:, attack_mask, 0], axis=1)
    defend_centroid_x = np.nanmean(coords[:, defend_mask, 0], axis=1)
    frame_window = max(5, int(coords.shape[0] * 0.2))
    attack_progress_m = float(
        np.nanmean(attack_centroid_x[-frame_window:]) - np.nanmean(attack_centroid_x[:frame_window])
    )

    valid_support = ball_valid & np.isfinite(attack_centroid_x)
    support_ratio = 0.0
    defense_ahead_ratio = 0.0
    if np.any(valid_support):
        closest_attacker = np.full(coords.shape[0], np.nan, dtype=np.float64)
        for t in range(coords.shape[0]):
            if not valid_support[t]:
                continue
            ax = coords[t, attack_mask, 0] - ball_xy[t, 0]
            ay = coords[t, attack_mask, 1] - ball_xy[t, 1]
            d = np.sqrt(ax * ax + ay * ay)
            d = d[np.isfinite(d)]
            if d.size > 0:
                closest_attacker[t] = float(np.min(d))
        finite_close = np.isfinite(closest_attacker)
        if np.any(finite_close):
            support_ratio = float(
                np.mean((closest_attacker[finite_close] <= config.support_distance_m).astype(np.float32))
            )

        defend_ok = np.isfinite(defend_centroid_x) & ball_valid
        if np.any(defend_ok):
            defense_ahead_ratio = float(
                np.mean((defend_centroid_x[defend_ok] >= ball_xy[defend_ok, 0]).astype(np.float32))
            )

    metrics = {
        "ball_in_bounds_ratio": ball_in_bounds_ratio,
        "majority_possession_ratio": majority_ratio,
        "attack_progress_m": attack_progress_m,
        "support_ratio": support_ratio,
        "defense_ahead_ratio": defense_ahead_ratio,
        "inferred_dx": float(inferred_dx) if inferred_dx is not None else float("nan"),
    }

    reasons: list[str] = []
    if ball_in_bounds_ratio < config.min_ball_in_bounds_ratio:
        reasons.append("ball_out_of_bounds")
    if config.min_majority_possession_ratio > 0.0 and majority_ratio < config.min_majority_possession_ratio:
        reasons.append("possession_incoherent")
    if attack_progress_m < config.min_attack_progress_m:
        reasons.append("insufficient_attack_progress")
    if support_ratio < config.min_support_ratio:
        reasons.append("attack_support_too_weak")
    if config.min_defense_ahead_ratio > 0.0 and defense_ahead_ratio < config.min_defense_ahead_ratio:
        reasons.append("defense_shape_unrealistic")

    return ClipRealismReport(passed=(len(reasons) == 0), reasons=reasons, metrics=metrics)


@dataclass
class ClipAnalysis:
    dominant_team: str
    dominant_team_poss_ratio: float
    possession_changes: int
    pass_count: int
    ball_advance_m: float
    ball_lateral_m: float
    clip_duration_s: float
    possession_change_times_s: list[float]


def _stabilize_label_sequence(
    labels: list[Optional[str]],
    min_run_frames: int,
) -> list[Optional[str]]:
    if min_run_frames <= 1 or len(labels) <= 1:
        return list(labels)

    stable = list(labels)
    for _ in range(2):
        run_start = 0
        for idx in range(1, len(stable) + 1):
            if idx < len(stable) and stable[idx] == stable[run_start]:
                continue

            run_value = stable[run_start]
            run_len = idx - run_start
            if run_len < min_run_frames:
                prev_value = stable[run_start - 1] if run_start > 0 else None
                next_value = stable[idx] if idx < len(stable) else None

                replacement = None
                if prev_value == next_value and prev_value is not None:
                    replacement = prev_value
                elif prev_value is not None:
                    replacement = prev_value
                elif next_value is not None:
                    replacement = next_value

                if replacement is not None:
                    for j in range(run_start, idx):
                        stable[j] = replacement

            run_start = idx
    return stable


def _segment_labels(labels: list[Optional[str]]) -> list[tuple[int, int, Optional[str]]]:
    if not labels:
        return []

    segments: list[tuple[int, int, Optional[str]]] = []
    start = 0
    for idx in range(1, len(labels) + 1):
        if idx < len(labels) and labels[idx] == labels[start]:
            continue
        segments.append((start, idx, labels[start]))
        start = idx
    return segments


def analyze_clip_events(
    clip: ParsedTrackingClip,
    coords: np.ndarray,
) -> ClipAnalysis:
    num_frames = coords.shape[0]
    fps = clip.fps
    duration_s = num_frames / fps

    ball_idx_arr = np.where(np.array([a == AgentType.BALL for a in clip.agent_types]))[0]
    if ball_idx_arr.size == 0:
        return ClipAnalysis("unknown", 0.0, 0, 0, 0.0, 0.0, duration_s, [])
    ball_idx = int(ball_idx_arr[0])
    ball_xy = coords[:, ball_idx, :]

    home_mask = np.array([a == AgentType.HOME for a in clip.agent_types], dtype=bool)
    away_mask = np.array([a == AgentType.AWAY for a in clip.agent_types], dtype=bool)
    home_indices = np.where(home_mask)[0]
    away_indices = np.where(away_mask)[0]

    raw_team: list[Optional[str]] = [None] * num_frames
    if clip.possession_team_by_frame is not None and len(clip.possession_team_by_frame) == num_frames:
        raw_team = [_normalize_team_label(value) for value in clip.possession_team_by_frame]
    else:
        inferred = infer_possession_by_proximity(coords, clip.agent_types)
        raw_team = [_normalize_team_label(value) for value in inferred]

    min_team_hold_frames = max(2, int(round(fps * 0.2)))
    stable_team = _stabilize_label_sequence(raw_team, min_team_hold_frames)

    frame_nearest_id: list[Optional[str]] = [None] * num_frames
    control_radius_m = 4.0

    for t in range(num_frames):
        bx, by = float(ball_xy[t, 0]), float(ball_xy[t, 1])
        if not (np.isfinite(bx) and np.isfinite(by)):
            continue

        team_label = stable_team[t]
        if team_label == AgentType.HOME.value:
            candidate_indices = home_indices
        elif team_label == AgentType.AWAY.value:
            candidate_indices = away_indices
        else:
            continue

        best_dist = np.inf
        best_agent: Optional[str] = None

        for ai in candidate_indices:
            px, py = float(coords[t, ai, 0]), float(coords[t, ai, 1])
            if not (np.isfinite(px) and np.isfinite(py)):
                continue
            d = float(np.hypot(px - bx, py - by))
            if d < best_dist:
                best_dist = d
                best_agent = clip.agent_ids[ai]

        if best_agent is not None and best_dist <= control_radius_m:
            frame_nearest_id[t] = best_agent

    min_carrier_hold_frames = max(2, int(round(fps * 0.2)))
    stable_carrier = _stabilize_label_sequence(frame_nearest_id, min_carrier_hold_frames)

    home_count = sum(1 for value in stable_team if value == AgentType.HOME.value)
    away_count = sum(1 for value in stable_team if value == AgentType.AWAY.value)
    if home_count == 0 and away_count == 0:
        return ClipAnalysis("unknown", 0.0, 0, 0, 0.0, 0.0, duration_s, [])

    dominant = AgentType.HOME.value if home_count >= away_count else AgentType.AWAY.value
    dom_ratio = max(home_count, away_count) / max(1, num_frames)

    possession_changes = 0
    change_times: list[float] = []
    prev_team: Optional[str] = None
    for seg_start, _, seg_team in _segment_labels(stable_team):
        if seg_team is None:
            continue
        if prev_team is not None and seg_team != prev_team:
            possession_changes += 1
            change_times.append(seg_start / fps)
        prev_team = seg_team

    pass_count = 0
    prev_carrier: Optional[str] = None
    prev_carrier_team: Optional[str] = None
    for seg_start, seg_end, carrier in _segment_labels(stable_carrier):
        if carrier is None:
            continue
        team_slice = [value for value in stable_team[seg_start:seg_end] if value is not None]
        if not team_slice:
            continue
        team = max(set(team_slice), key=team_slice.count)
        if prev_carrier is not None and prev_carrier_team == team and prev_carrier != carrier:
            pass_count += 1
        prev_carrier = carrier
        prev_carrier_team = team

    valid_start = None
    valid_end = None
    for t in range(num_frames):
        if np.isfinite(ball_xy[t, 0]):
            if valid_start is None:
                valid_start = t
            valid_end = t
    if valid_start is not None and valid_end is not None and valid_start != valid_end:
        ball_advance_m = float(ball_xy[valid_end, 0] - ball_xy[valid_start, 0])
        ball_lateral_m = float(ball_xy[valid_end, 1] - ball_xy[valid_start, 1])
    else:
        ball_advance_m = 0.0
        ball_lateral_m = 0.0

    return ClipAnalysis(
        dominant_team=dominant,
        dominant_team_poss_ratio=dom_ratio,
        possession_changes=possession_changes,
        pass_count=pass_count,
        ball_advance_m=ball_advance_m,
        ball_lateral_m=ball_lateral_m,
        clip_duration_s=duration_s,
        possession_change_times_s=change_times,
    )


def generate_clip_prompt(analysis: ClipAnalysis) -> str:
    team = analysis.dominant_team
    dur = analysis.clip_duration_s
    adv = analysis.ball_advance_m
    lat = analysis.ball_lateral_m
    passes = analysis.pass_count
    changes = analysis.possession_changes

    if abs(adv) < 2.0:
        move_desc = "with the ball held in a relatively static position"
    else:
        fwd = "forward toward the opponent's goal" if adv > 0 else "backward toward their own goal"
        move_desc = f"with the ball advancing {abs(adv):.1f}m {fwd}"

    if abs(lat) > 5.0:
        side = "upper" if lat > 0 else "lower"
        move_desc += f" and shifting {abs(lat):.1f}m toward the {side} flank"

    if passes == 0:
        pass_desc = "No completed passes are detected"
    elif passes == 1:
        pass_desc = f"1 pass is completed among {team} players"
    else:
        pass_desc = f"{passes} passes are completed among {team} players"

    if changes == 0:
        change_desc = "No possession change occurs during this sequence."
    elif changes == 1:
        t_str = f" at approximately {analysis.possession_change_times_s[0]:.1f}s"
        change_desc = f"1 possession change occurs{t_str}."
    else:
        change_desc = f"{changes} possession changes occur during this sequence."

    poss_pct = analysis.dominant_team_poss_ratio * 100
    prompt = (
        f"In this {dur:.0f}-second clip, the {team} team dominates possession "
        f"({poss_pct:.0f}% of frames) moving left to right, "
        f"{move_desc}. {pass_desc}. {change_desc}"
    )
    return prompt


def export_prompt_txt(prompt: str, output_dir: str | Path) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")


class DataParser(ABC):
    def __init__(self, pitch_length_m: float = 105.0, pitch_width_m: float = 68.0) -> None:
        self.pitch_length_m = float(pitch_length_m)
        self.pitch_width_m = float(pitch_width_m)
        self._loaded = False
        self._long_df: Optional[pd.DataFrame] = None
        self._home_ids: list[str] = []
        self._away_ids: list[str] = []
        self._ball_id: Optional[str] = None
        self._possession_by_frame: Optional[pd.Series] = None

    @abstractmethod
    def load(self) -> None:
        ...

    def available_frames(self) -> np.ndarray:
        if not self._loaded or self._long_df is None:
            raise RuntimeError("Parser is not loaded. Call load() first.")
        frames = np.sort(self._long_df["frame_id"].dropna().astype(int).unique())
        if frames.size == 0:
            raise ValueError("No frames available in parsed tracking data.")
        return frames

    def parse_clip(self, start_frame: int, num_frames: int, fps: int) -> ParsedTrackingClip:
        if not self._loaded or self._long_df is None:
            raise RuntimeError("Parser is not loaded. Call load() first.")
        if num_frames <= 0:
            raise ValueError(f"num_frames must be > 0, got {num_frames}.")

        frame_ids = np.arange(start_frame, start_frame + num_frames, dtype=np.int64)
        agent_ids = self._ordered_agent_ids()
        agent_types = np.array([self._agent_type_from_id(aid) for aid in agent_ids], dtype=object)

        clip_df = self._long_df[self._long_df["frame_id"].isin(frame_ids)].copy()

        x_pivot = (
            clip_df.pivot_table(index="frame_id", columns="agent_id", values="x", aggfunc="first")
            .reindex(frame_ids)
            .reindex(columns=agent_ids)
        )
        y_pivot = (
            clip_df.pivot_table(index="frame_id", columns="agent_id", values="y", aggfunc="first")
            .reindex(frame_ids)
            .reindex(columns=agent_ids)
        )

        coords_xy = np.full((num_frames, len(agent_ids), 2), np.nan, dtype=np.float32)

        for col_idx, agent_id in enumerate(agent_ids):
            x_series = x_pivot[agent_id].astype(float).interpolate(method="linear", limit_direction="both")
            y_series = y_pivot[agent_id].astype(float).interpolate(method="linear", limit_direction="both")
            x_series = x_series.ffill().bfill()
            y_series = y_series.ffill().bfill()

            coords_xy[:, col_idx, 0] = x_series.to_numpy(dtype=np.float32)
            coords_xy[:, col_idx, 1] = y_series.to_numpy(dtype=np.float32)

        possession_by_frame: Optional[np.ndarray] = None
        if self._possession_by_frame is not None:
            possession_slice = self._possession_by_frame.reindex(frame_ids).astype(object)
            possession_by_frame = possession_slice.to_numpy(dtype=object)

        return ParsedTrackingClip(
            coords_xy=coords_xy,
            agent_ids=agent_ids,
            agent_types=agent_types,
            frame_ids=frame_ids,
            fps=int(fps),
            pitch_length_m=self.pitch_length_m,
            pitch_width_m=self.pitch_width_m,
            possession_team_by_frame=possession_by_frame,
        )

    def _ordered_agent_ids(self) -> list[str]:
        if not self._home_ids and not self._away_ids:
            raise ValueError("No player agent IDs were parsed from data.")
        if self._ball_id is None:
            raise ValueError("Ball trajectory is missing. A BALL agent is required.")
        return [*self._home_ids, *self._away_ids, self._ball_id]

    def _agent_type_from_id(self, agent_id: str) -> AgentType:
        if agent_id == self._ball_id:
            return AgentType.BALL
        if agent_id in self._home_ids:
            return AgentType.HOME
        if agent_id in self._away_ids:
            return AgentType.AWAY
        raise ValueError(f"Unknown agent ID in type lookup: {agent_id}")

    def _finalize_long_df(
        self,
        long_df: pd.DataFrame,
        source_name: str,
        possession_df: Optional[pd.DataFrame] = None,
    ) -> None:
        required_cols = {"frame_id", "agent_id", "agent_type", "x", "y"}
        missing = required_cols - set(long_df.columns)
        if missing:
            raise ValueError(f"{source_name}: missing required long-form columns: {sorted(missing)}")

        long_df = long_df.copy()
        long_df["frame_id"] = pd.to_numeric(long_df["frame_id"], errors="coerce").astype("Int64")
        long_df["x"] = pd.to_numeric(long_df["x"], errors="coerce")
        long_df["y"] = pd.to_numeric(long_df["y"], errors="coerce")
        long_df = long_df.dropna(subset=["frame_id"]).copy()
        long_df["frame_id"] = long_df["frame_id"].astype(int)

        if long_df.empty:
            raise ValueError(f"{source_name}: parsed tracking data is empty after cleaning.")

        normalize_coordinates_inplace(long_df, self.pitch_length_m, self.pitch_width_m, source_name)

        # Keep first duplicate per frame-agent pair to ensure deterministic canonicalization.
        long_df = long_df.sort_values(["frame_id", "agent_id"]).drop_duplicates(
            subset=["frame_id", "agent_id"], keep="first"
        )

        long_df["agent_type"] = long_df["agent_type"].astype(str).str.lower().map(
            {
                AgentType.HOME.value: AgentType.HOME.value,
                AgentType.AWAY.value: AgentType.AWAY.value,
                AgentType.BALL.value: AgentType.BALL.value,
            }
        )

        if long_df["agent_type"].isna().any():
            bad_rows = long_df[long_df["agent_type"].isna()].head(5)
            raise ValueError(
                f"{source_name}: found invalid agent_type values in parsed records. "
                f"Sample:\n{bad_rows[['frame_id', 'agent_id', 'agent_type']]}"
            )

        home_ids = sorted(
            long_df.loc[long_df["agent_type"] == AgentType.HOME.value, "agent_id"].unique(),
            key=_natural_sort_key,
        )
        away_ids = sorted(
            long_df.loc[long_df["agent_type"] == AgentType.AWAY.value, "agent_id"].unique(),
            key=_natural_sort_key,
        )
        ball_ids = sorted(
            long_df.loc[long_df["agent_type"] == AgentType.BALL.value, "agent_id"].unique(),
            key=_natural_sort_key,
        )

        if len(ball_ids) == 0:
            raise ValueError(f"{source_name}: no BALL agent found.")
        if len(ball_ids) > 1:
            # Canonicalize to one ball agent ID.
            primary_ball = ball_ids[0]
            long_df.loc[long_df["agent_type"] == AgentType.BALL.value, "agent_id"] = primary_ball
            ball_ids = [primary_ball]

        self._home_ids = home_ids
        self._away_ids = away_ids
        self._ball_id = ball_ids[0]
        self._long_df = long_df

        if possession_df is not None and not possession_df.empty:
            p = possession_df.copy()
            if "frame_id" not in p.columns or "possession_team" not in p.columns:
                raise ValueError(
                    f"{source_name}: possession_df must contain frame_id and possession_team columns."
                )
            p["frame_id"] = pd.to_numeric(p["frame_id"], errors="coerce").astype("Int64")
            p = p.dropna(subset=["frame_id"]).copy()
            p["frame_id"] = p["frame_id"].astype(int)
            p["possession_team"] = p["possession_team"].map(_normalize_team_label)
            p = p.dropna(subset=["possession_team"])
            if not p.empty:
                p = p.drop_duplicates(subset=["frame_id"], keep="last").set_index("frame_id")
                self._possession_by_frame = p["possession_team"]
            else:
                self._possession_by_frame = None
        else:
            self._possession_by_frame = None

        self._loaded = True


class BaseTrackingAdapter(ABC):
    dataset_name = "unknown"
    source_name = "Unknown"

    def __init__(self, pitch_length_m: float = 105.0, pitch_width_m: float = 68.0) -> None:
        self.pitch_length_m = float(pitch_length_m)
        self.pitch_width_m = float(pitch_width_m)

    @classmethod
    @abstractmethod
    def required_input_names(cls) -> tuple[str, ...]:
        ...

    @abstractmethod
    def load_canonical(self) -> CanonicalTrackingSource:
        ...

    def _build_canonical_source(
        self,
        long_df: pd.DataFrame,
        source_name: Optional[str] = None,
        possession_df: Optional[pd.DataFrame] = None,
        source_match_id: Optional[str] = None,
        schema_version: Optional[str] = None,
    ) -> CanonicalTrackingSource:
        missing = set(CANONICAL_REQUIRED_COLUMNS) - set(long_df.columns)
        if missing:
            raise ValueError(
                f"{source_name or self.source_name}: canonical source is missing required columns: {sorted(missing)}"
            )

        canonical = long_df.copy()
        for col in CANONICAL_OPTIONAL_COLUMNS:
            if col not in canonical.columns:
                canonical[col] = pd.NA

        if canonical["source_dataset"].isna().all():
            canonical["source_dataset"] = self.dataset_name
        if source_match_id is not None and canonical["source_match_id"].isna().all():
            canonical["source_match_id"] = source_match_id

        clean_possession = possession_df.copy() if possession_df is not None else None
        if clean_possession is not None and not clean_possession.empty:
            required = {"frame_id", "possession_team"}
            missing_pos = required - set(clean_possession.columns)
            if missing_pos:
                raise ValueError(
                    f"{source_name or self.source_name}: possession source is missing columns: {sorted(missing_pos)}"
                )

        return CanonicalTrackingSource(
            long_df=canonical,
            source_name=source_name or self.source_name,
            possession_df=clean_possession,
            source_dataset=self.dataset_name,
            source_match_id=source_match_id,
            schema_version=schema_version,
        )

    @staticmethod
    def _infer_match_id_from_path(path: Path) -> str:
        return path.parent.name if path.parent.name else path.stem


class MetricaAdapter(BaseTrackingAdapter):
    dataset_name = "metrica"
    source_name = "Metrica"

    def __init__(
        self,
        home_csv_path: str | Path,
        away_csv_path: str | Path,
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0,
    ) -> None:
        super().__init__(pitch_length_m=pitch_length_m, pitch_width_m=pitch_width_m)
        self.home_csv_path = Path(home_csv_path)
        self.away_csv_path = Path(away_csv_path)

    @classmethod
    def required_input_names(cls) -> tuple[str, ...]:
        return ("home_csv", "away_csv")

    def load_canonical(self) -> CanonicalTrackingSource:
        home_df = self._read_metrica_csv(self.home_csv_path)
        away_df = self._read_metrica_csv(self.away_csv_path)

        home_long, home_pos = self._wide_to_long(home_df, team_type=AgentType.HOME, source="home")
        away_long, away_pos = self._wide_to_long(away_df, team_type=AgentType.AWAY, source="away")

        combined = pd.concat([home_long, away_long], ignore_index=True)
        combined = combined.drop_duplicates(subset=["frame_id", "agent_id"], keep="first")
        possession = (
            pd.concat([home_pos, away_pos], ignore_index=True)
            if (not home_pos.empty or not away_pos.empty)
            else None
        )
        source_match_id = self._infer_match_id_from_path(self.home_csv_path)
        return self._build_canonical_source(
            combined,
            source_name=self.source_name,
            possession_df=possession,
            source_match_id=source_match_id,
        )

    def _read_metrica_csv(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Metrica file not found: {path}")

        metrica_df = self._try_read_metrica_multiline_header(path)
        if metrica_df is not None:
            return metrica_df

        candidates: list[pd.DataFrame] = []
        read_attempts = [
            {"skiprows": 0, "header": 0},
            {"skiprows": 2, "header": 0},
            {"skiprows": 0, "header": [0, 1]},
        ]
        for params in read_attempts:
            try:
                df = pd.read_csv(path, low_memory=False, **params)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [
                        "_".join([str(c) for c in col if str(c) != "nan"]).strip("_")
                        for col in df.columns
                    ]
                candidates.append(df)
            except Exception:
                continue

        if not candidates:
            raise ValueError(f"Failed to read Metrica CSV: {path}")

        best_df: Optional[pd.DataFrame] = None
        best_score = -1
        for df in candidates:
            cols = [str(c) for c in df.columns]
            score = sum(
                1 for c in cols if re.search(r"(?:^|[_\-\s])x$", c.strip(), flags=re.IGNORECASE)
            )
            if any(c.lower() == "frame" for c in cols):
                score += 3
            if score > best_score:
                best_score = score
                best_df = df

        if best_df is None:
            raise ValueError(f"Could not parse Metrica CSV headers for file: {path}")

        best_df = best_df.copy()
        best_df.columns = [str(c).strip() for c in best_df.columns]
        return best_df

    @staticmethod
    def _try_read_metrica_multiline_header(path: Path) -> Optional[pd.DataFrame]:
        try:
            header_rows = pd.read_csv(path, header=None, nrows=3, low_memory=False)
        except Exception:
            return None

        if header_rows.shape[0] < 3:
            return None

        row3 = ["" if pd.isna(v) else str(v).strip() for v in header_rows.iloc[2].tolist()]
        if "Frame" not in row3 or not any(v.startswith("Player") for v in row3):
            return None

        col_names: list[str] = []
        i = 0
        while i < len(row3):
            token = row3[i]
            token_l = token.lower()
            if token_l == "period":
                col_names.append("Period")
                i += 1
                continue
            if token_l == "frame":
                col_names.append("Frame")
                i += 1
                continue
            if token_l in {"time [s]", "time"}:
                col_names.append("Time [s]")
                i += 1
                continue

            if token.startswith("Player") or token_l == "ball":
                col_names.append(f"{token}_x")
                if i + 1 < len(row3):
                    col_names.append(f"{token}_y")
                    i += 2
                else:
                    i += 1
                continue

            col_names.append(f"col_{i}")
            i += 1

        try:
            df = pd.read_csv(
                path,
                skiprows=3,
                header=None,
                names=col_names,
                usecols=list(range(len(col_names))),
                low_memory=False,
            )
        except Exception:
            return None

        drop_cols = [c for c in df.columns if c.startswith("col_") and df[c].isna().all()]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        return df

    def _wide_to_long(
        self,
        df: pd.DataFrame,
        team_type: AgentType,
        source: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        frame_col = self._find_frame_col(df.columns)
        if frame_col is None:
            raise ValueError(f"Metrica ({source}): could not find frame column in CSV.")

        period_col = self._find_named_col(df.columns, {"period"})
        time_col = self._find_named_col(df.columns, {"time[s]", "time"})
        lower_col_map = {c.lower(): c for c in df.columns}
        pair_map: dict[str, tuple[str, str]] = {}

        for col in df.columns:
            col_clean = col.strip()
            m = re.match(r"(.+?)[_\-\s]x$", col_clean, flags=re.IGNORECASE)
            if not m:
                continue
            base = m.group(1)
            y_candidate = re.sub(r"[_\-\s]x$", "_y", col_clean, flags=re.IGNORECASE)
            if y_candidate.lower() in lower_col_map:
                pair_map[base] = (col, lower_col_map[y_candidate.lower()])
                continue

            y_candidate2 = re.sub(r"x$", "y", col_clean, flags=re.IGNORECASE)
            if y_candidate2.lower() in lower_col_map:
                pair_map[base] = (col, lower_col_map[y_candidate2.lower()])

        if not pair_map:
            raise ValueError(
                f"Metrica ({source}): no x/y tracking column pairs detected. "
                f"Expected columns like '<agent>_x' and '<agent>_y'."
            )

        records: list[pd.DataFrame] = []
        team_id = team_type.value
        source_match_id = self._infer_match_id_from_path(self.home_csv_path)
        for base, (x_col, y_col) in pair_map.items():
            base_l = base.lower()
            if "ball" in base_l:
                agent_id = "ball"
                agent_type = AgentType.BALL.value
                player_id = pd.NA
                obj_team_id = pd.NA
            else:
                clean_base = re.sub(r"[^a-zA-Z0-9]+", "_", base).strip("_")
                agent_id = f"{team_id}_{clean_base}"
                agent_type = team_type.value
                player_id = clean_base
                obj_team_id = team_id

            data: dict[str, Any] = {
                "frame_id": df[frame_col],
                "agent_id": agent_id,
                "agent_type": agent_type,
                "x": df[x_col],
                "y": df[y_col],
                "team_id": obj_team_id,
                "player_id": player_id,
                "source_dataset": self.dataset_name,
                "source_match_id": source_match_id,
            }
            if period_col is not None:
                data["period"] = df[period_col]
            if time_col is not None:
                data["timestamp"] = df[time_col]
            records.append(pd.DataFrame(data))

        long_df = pd.concat(records, ignore_index=True)

        possession_col = self._find_possession_col(df.columns)
        if possession_col is not None:
            possession_df = pd.DataFrame(
                {
                    "frame_id": df[frame_col],
                    "possession_team": df[possession_col],
                }
            )
        else:
            possession_df = pd.DataFrame(columns=["frame_id", "possession_team"])

        return long_df, possession_df

    @staticmethod
    def _find_frame_col(columns: Iterable[str]) -> Optional[str]:
        cols = list(columns)
        direct = {"frame", "frame_id", "frameid"}
        for c in cols:
            if c.lower().replace(" ", "") in direct:
                return c
        return cols[0] if cols else None

    @staticmethod
    def _find_named_col(columns: Iterable[str], names: set[str]) -> Optional[str]:
        normalized = {name.replace(" ", "").lower() for name in names}
        for c in columns:
            if c.replace(" ", "").lower() in normalized:
                return c
        return None

    @staticmethod
    def _find_possession_col(columns: Iterable[str]) -> Optional[str]:
        for c in columns:
            if "possession" in c.lower():
                return c
        return None


class SkillCornerAdapter(BaseTrackingAdapter):
    dataset_name = "skillcorner_v1"
    source_name = "SkillCorner"
    schema_version = "skillcorner_tracking_v1"

    def __init__(
        self,
        tracking_json_path: str | Path,
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0,
    ) -> None:
        super().__init__(pitch_length_m=pitch_length_m, pitch_width_m=pitch_width_m)
        self.tracking_json_path = Path(tracking_json_path)

    @classmethod
    def required_input_names(cls) -> tuple[str, ...]:
        return ("tracking_json",)

    def load_canonical(self) -> CanonicalTrackingSource:
        if not self.tracking_json_path.exists():
            raise FileNotFoundError(f"SkillCorner file not found: {self.tracking_json_path}")

        with self.tracking_json_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        detected_version = self._detect_schema_version(raw)
        if detected_version != self.schema_version:
            raise ValueError(
                f"SkillCorner: detected unsupported schema '{detected_version}'. "
                f"This pipeline currently supports only '{self.schema_version}'."
            )

        frames = self._extract_frames_v1(raw)
        team_map = self._build_team_map_v1(raw)
        source_match_id = self._extract_match_id_v1(raw)

        records: list[dict[str, Any]] = []
        possession_records: list[dict[str, Any]] = []
        for idx, frame_obj in enumerate(frames):
            frame_id = self._extract_frame_id_v1(frame_obj, idx)
            timestamp = frame_obj.get("timestamp", frame_obj.get("time"))
            period = frame_obj.get("period")
            possession_team = self._extract_possession_v1(frame_obj, team_map)
            if possession_team is not None:
                possession_records.append({"frame_id": frame_id, "possession_team": possession_team})

            objects = self._extract_objects_v1(frame_obj, frame_id)
            for obj_idx, obj in enumerate(objects):
                if not isinstance(obj, dict):
                    raise ValueError(
                        f"SkillCorner: frame={frame_id}, object index {obj_idx} is not an object."
                    )

                agent_type, team_id = self._extract_agent_type_v1(obj, team_map, frame_id, obj_idx)
                player_id, agent_id = self._extract_agent_id_v1(obj, agent_type, frame_id, obj_idx)
                x_val, y_val = self._extract_xy_v1(obj, frame_id, obj_idx)

                records.append(
                    {
                        "frame_id": frame_id,
                        "agent_id": agent_id,
                        "agent_type": agent_type.value,
                        "x": x_val,
                        "y": y_val,
                        "period": period,
                        "timestamp": timestamp,
                        "team_id": team_id,
                        "player_id": player_id,
                        "source_dataset": self.dataset_name,
                        "source_match_id": source_match_id,
                    }
                )

        if not records:
            raise ValueError("SkillCorner: no tracking records parsed from JSON.")

        return self._build_canonical_source(
            pd.DataFrame(records),
            source_name=self.source_name,
            possession_df=pd.DataFrame(possession_records),
            source_match_id=source_match_id,
            schema_version=self.schema_version,
        )

    def _detect_schema_version(self, raw: Any) -> str:
        if not isinstance(raw, dict):
            raise ValueError("SkillCorner: root JSON must be an object for supported schemas.")
        if isinstance(raw.get("frames"), list) and isinstance(raw.get("teams"), list):
            return "skillcorner_tracking_v1"
        if isinstance(raw.get("tracking"), list):
            return "skillcorner_tracking_v2"
        raise ValueError(
            "SkillCorner: unsupported schema. Expected top-level keys 'frames' and 'teams' "
            "for skillcorner_tracking_v1."
        )

    @staticmethod
    def _extract_frames_v1(raw: dict[str, Any]) -> list[dict[str, Any]]:
        frames = raw.get("frames")
        if not isinstance(frames, list):
            raise ValueError("SkillCorner v1: missing required top-level list 'frames'.")
        return frames

    @staticmethod
    def _extract_match_id_v1(raw: dict[str, Any]) -> Optional[str]:
        for key in ("match_id", "game_id", "id"):
            if key in raw and raw[key] is not None:
                return str(raw[key])
        metadata = raw.get("metadata")
        if isinstance(metadata, dict):
            for key in ("match_id", "game_id", "id"):
                if key in metadata and metadata[key] is not None:
                    return str(metadata[key])
        return None

    def _build_team_map_v1(self, raw: dict[str, Any]) -> dict[str, str]:
        teams_obj = raw.get("teams")
        if not isinstance(teams_obj, list) or not teams_obj:
            raise ValueError("SkillCorner v1: missing required top-level team list 'teams'.")

        team_map: dict[str, str] = {}
        for idx, t in enumerate(teams_obj):
            if not isinstance(t, dict):
                raise ValueError(f"SkillCorner v1: teams[{idx}] must be an object.")
            role = None
            for key in ("role", "side", "type", "name"):
                if key in t:
                    role = _normalize_team_label(t[key])
                    if role is not None:
                        break
            team_id = t.get("id", t.get("team_id"))
            if team_id is None or role is None:
                raise ValueError(
                    f"SkillCorner v1: teams[{idx}] must include an id/team_id and a home/away role."
                )
            team_map[str(team_id)] = role
            team_map[str(team_id).lower()] = role
        return team_map

    @staticmethod
    def _extract_frame_id_v1(frame_obj: dict[str, Any], frame_index: int) -> int:
        for key in ("frame_id", "id"):
            if key in frame_obj:
                try:
                    return int(frame_obj[key])
                except Exception as exc:
                    raise ValueError(
                        f"SkillCorner v1: invalid frame id in key '{key}' at index {frame_index}: {frame_obj[key]}"
                    ) from exc
        raise ValueError(f"SkillCorner v1: missing frame_id at frame index {frame_index}.")

    @staticmethod
    def _extract_objects_v1(frame_obj: dict[str, Any], frame_id: int) -> list[dict[str, Any]]:
        objects = frame_obj.get("objects")
        if not isinstance(objects, list):
            raise ValueError(
                f"SkillCorner v1: frame={frame_id} must contain an 'objects' list."
            )
        return objects

    @staticmethod
    def _extract_possession_v1(frame_obj: dict[str, Any], team_map: dict[str, str]) -> Optional[str]:
        for key in ("possession_team", "team_in_possession"):
            if key not in frame_obj:
                continue
            raw = frame_obj[key]
            if raw is None:
                continue
            mapped = _normalize_team_label(raw)
            if mapped is not None:
                return mapped
            raw_str = str(raw)
            if raw_str in team_map:
                return team_map[raw_str]
            if raw_str.lower() in team_map:
                return team_map[raw_str.lower()]
        return None

    @staticmethod
    def _extract_agent_type_v1(
        obj: dict[str, Any],
        team_map: dict[str, str],
        frame_id: int,
        obj_idx: int,
    ) -> tuple[AgentType, Any]:
        if obj.get("is_ball") is True:
            return AgentType.BALL, pd.NA

        obj_type = obj.get("object_type", obj.get("type"))
        if isinstance(obj_type, str) and "ball" in obj_type.lower():
            return AgentType.BALL, pd.NA

        for key in ("team_id", "team", "team_name"):
            if key not in obj or obj[key] is None:
                continue
            raw_team = obj[key]
            mapped = _normalize_team_label(raw_team)
            if mapped is not None:
                return AgentType(mapped), raw_team
            raw_team_str = str(raw_team)
            if raw_team_str in team_map:
                return AgentType(team_map[raw_team_str]), raw_team
            if raw_team_str.lower() in team_map:
                return AgentType(team_map[raw_team_str.lower()]), raw_team

        raise ValueError(
            f"SkillCorner v1: frame={frame_id}, object_index={obj_idx} missing a valid team identifier."
        )

    @staticmethod
    def _extract_agent_id_v1(
        obj: dict[str, Any],
        agent_type: AgentType,
        frame_id: int,
        obj_idx: int,
    ) -> tuple[Any, str]:
        if agent_type == AgentType.BALL:
            return pd.NA, "ball"

        for key in ("player_id", "track_id", "object_id", "id"):
            if key in obj and obj[key] is not None:
                raw = str(obj[key]).strip()
                if raw:
                    return raw, f"{agent_type.value}_{raw}"

        raise ValueError(
            f"SkillCorner v1: frame={frame_id}, object_index={obj_idx} missing player identifier."
        )

    @staticmethod
    def _extract_xy_v1(obj: dict[str, Any], frame_id: int, obj_idx: int) -> tuple[float, float]:
        if "x" in obj and "y" in obj:
            return float(obj["x"]), float(obj["y"])

        position = obj.get("position")
        if isinstance(position, dict) and "x" in position and "y" in position:
            return float(position["x"]), float(position["y"])
        if isinstance(position, (list, tuple)) and len(position) >= 2:
            return float(position[0]), float(position[1])

        coordinates = obj.get("coordinates")
        if isinstance(coordinates, dict) and "x" in coordinates and "y" in coordinates:
            return float(coordinates["x"]), float(coordinates["y"])
        if isinstance(coordinates, (list, tuple)) and len(coordinates) >= 2:
            return float(coordinates[0]), float(coordinates[1])

        raise ValueError(
            f"SkillCorner v1: frame={frame_id}, object_index={obj_idx} missing x/y coordinates."
        )


class SkillCornerV2Adapter(BaseTrackingAdapter):
    dataset_name = "skillcorner_v2"
    source_name = "SkillCorner"
    schema_version = "skillcorner_tracking_v2"

    def __init__(
        self,
        tracking_json_path: str | Path,
        match_json_path: str | Path | None = None,
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0,
    ) -> None:
        super().__init__(pitch_length_m=pitch_length_m, pitch_width_m=pitch_width_m)
        self.tracking_json_path = Path(tracking_json_path)
        self.match_json_path = Path(match_json_path) if match_json_path is not None else None

    @classmethod
    def required_input_names(cls) -> tuple[str, ...]:
        return ("tracking_json",)

    def load_canonical(self) -> CanonicalTrackingSource:
        if not self.tracking_json_path.exists():
            raise FileNotFoundError(f"SkillCorner file not found: {self.tracking_json_path}")
        match_path = self._resolve_match_json_path()
        if not match_path.exists():
            raise FileNotFoundError(
                f"SkillCorner v2: required companion match file not found: {match_path}"
            )

        with match_path.open("r", encoding="utf-8") as f:
            match_obj = json.load(f)

        team_role_by_team_id, player_team_by_player_id = self._build_match_maps_v2(match_obj)
        source_match_id = self._extract_match_id_v2(match_obj, match_path)

        records: list[dict[str, Any]] = []
        possession_records: list[dict[str, Any]] = []
        with self.tracking_json_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    frame_obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"SkillCorner v2: invalid JSONL at line {line_idx} in {self.tracking_json_path}."
                    ) from exc

                frame_id = self._extract_frame_id_v2(frame_obj, line_idx)
                timestamp = frame_obj.get("timestamp")
                period = frame_obj.get("period")

                possession_team = self._extract_possession_v2(
                    frame_obj, player_team_by_player_id, team_role_by_team_id
                )
                if possession_team is not None:
                    possession_records.append({"frame_id": frame_id, "possession_team": possession_team})

                ball_data = frame_obj.get("ball_data")
                if isinstance(ball_data, dict):
                    bx, by = ball_data.get("x"), ball_data.get("y")
                    if bx is not None and by is not None:
                        records.append(
                            {
                                "frame_id": frame_id,
                                "agent_id": "ball",
                                "agent_type": AgentType.BALL.value,
                                "x": bx,
                                "y": by,
                                "period": period,
                                "timestamp": timestamp,
                                "team_id": pd.NA,
                                "player_id": pd.NA,
                                "source_dataset": self.dataset_name,
                                "source_match_id": source_match_id,
                            }
                        )

                player_data = frame_obj.get("player_data")
                if player_data is None:
                    raise ValueError(f"SkillCorner v2: frame={frame_id} missing required 'player_data' list.")
                if not isinstance(player_data, list):
                    raise ValueError(f"SkillCorner v2: frame={frame_id} field 'player_data' must be a list.")

                for obj_idx, obj in enumerate(player_data):
                    if not isinstance(obj, dict):
                        raise ValueError(
                            f"SkillCorner v2: frame={frame_id}, player_data[{obj_idx}] must be an object."
                        )
                    player_id = obj.get("player_id")
                    if player_id is None:
                        raise ValueError(
                            f"SkillCorner v2: frame={frame_id}, player_data[{obj_idx}] missing player_id."
                        )
                    if player_id not in player_team_by_player_id:
                        raise ValueError(
                            f"SkillCorner v2: frame={frame_id}, player_id={player_id} not found in companion match metadata."
                        )
                    team_id = player_team_by_player_id[player_id]
                    team_role = team_role_by_team_id.get(team_id)
                    if team_role is None:
                        raise ValueError(
                            f"SkillCorner v2: frame={frame_id}, player_id={player_id} has unknown team_id={team_id}."
                        )
                    x_val, y_val = self._extract_xy_v2(obj, frame_id, obj_idx)
                    records.append(
                        {
                            "frame_id": frame_id,
                            "agent_id": f"{team_role}_{player_id}",
                            "agent_type": team_role,
                            "x": x_val,
                            "y": y_val,
                            "period": period,
                            "timestamp": timestamp,
                            "team_id": team_id,
                            "player_id": player_id,
                            "source_dataset": self.dataset_name,
                            "source_match_id": source_match_id,
                        }
                    )

        if not records:
            raise ValueError("SkillCorner v2: no tracking records parsed from JSONL.")

        return self._build_canonical_source(
            pd.DataFrame(records),
            source_name=self.source_name,
            possession_df=pd.DataFrame(possession_records),
            source_match_id=source_match_id,
            schema_version=self.schema_version,
        )

    def _resolve_match_json_path(self) -> Path:
        if self.match_json_path is not None:
            return self.match_json_path
        filename = self.tracking_json_path.name
        if filename.endswith("_tracking_extrapolated.jsonl"):
            return self.tracking_json_path.with_name(
                filename.replace("_tracking_extrapolated.jsonl", "_match.json")
            )
        return self.tracking_json_path.with_suffix(".match.json")

    @staticmethod
    def _build_match_maps_v2(match_obj: dict[str, Any]) -> tuple[dict[Any, str], dict[Any, Any]]:
        if not isinstance(match_obj, dict):
            raise ValueError("SkillCorner v2: companion match file must be a JSON object.")

        home_team = match_obj.get("home_team")
        away_team = match_obj.get("away_team")
        if not isinstance(home_team, dict) or not isinstance(away_team, dict):
            raise ValueError("SkillCorner v2: companion match file must include home_team and away_team objects.")

        team_role_by_team_id = {
            home_team.get("id"): AgentType.HOME.value,
            away_team.get("id"): AgentType.AWAY.value,
        }
        if None in team_role_by_team_id:
            raise ValueError("SkillCorner v2: companion match file must include home_team.id and away_team.id.")

        players = match_obj.get("players")
        if not isinstance(players, list):
            raise ValueError("SkillCorner v2: companion match file must include a players list.")

        player_team_by_player_id: dict[Any, Any] = {}
        for idx, player in enumerate(players):
            if not isinstance(player, dict):
                raise ValueError(f"SkillCorner v2: players[{idx}] must be an object.")
            player_id = player.get("id")
            team_id = player.get("team_id")
            if player_id is None or team_id is None:
                continue
            player_team_by_player_id[player_id] = team_id

        if not player_team_by_player_id:
            raise ValueError("SkillCorner v2: companion match file contains no usable player/team mappings.")
        return team_role_by_team_id, player_team_by_player_id

    @staticmethod
    def _extract_match_id_v2(match_obj: dict[str, Any], match_path: Path) -> str:
        if match_obj.get("id") is not None:
            return str(match_obj["id"])
        return match_path.parent.name if match_path.parent.name else match_path.stem

    @staticmethod
    def _extract_frame_id_v2(frame_obj: dict[str, Any], line_idx: int) -> int:
        if "frame" not in frame_obj:
            raise ValueError(f"SkillCorner v2: line {line_idx} missing required key 'frame'.")
        try:
            return int(frame_obj["frame"])
        except Exception as exc:
            raise ValueError(
                f"SkillCorner v2: invalid frame value at line {line_idx}: {frame_obj['frame']}"
            ) from exc

    @staticmethod
    def _extract_possession_v2(
        frame_obj: dict[str, Any],
        player_team_by_player_id: dict[Any, Any],
        team_role_by_team_id: dict[Any, str],
    ) -> Optional[str]:
        possession = frame_obj.get("possession")
        if not isinstance(possession, dict):
            return None

        mapped = _normalize_team_label(possession.get("group"))
        if mapped is not None:
            return mapped

        player_id = possession.get("player_id")
        if player_id in player_team_by_player_id:
            return team_role_by_team_id.get(player_team_by_player_id[player_id])
        return None

    @staticmethod
    def _extract_xy_v2(obj: dict[str, Any], frame_id: int, obj_idx: int) -> tuple[float, float]:
        if "x" not in obj or "y" not in obj:
            raise ValueError(
                f"SkillCorner v2: frame={frame_id}, player_data[{obj_idx}] missing x/y coordinates."
            )
        return float(obj["x"]), float(obj["y"])


class AdapterBackedParser(DataParser):
    def __init__(self, adapter: BaseTrackingAdapter) -> None:
        super().__init__(pitch_length_m=adapter.pitch_length_m, pitch_width_m=adapter.pitch_width_m)
        self.adapter = adapter
        self._source_dataset: Optional[str] = None
        self._source_match_id: Optional[str] = None
        self._schema_version: Optional[str] = None

    def load(self) -> None:
        canonical = self.adapter.load_canonical()
        self._source_dataset = canonical.source_dataset
        self._source_match_id = canonical.source_match_id
        self._schema_version = canonical.schema_version
        self._finalize_long_df(
            canonical.long_df,
            source_name=canonical.source_name,
            possession_df=canonical.possession_df,
        )


class MetricaParser(AdapterBackedParser):
    def __init__(
        self,
        home_csv_path: str | Path,
        away_csv_path: str | Path,
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0,
    ) -> None:
        super().__init__(
            MetricaAdapter(
                home_csv_path=home_csv_path,
                away_csv_path=away_csv_path,
                pitch_length_m=pitch_length_m,
                pitch_width_m=pitch_width_m,
            )
        )


class SkillCornerParser(AdapterBackedParser):
    def __init__(
        self,
        tracking_json_path: str | Path,
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0,
    ) -> None:
        super().__init__(
            SkillCornerAdapter(
                tracking_json_path=tracking_json_path,
                pitch_length_m=pitch_length_m,
                pitch_width_m=pitch_width_m,
            )
        )


class SkillCornerV2Parser(AdapterBackedParser):
    def __init__(
        self,
        tracking_json_path: str | Path,
        match_json_path: str | Path | None = None,
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0,
    ) -> None:
        super().__init__(
            SkillCornerV2Adapter(
                tracking_json_path=tracking_json_path,
                match_json_path=match_json_path,
                pitch_length_m=pitch_length_m,
                pitch_width_m=pitch_width_m,
            )
        )


class BEVRenderer:
    PITCH_GREEN = _hex_to_bgr("#1A6B37")
    PITCH_GREEN_ALT = _hex_to_bgr("#1E7A3F")
    LINE_WHITE = (255, 255, 255)
    GOAL_NET_COLOR = _hex_to_bgr("#CCCCCC")

    PENALTY_AREA_DEPTH = 16.5
    PENALTY_AREA_WIDTH = 40.32
    GOAL_AREA_DEPTH = 5.5
    GOAL_AREA_WIDTH = 18.32
    CENTER_CIRCLE_RADIUS = 9.15
    PENALTY_SPOT_DIST = 11.0
    GOAL_WIDTH = 7.32
    GOAL_DEPTH = 2.0
    CORNER_ARC_RADIUS = 1.0

    def __init__(
        self,
        width: int = 800,
        height: int = 520,
        home_color: str = "#E63946",
        away_color: str = "#457B9D",
        ball_color: str = "#F5F500",
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f"Renderer width/height must be > 0, got {width}x{height}.")
        self.width = int(width)
        self.height = int(height)
        self.home_color = _hex_to_bgr(home_color)
        self.away_color = _hex_to_bgr(away_color)
        self.ball_color = _hex_to_bgr(ball_color)

    def _to_px(self, x_m: float, y_m: float, scale: float, pad_x: float, pad_y: float) -> tuple[int, int]:
        px = int(round(pad_x + x_m * scale))
        py = int(round(self.height - 1 - (pad_y + y_m * scale)))
        return px, py

    def _draw_pitch(self, frame: np.ndarray, clip: ParsedTrackingClip, scale: float, pad_x: float, pad_y: float) -> None:
        L = clip.pitch_length_m
        W = clip.pitch_width_m
        lw = max(1, int(round(scale * 0.3)))

        stripe_w_m = L / 16.0
        for i in range(16):
            x0_m = i * stripe_w_m
            x1_m = (i + 1) * stripe_w_m
            tl = self._to_px(x0_m, W, scale, pad_x, pad_y)
            br = self._to_px(x1_m, 0, scale, pad_x, pad_y)
            color = self.PITCH_GREEN if i % 2 == 0 else self.PITCH_GREEN_ALT
            cv2.rectangle(frame, tl, br, color, thickness=-1)

        def rect(x0: float, y0: float, x1: float, y1: float) -> None:
            p1 = self._to_px(x0, y1, scale, pad_x, pad_y)
            p2 = self._to_px(x1, y0, scale, pad_x, pad_y)
            cv2.rectangle(frame, p1, p2, self.LINE_WHITE, thickness=lw)

        rect(0, 0, L, W)
        mid = self._to_px(L / 2, 0, scale, pad_x, pad_y)
        mid_top = self._to_px(L / 2, W, scale, pad_x, pad_y)
        cv2.line(frame, mid, mid_top, self.LINE_WHITE, thickness=lw)

        center = self._to_px(L / 2, W / 2, scale, pad_x, pad_y)
        cv2.circle(frame, center, int(round(self.CENTER_CIRCLE_RADIUS * scale)), self.LINE_WHITE, thickness=lw)
        cv2.circle(frame, center, max(2, int(round(0.3 * scale))), self.LINE_WHITE, thickness=-1)

        pw = self.PENALTY_AREA_WIDTH
        pd_ = self.PENALTY_AREA_DEPTH
        gw = self.GOAL_AREA_WIDTH
        gd = self.GOAL_AREA_DEPTH

        rect(0, (W - pw) / 2, pd_, (W + pw) / 2)
        rect(L - pd_, (W - pw) / 2, L, (W + pw) / 2)
        rect(0, (W - gw) / 2, gd, (W + gw) / 2)
        rect(L - gd, (W - gw) / 2, L, (W + gw) / 2)

        ps_left = self._to_px(self.PENALTY_SPOT_DIST, W / 2, scale, pad_x, pad_y)
        ps_right = self._to_px(L - self.PENALTY_SPOT_DIST, W / 2, scale, pad_x, pad_y)
        spot_r = max(2, int(round(0.25 * scale)))
        cv2.circle(frame, ps_left, spot_r, self.LINE_WHITE, thickness=-1)
        cv2.circle(frame, ps_right, spot_r, self.LINE_WHITE, thickness=-1)

        arc_r = int(round(self.CENTER_CIRCLE_RADIUS * scale))
        arc_center_left = self._to_px(self.PENALTY_SPOT_DIST, W / 2, scale, pad_x, pad_y)
        arc_center_right = self._to_px(L - self.PENALTY_SPOT_DIST, W / 2, scale, pad_x, pad_y)
        cv2.ellipse(frame, arc_center_left, (arc_r, arc_r), 0, -53, 53, self.LINE_WHITE, thickness=lw)
        cv2.ellipse(frame, arc_center_right, (arc_r, arc_r), 0, 127, 233, self.LINE_WHITE, thickness=lw)

        corner_r = max(2, int(round(self.CORNER_ARC_RADIUS * scale)))
        for cx, cy, sa in [(0, 0, 0), (L, 0, 90), (L, W, 180), (0, W, 270)]:
            cp = self._to_px(cx, cy, scale, pad_x, pad_y)
            cv2.ellipse(frame, cp, (corner_r, corner_r), 0, sa, sa + 90, self.LINE_WHITE, thickness=lw)

        goal_hw = self.GOAL_WIDTH / 2
        goal_d = self.GOAL_DEPTH
        g_tl = self._to_px(-goal_d, W / 2 + goal_hw, scale, pad_x, pad_y)
        g_br = self._to_px(0, W / 2 - goal_hw, scale, pad_x, pad_y)
        cv2.rectangle(frame, g_tl, g_br, self.GOAL_NET_COLOR, thickness=-1)
        cv2.rectangle(frame, g_tl, g_br, self.LINE_WHITE, thickness=lw)

        g_tl2 = self._to_px(L, W / 2 + goal_hw, scale, pad_x, pad_y)
        g_br2 = self._to_px(L + goal_d, W / 2 - goal_hw, scale, pad_x, pad_y)
        cv2.rectangle(frame, g_tl2, g_br2, self.GOAL_NET_COLOR, thickness=-1)
        cv2.rectangle(frame, g_tl2, g_br2, self.LINE_WHITE, thickness=lw)

    def render_frames(self, clip: ParsedTrackingClip, normalize_orientation: bool = True) -> list[np.ndarray]:
        coords = clip.coords_xy.astype(np.float32, copy=True)
        if normalize_orientation:
            coords, _, _ = normalize_attack_direction(coords, clip)

        margin_m = 4.0
        total_w_m = clip.pitch_length_m + 2 * margin_m
        total_h_m = clip.pitch_width_m + 2 * margin_m
        scale = min(self.width / total_w_m, self.height / total_h_m)
        used_w = total_w_m * scale
        used_h = total_h_m * scale
        pad_x = (self.width - used_w) / 2.0 + margin_m * scale
        pad_y = (self.height - used_h) / 2.0 + margin_m * scale

        player_r = max(3, int(round(0.9 * scale)))
        ball_r = max(2, int(round(0.55 * scale)))
        outline_r = player_r + max(1, int(round(0.15 * scale)))

        pitch_bg = np.full((self.height, self.width, 3), self.PITCH_GREEN[0], dtype=np.uint8)
        pitch_bg[:, :] = self.PITCH_GREEN
        self._draw_pitch(pitch_bg, clip, scale, pad_x, pad_y)

        frames: list[np.ndarray] = []
        for t in range(coords.shape[0]):
            frame = pitch_bg.copy()

            for i, agent_type in enumerate(clip.agent_types):
                x_m = float(coords[t, i, 0])
                y_m = float(coords[t, i, 1])
                if not (np.isfinite(x_m) and np.isfinite(y_m)):
                    continue

                px, py = self._to_px(x_m, y_m, scale, pad_x, pad_y)
                px = int(np.clip(px, 0, self.width - 1))
                py = int(np.clip(py, 0, self.height - 1))

                if agent_type == AgentType.BALL:
                    cv2.circle(frame, (px, py), ball_r + 1, (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
                    cv2.circle(frame, (px, py), ball_r, self.ball_color, thickness=-1, lineType=cv2.LINE_AA)
                else:
                    color = self.home_color if agent_type == AgentType.HOME else self.away_color
                    cv2.circle(frame, (px, py), outline_r, (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
                    cv2.circle(frame, (px, py), player_r, color, thickness=-1, lineType=cv2.LINE_AA)

            frames.append(frame)

        return frames


def export_vbvr_clip(frames: list[np.ndarray], output_dir: str | Path, fps: int) -> None:
    if not frames:
        raise ValueError("No frames to export.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    keep_names = {"ground_truth.mp4", "first_frame.png", "final_frame.png", "prompt.txt"}
    for ext in ("*.mp4", "*.png", "*.jpg", "*.jpeg"):
        for candidate in out_dir.glob(ext):
            if candidate.name not in keep_names and candidate.is_file():
                candidate.unlink()

    video_path = out_dir / "ground_truth.mp4"
    first_path = out_dir / "first_frame.png"
    last_path = out_dir / "final_frame.png"

    h, w = frames[0].shape[:2]
    for idx, fr in enumerate(frames):
        if fr.shape[:2] != (h, w):
            raise ValueError(
                f"Inconsistent frame size at index {idx}: expected {(h, w)}, got {fr.shape[:2]}"
            )

    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"avc1"), float(fps), (w, h))
    used_codec = "avc1"
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
        used_codec = "mp4v"

    if not writer.isOpened():
        raise RuntimeError("Failed to initialize VideoWriter with both avc1 and mp4v codecs.")

    if used_codec != "avc1":
        print("Warning: 'avc1' unavailable, fell back to 'mp4v'.")

    for fr in frames:
        writer.write(fr)
    writer.release()

    ok1 = cv2.imwrite(str(first_path), frames[0])
    ok2 = cv2.imwrite(str(last_path), frames[-1])
    if not ok1 or not ok2:
        raise RuntimeError("Failed to write first_frame.png or final_frame.png.")


def _clip_start_bounds(parser: DataParser, num_frames: int) -> tuple[int, int]:
    frames = parser.available_frames()
    min_f = int(frames.min())
    max_f = int(frames.max())
    last_start = max_f - num_frames + 1
    if last_start < min_f:
        raise ValueError(
            f"Not enough frames for requested clip length {num_frames}. "
            f"Available range: [{min_f}, {max_f}]"
        )
    return min_f, last_start


def _derived_seed(global_seed: int, logical_clip_index: int) -> int:
    seq = np.random.SeedSequence([int(global_seed), int(logical_clip_index)])
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def sample_clip_specs(
    parser: DataParser,
    num_frames: int,
    fps: int,
    num_clips: int,
    global_seed: int,
    clip_index_offset: int,
    start_frame_override: Optional[int],
    max_sampling_attempts: int,
    realism_config: RealismConfig,
    allow_duplicate_starts: bool,
) -> list[SampledClipSpec]:
    if num_clips <= 0:
        raise ValueError(f"num_clips must be > 0, got {num_clips}.")

    min_f, last_start = _clip_start_bounds(parser, num_frames)
    sampled: list[SampledClipSpec] = []
    used_starts: set[int] = set()
    total_unique = last_start - min_f + 1

    if (
        start_frame_override is None
        and not allow_duplicate_starts
        and num_clips > total_unique
    ):
        raise ValueError(
            "Requested clips exceed unique start windows for this source. "
            f"num_clips={num_clips}, unique_windows={total_unique}, "
            f"frame_range=[{min_f},{last_start}], clip_num_frames={num_frames}. "
            "Use shorter --seconds, add more source matches, or pass --allow_duplicate_starts."
        )

    if start_frame_override is not None:
        if num_clips != 1:
            raise ValueError("--start_frame can only be used when --num_clips=1.")
        if start_frame_override < min_f or start_frame_override > last_start:
            raise ValueError(
                f"start_frame={start_frame_override} out of valid range [{min_f}, {last_start}]"
            )
        logical_idx = clip_index_offset
        clip = parser.parse_clip(start_frame=start_frame_override, num_frames=num_frames, fps=fps)
        report = evaluate_clip_realism(clip, realism_config)
        if realism_config.enabled and not report.passed:
            raise ValueError(
                f"start_frame={start_frame_override} rejected by realism filter: "
                f"reasons={report.reasons}, metrics={report.metrics}"
            )
        sampled.append(
            SampledClipSpec(
                logical_clip_index=logical_idx,
                start_frame=int(start_frame_override),
                seed=_derived_seed(global_seed, logical_idx),
                realism_report=report,
            )
        )
        return sampled

    for i in range(num_clips):
        logical_idx = clip_index_offset + i
        clip_seed = _derived_seed(global_seed, logical_idx)
        rng = np.random.default_rng(clip_seed)
        accepted: Optional[SampledClipSpec] = None
        last_report: Optional[ClipRealismReport] = None

        for _ in range(max_sampling_attempts):
            candidate = int(rng.integers(min_f, last_start + 1))
            if not allow_duplicate_starts and candidate in used_starts:
                continue

            clip = parser.parse_clip(start_frame=candidate, num_frames=num_frames, fps=fps)
            report = evaluate_clip_realism(clip, realism_config)
            last_report = report
            if realism_config.enabled and not report.passed:
                continue

            accepted = SampledClipSpec(
                logical_clip_index=logical_idx,
                start_frame=candidate,
                seed=clip_seed,
                realism_report=report,
            )
            used_starts.add(candidate)
            break

        if accepted is None:
            raise RuntimeError(
                f"Failed to sample a valid clip for logical_clip_index={logical_idx} "
                f"after {max_sampling_attempts} attempts. "
                f"Last realism report: {last_report}"
            )
        sampled.append(accepted)

    return sampled


@dataclass(frozen=True)
class DatasetRegistration:
    name: str
    parser_factory: Callable[[argparse.Namespace], DataParser]
    required_args: tuple[str, ...]


DATASET_ALIASES = {
    "skillcorner": "skillcorner_v2",
}


def _build_metrica_parser(args: argparse.Namespace) -> DataParser:
    return MetricaParser(
        home_csv_path=args.home_csv,
        away_csv_path=args.away_csv,
        pitch_length_m=args.pitch_length_m,
        pitch_width_m=args.pitch_width_m,
    )


def _build_skillcorner_v1_parser(args: argparse.Namespace) -> DataParser:
    return SkillCornerParser(
        tracking_json_path=args.tracking_json,
        pitch_length_m=args.pitch_length_m,
        pitch_width_m=args.pitch_width_m,
    )


def _build_skillcorner_v2_parser(args: argparse.Namespace) -> DataParser:
    return SkillCornerV2Parser(
        tracking_json_path=args.tracking_json,
        match_json_path=args.match_json,
        pitch_length_m=args.pitch_length_m,
        pitch_width_m=args.pitch_width_m,
    )


DATASET_REGISTRY: dict[str, DatasetRegistration] = {
    "metrica": DatasetRegistration(
        name="metrica",
        parser_factory=_build_metrica_parser,
        required_args=MetricaAdapter.required_input_names(),
    ),
    "skillcorner_v1": DatasetRegistration(
        name="skillcorner_v1",
        parser_factory=_build_skillcorner_v1_parser,
        required_args=SkillCornerAdapter.required_input_names(),
    ),
    "skillcorner_v2": DatasetRegistration(
        name="skillcorner_v2",
        parser_factory=_build_skillcorner_v2_parser,
        required_args=SkillCornerV2Adapter.required_input_names(),
    ),
}


def _create_dataset_parser(dataset_name: str, args: argparse.Namespace) -> DataParser:
    dataset = DATASET_ALIASES.get(dataset_name, dataset_name)
    registration = DATASET_REGISTRY.get(dataset)
    if registration is None:
        supported = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset '{dataset_name}'. Supported datasets: {supported}.")

    missing: list[str] = []
    for arg_name in registration.required_args:
        if not getattr(args, arg_name, None):
            missing.append(f"--{arg_name}")
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"--dataset {dataset} requires {missing_text}.")

    return registration.parser_factory(args)


def _resolve_dataset_name(args: argparse.Namespace) -> str:
    if args.dataset:
        return str(args.dataset).strip().lower()
    if args.mode:
        legacy = str(args.mode).strip().lower()
        return DATASET_ALIASES.get(legacy, legacy)
    supported = ", ".join(sorted(DATASET_REGISTRY))
    raise ValueError(f"Missing dataset selection. Use --dataset {{{supported}}}.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse soccer tracking data and export VBVR-style BEV clip artifacts."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=f"Dataset adapter name. Supported: {', '.join(sorted(DATASET_REGISTRY))}.",
    )
    parser.add_argument("--mode", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--home_csv", type=str, default=None, help="Metrica home CSV path.")
    parser.add_argument("--away_csv", type=str, default=None, help="Metrica away CSV path.")
    parser.add_argument(
        "--tracking_json",
        type=str,
        default=None,
        help="SkillCorner tracking path. JSON for v1, JSONL for official open-data v2.",
    )
    parser.add_argument(
        "--match_json",
        type=str,
        default=None,
        help="Optional SkillCorner companion match metadata JSON path for v2. If omitted, inferred from the tracking file name.",
    )

    parser.add_argument("--output_root", type=str, default="output")
    parser.add_argument("--clip_id", type=str, default="soccer_bev_00000000")
    parser.add_argument("--clip_id_prefix", type=str, default="soccer_bev")
    parser.add_argument("--num_clips", type=int, default=1)
    parser.add_argument("--clip_index_offset", type=int, default=0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_frame", type=int, default=None)
    parser.add_argument("--max_sampling_attempts", type=int, default=100)
    parser.add_argument("--allow_duplicate_starts", action="store_true")

    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=520)
    parser.add_argument("--pitch_length_m", type=float, default=105.0)
    parser.add_argument("--pitch_width_m", type=float, default=68.0)
    parser.add_argument("--home_color", type=str, default="#E63946")
    parser.add_argument("--away_color", type=str, default="#457B9D")
    parser.add_argument("--ball_color", type=str, default="#F5F500")
    parser.add_argument("--disable_orientation_normalization", action="store_true")

    parser.add_argument("--disable_realism_filter", action="store_true")
    parser.add_argument("--min_ball_in_bounds_ratio", type=float, default=0.98)
    parser.add_argument("--min_attack_progress_m", type=float, default=3.0)
    parser.add_argument("--support_distance_m", type=float, default=20.0)
    parser.add_argument("--min_support_ratio", type=float, default=0.55)
    parser.add_argument("--min_defense_ahead_ratio", type=float, default=0.0)
    parser.add_argument("--min_majority_possession_ratio", type=float, default=0.50)

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset_name = _resolve_dataset_name(args)
    clip_frames = int(args.fps * args.seconds)
    if clip_frames <= 0:
        raise ValueError("fps * seconds must be > 0.")
    if args.max_sampling_attempts <= 0:
        raise ValueError("--max_sampling_attempts must be > 0.")

    data_parser = _create_dataset_parser(dataset_name, args)
    data_parser.load()
    realism_config = RealismConfig(
        enabled=not args.disable_realism_filter,
        min_ball_in_bounds_ratio=float(args.min_ball_in_bounds_ratio),
        min_attack_progress_m=float(args.min_attack_progress_m),
        support_distance_m=float(args.support_distance_m),
        min_support_ratio=float(args.min_support_ratio),
        min_defense_ahead_ratio=float(args.min_defense_ahead_ratio),
        min_majority_possession_ratio=float(args.min_majority_possession_ratio),
    )
    sampled_specs = sample_clip_specs(
        parser=data_parser,
        num_frames=clip_frames,
        fps=args.fps,
        num_clips=args.num_clips,
        global_seed=args.seed,
        clip_index_offset=args.clip_index_offset,
        start_frame_override=args.start_frame,
        max_sampling_attempts=args.max_sampling_attempts,
        realism_config=realism_config,
        allow_duplicate_starts=args.allow_duplicate_starts,
    )

    renderer = BEVRenderer(
        width=args.width,
        height=args.height,
        home_color=args.home_color,
        away_color=args.away_color,
        ball_color=args.ball_color,
    )
    normalize_orientation = not args.disable_orientation_normalization

    for idx, spec in enumerate(sampled_specs):
        clip = data_parser.parse_clip(start_frame=spec.start_frame, num_frames=clip_frames, fps=args.fps)
        frames = renderer.render_frames(clip, normalize_orientation=normalize_orientation)

        norm_coords, _, _ = normalize_attack_direction(clip.coords_xy, clip)
        analysis = analyze_clip_events(clip, norm_coords)
        prompt_text = generate_clip_prompt(analysis)

        if args.num_clips == 1:
            clip_id = args.clip_id
        else:
            clip_id = f"{args.clip_id_prefix}_{spec.logical_clip_index:08d}"

        out_dir = Path(args.output_root) / clip_id
        export_vbvr_clip(frames, out_dir, fps=args.fps)
        export_prompt_txt(prompt_text, out_dir)

        print(
            f"[{idx + 1}/{len(sampled_specs)}] Export complete: dataset={dataset_name}, "
            f"clip_id={clip_id}, start_frame={spec.start_frame}, clip_seed={spec.seed}, "
            f"frames={clip_frames}, agents={len(clip.agent_ids)}, output={out_dir}, "
            f"realism_passed={spec.realism_report.passed}, "
            f"realism_reasons={spec.realism_report.reasons}, "
            f"realism_metrics={spec.realism_report.metrics}"
        )


if __name__ == "__main__":
    main()
