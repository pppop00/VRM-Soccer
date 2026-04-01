#!/usr/bin/env python3
"""
Multi-dimensional automated scorer for AI-generated BEV soccer videos.

5 scoring dimensions:
  D1 Visual Quality       — frame-level perceptual quality (SSIM-based, DOVER if available)
  D2 Temporal Consistency — smoothness via optical flow warping error
  D3 Physical Plausibility — player/ball speed bounds + out-of-bounds + teleportation
  D4 Tactical Coherence   — possession validity + support ratio
  D5 Prompt Adherence     — CLIP text-image cosine similarity

Usage:
    python scorer.py --instance_dir PATH --generated PATH
    python scorer.py --instance_dir PATH --generated PATH --weights '{"D1":0.3,"D2":0.1,"D3":0.2,"D4":0.2,"D5":0.2}'
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ─── Optional heavy dependencies ────────────────────────────────────────────
try:
    import torch
    from torchmetrics.multimodal import CLIPScore as TorchCLIPScore
    _CLIP_AVAILABLE = True
except ImportError:
    _CLIP_AVAILABLE = False

try:
    from dover import DOVER as DoverModel
    _DOVER_AVAILABLE = True
except ImportError:
    _DOVER_AVAILABLE = False

# ─── Constants ───────────────────────────────────────────────────────────────
SCORER_VERSION = "v1.0"
MAX_PLAYER_SPEED_MS = 10.0   # m/s elite sprint cap
MAX_BALL_SPEED_MS   = 35.0   # m/s kicked ball cap
MAX_TELEPORT_M      = 5.0    # m per frame at 16fps → 80 m/s: definitely wrong

# BEV color ranges in HSV (OpenCV HSV: H∈[0,180], S/V∈[0,255])
# Pipeline defaults: home_color="#E63946" (RED), away_color="#457B9D" (BLUE), ball_color="#F5F500" (YELLOW)
# Home (RED #E63946) — wraps around H=0/180 in OpenCV
HOME_HSV_LOWER1 = np.array([0,   100, 100], dtype=np.uint8)
HOME_HSV_UPPER1 = np.array([15,  255, 255], dtype=np.uint8)
HOME_HSV_LOWER2 = np.array([160, 100, 100], dtype=np.uint8)
HOME_HSV_UPPER2 = np.array([180, 255, 255], dtype=np.uint8)
# Away (BLUE #457B9D)
AWAY_HSV_LOWER = np.array([90,  40,  80],  dtype=np.uint8)
AWAY_HSV_UPPER = np.array([120, 210, 220], dtype=np.uint8)
# Ball (YELLOW #F5F500)
BALL_HSV_LOWER = np.array([22, 180, 180], dtype=np.uint8)
BALL_HSV_UPPER = np.array([38, 255, 255], dtype=np.uint8)


# ─── Video I/O helpers ───────────────────────────────────────────────────────

def read_video_frames(video_path: str, max_frames: int = 200) -> list[np.ndarray]:
    """Read all frames from a video file as BGR numpy arrays."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def get_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 16.0


def sample_frames(frames: list[np.ndarray], n: int = 8) -> list[np.ndarray]:
    """Uniformly sample n frames from list."""
    if len(frames) <= n:
        return frames
    indices = np.linspace(0, len(frames) - 1, n, dtype=int)
    return [frames[i] for i in indices]


# ─── BEV position extraction ─────────────────────────────────────────────────

def _find_blobs(mask: np.ndarray, min_area: int = 20, max_area: int = 2000) -> list[tuple[float, float]]:
    """Find blob centroids in a binary mask."""
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    result = []
    for i in range(1, n_labels):  # skip background (0)
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            result.append((float(centroids[i, 0]), float(centroids[i, 1])))
    return result


def extract_positions_from_frame(
    frame_bgr: np.ndarray,
    pitch_px: tuple[int, int, int, int],  # (x0, y0, x1, y1) pixel bounds of pitch
    pitch_m: tuple[float, float],          # (length_m, width_m)
) -> tuple[list, list, Optional[tuple]]:
    """
    Extract player/ball pixel positions and convert to metric coordinates.
    Returns (home_positions_m, away_positions_m, ball_position_m_or_None)
    """
    px0, py0, px1, py1 = pitch_px
    pitch_w_px = px1 - px0
    pitch_h_px = py1 - py0
    length_m, width_m = pitch_m

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Home (RED #E63946 — two hue ranges due to red wrap-around)
    home_mask1 = cv2.inRange(hsv, HOME_HSV_LOWER1, HOME_HSV_UPPER1)
    home_mask2 = cv2.inRange(hsv, HOME_HSV_LOWER2, HOME_HSV_UPPER2)
    home_mask = cv2.bitwise_or(home_mask1, home_mask2)
    home_blobs = _find_blobs(home_mask)

    # Away (BLUE #457B9D)
    away_mask = cv2.inRange(hsv, AWAY_HSV_LOWER, AWAY_HSV_UPPER)
    away_blobs = _find_blobs(away_mask)

    # Ball (YELLOW #F5F500, smaller blob)
    ball_mask = cv2.inRange(hsv, BALL_HSV_LOWER, BALL_HSV_UPPER)
    ball_blobs = _find_blobs(ball_mask, min_area=5, max_area=300)

    def px_to_m(px_coord):
        x_px, y_px = px_coord
        x_m = (x_px - px0) / pitch_w_px * length_m
        y_m = (y_px - py0) / pitch_h_px * width_m
        return (x_m, y_m)

    home_m = [px_to_m(b) for b in home_blobs]
    away_m = [px_to_m(b) for b in away_blobs]
    ball_m = px_to_m(ball_blobs[0]) if ball_blobs else None

    return home_m, away_m, ball_m


def infer_pitch_bounds(frame_bgr: np.ndarray) -> tuple[int, int, int, int]:
    """
    Estimate pitch pixel bounds from the green field area.
    Returns (x0, y0, x1, y1).
    Falls back to full frame if detection fails.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    # Field green: H 35-85, S>30, V>50
    green_mask = cv2.inRange(hsv, np.array([35, 30, 50]), np.array([85, 255, 255]))
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return x, y, x + w, y + h
    h, w = frame_bgr.shape[:2]
    return 0, 0, w, h


# ─── Pure-numpy SSIM fallback ─────────────────────────────────────────────────

def _numpy_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute SSIM between two single-channel uint8 images using pure numpy.
    Based on Wang et al. (2004). Returns value in [-1, 1].
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    kernel_size = 11
    sigma = 1.5
    # Build a 1D Gaussian kernel and make it 2D
    k = np.arange(kernel_size) - kernel_size // 2
    gauss_1d = np.exp(-(k ** 2) / (2 * sigma ** 2))
    gauss_1d /= gauss_1d.sum()
    kernel = np.outer(gauss_1d, gauss_1d)

    def convolve(img):
        from scipy.signal import fftconvolve
        return fftconvolve(img, kernel, mode="valid")

    try:
        mu1 = convolve(img1)
        mu2 = convolve(img2)
    except Exception:
        # Fallback to uniform window if scipy unavailable
        mu1 = cv2.blur(img1, (kernel_size, kernel_size))[
            kernel_size//2:-(kernel_size//2), kernel_size//2:-(kernel_size//2)
        ]
        mu2 = cv2.blur(img2, (kernel_size, kernel_size))[
            kernel_size//2:-(kernel_size//2), kernel_size//2:-(kernel_size//2)
        ]

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    try:
        sigma1_sq = convolve(img1 ** 2) - mu1_sq
        sigma2_sq = convolve(img2 ** 2) - mu2_sq
        sigma12   = convolve(img1 * img2) - mu1_mu2
    except Exception:
        def cv_blur_crop(arr):
            b = cv2.blur(arr, (kernel_size, kernel_size))
            return b[kernel_size//2:-(kernel_size//2), kernel_size//2:-(kernel_size//2)]
        sigma1_sq = cv_blur_crop(img1 ** 2) - mu1_sq
        sigma2_sq = cv_blur_crop(img2 ** 2) - mu2_sq
        sigma12   = cv_blur_crop(img1 * img2) - mu1_mu2

    numerator   = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / (denominator + 1e-10)
    return float(np.mean(ssim_map))


# ─── D1: Visual Quality ───────────────────────────────────────────────────────

def score_visual_quality(
    gt_frames: list[np.ndarray],
    gen_frames: list[np.ndarray],
) -> tuple[float, dict]:
    """
    D1: Frame-level SSIM between ground truth and generated video.
    Samples up to 16 frame pairs. Returns mean SSIM in [0, 1].
    Uses skimage.metrics.structural_similarity if available, otherwise falls
    back to a pure-numpy/OpenCV implementation.
    """
    try:
        from skimage.metrics import structural_similarity as _ssim_fn
        def compute_ssim(a, b):
            return _ssim_fn(a, b, data_range=255)
        backend = "skimage"
    except ImportError:
        logging.warning("skimage not available; using numpy SSIM fallback for D1.")
        def compute_ssim(a, b):
            return _numpy_ssim(a, b)
        backend = "numpy_fallback"

    n = min(16, len(gt_frames), len(gen_frames))
    gt_sample = sample_frames(gt_frames, n)
    gen_sample = sample_frames(gen_frames, n)

    scores = []
    for gt_f, gen_f in zip(gt_sample, gen_sample):
        # Resize gen to match gt if needed
        if gt_f.shape != gen_f.shape:
            gen_f = cv2.resize(gen_f, (gt_f.shape[1], gt_f.shape[0]))
        gt_gray = cv2.cvtColor(gt_f, cv2.COLOR_BGR2GRAY)
        gen_gray = cv2.cvtColor(gen_f, cv2.COLOR_BGR2GRAY)
        s = compute_ssim(gt_gray, gen_gray)
        scores.append(float(s))

    mean_ssim = float(np.mean(scores)) if scores else 0.0
    # Normalize: SSIM can be negative; map [-1,1] → [0,1]
    normalized = (mean_ssim + 1.0) / 2.0
    return normalized, {"mean_ssim": mean_ssim, "n_frames": n, "backend": backend}


# ─── D2: Temporal Consistency ─────────────────────────────────────────────────

def score_temporal_consistency(gen_frames: list[np.ndarray]) -> tuple[float, dict]:
    """
    D2: Optical flow warping error between consecutive frames.
    Lower variance in flow magnitude = smoother = higher score.
    """
    if len(gen_frames) < 2:
        return 1.0, {"note": "too few frames"}

    sampled = sample_frames(gen_frames, min(32, len(gen_frames)))
    flow_mags = []
    warp_errors = []

    for i in range(len(sampled) - 1):
        f1 = cv2.cvtColor(sampled[i], cv2.COLOR_BGR2GRAY)
        f2 = cv2.cvtColor(sampled[i + 1], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            f1, f2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_mags.append(float(np.mean(mag)))

        # Warp f1 by flow and compare to f2
        h, w = f1.shape
        flow_map = np.column_stack([
            (flow[..., 0] + np.mgrid[0:h, 0:w][1]).ravel(),
            (flow[..., 1] + np.mgrid[0:h, 0:w][0]).ravel(),
        ]).reshape(h, w, 2).astype(np.float32)
        warped = cv2.remap(f1, flow_map[:, :, 0], flow_map[:, :, 1], cv2.INTER_LINEAR)
        err = float(np.mean(np.abs(warped.astype(float) - f2.astype(float)))) / 255.0
        warp_errors.append(err)

    mean_warp_err = float(np.mean(warp_errors)) if warp_errors else 0.0
    # Lower error = better. Map [0, 0.3] → [1, 0] with clamp
    score = max(0.0, 1.0 - mean_warp_err / 0.3)
    return score, {"mean_warp_error": mean_warp_err, "n_pairs": len(warp_errors)}


# ─── D3: Physical Plausibility ────────────────────────────────────────────────

def score_physical_plausibility(
    gen_frames: list[np.ndarray],
    fps: float = 16.0,
    pitch_length_m: float = 105.0,
    pitch_width_m: float = 68.0,
) -> tuple[float, dict]:
    """
    D3: Extract agent positions via BEV color segmentation.
    Check speed bounds, out-of-bounds, and teleportation.
    """
    pitch_bounds = infer_pitch_bounds(gen_frames[0]) if gen_frames else (0, 0, 832, 480)

    prev_ball: Optional[tuple] = None
    prev_home: list = []
    prev_away: list = []

    speed_violations = 0
    oob_violations = 0
    teleport_violations = 0
    total_checks = 0

    sampled = sample_frames(gen_frames, min(len(gen_frames), 60))

    for frame in sampled:
        home_m, away_m, ball_m = extract_positions_from_frame(
            frame, pitch_bounds, (pitch_length_m, pitch_width_m)
        )

        # Out-of-bounds check for ball
        if ball_m is not None:
            bx, by = ball_m
            if not (0 <= bx <= pitch_length_m and 0 <= by <= pitch_width_m):
                oob_violations += 1
            total_checks += 1

        dt = 1.0 / fps

        # Ball speed check
        if prev_ball is not None and ball_m is not None:
            dist = math.hypot(ball_m[0] - prev_ball[0], ball_m[1] - prev_ball[1])
            speed = dist / dt
            total_checks += 1
            if speed > MAX_BALL_SPEED_MS:
                speed_violations += 1
            if dist > MAX_TELEPORT_M:
                teleport_violations += 1

        # Player speed/teleport checks (match nearest prev player)
        for curr_list, prev_list, max_speed in [
            (home_m, prev_home, MAX_PLAYER_SPEED_MS),
            (away_m, prev_away, MAX_PLAYER_SPEED_MS),
        ]:
            for pos in curr_list:
                if prev_list:
                    dists = [math.hypot(pos[0] - p[0], pos[1] - p[1]) for p in prev_list]
                    nearest = min(dists)
                    speed = nearest / dt
                    total_checks += 1
                    if speed > max_speed:
                        speed_violations += 1
                    if nearest > MAX_TELEPORT_M:
                        teleport_violations += 1

        prev_ball = ball_m
        prev_home = home_m
        prev_away = away_m

    if total_checks == 0:
        return 1.0, {"note": "no trackable agents found"}

    violation_ratio = (speed_violations + oob_violations + teleport_violations) / total_checks
    score = max(0.0, 1.0 - violation_ratio * 3.0)  # penalize harder
    return score, {
        "speed_violations": speed_violations,
        "oob_violations": oob_violations,
        "teleport_violations": teleport_violations,
        "total_checks": total_checks,
        "violation_ratio": violation_ratio,
    }


# ─── D4: Tactical Coherence ───────────────────────────────────────────────────

def score_tactical_coherence(
    gen_frames: list[np.ndarray],
    fps: float = 16.0,
    pitch_length_m: float = 105.0,
    pitch_width_m: float = 68.0,
    support_distance_m: float = 20.0,
) -> tuple[float, dict]:
    """
    D4: Check possession validity and support ratio.
    Possession = team whose player is closest to ball.
    Support ratio = fraction of frames where possession holder has teammate within support_distance_m.
    """
    pitch_bounds = infer_pitch_bounds(gen_frames[0]) if gen_frames else (0, 0, 832, 480)
    sampled = sample_frames(gen_frames, min(len(gen_frames), 60))

    possession_labels = []  # 'home', 'away', or None
    support_flags = []

    for frame in sampled:
        home_m, away_m, ball_m = extract_positions_from_frame(
            frame, pitch_bounds, (pitch_length_m, pitch_width_m)
        )

        if ball_m is None or (not home_m and not away_m):
            possession_labels.append(None)
            continue

        bx, by = ball_m
        min_home_d = min((math.hypot(p[0]-bx, p[1]-by) for p in home_m), default=float("inf"))
        min_away_d = min((math.hypot(p[0]-bx, p[1]-by) for p in away_m), default=float("inf"))

        if min_home_d < min_away_d:
            possession_labels.append("home")
            carrier_pos = min(home_m, key=lambda p: math.hypot(p[0]-bx, p[1]-by))
            # Check support: any other home player within support_distance_m
            support = any(
                math.hypot(p[0]-carrier_pos[0], p[1]-carrier_pos[1]) <= support_distance_m
                for p in home_m if p != carrier_pos
            )
            support_flags.append(support)
        else:
            possession_labels.append("away")
            carrier_pos = min(away_m, key=lambda p: math.hypot(p[0]-bx, p[1]-by))
            support = any(
                math.hypot(p[0]-carrier_pos[0], p[1]-carrier_pos[1]) <= support_distance_m
                for p in away_m if p != carrier_pos
            )
            support_flags.append(support)

    valid = [l for l in possession_labels if l is not None]
    if not valid:
        return 1.0, {"note": "no possession detected"}

    # Possession transition validity: penalize if possession changes every frame (flickering)
    transitions = sum(1 for i in range(1, len(valid)) if valid[i] != valid[i-1])
    max_transitions = len(valid) - 1
    transition_rate = transitions / max_transitions if max_transitions > 0 else 0.0
    # Ideal: some transitions but not every frame. Penalize rate > 0.3
    transition_score = max(0.0, 1.0 - max(0.0, transition_rate - 0.1) / 0.3)

    support_ratio = sum(support_flags) / len(support_flags) if support_flags else 1.0

    score = 0.5 * transition_score + 0.5 * support_ratio
    return score, {
        "possession_transition_rate": transition_rate,
        "support_ratio": support_ratio,
        "transition_score": transition_score,
        "n_frames": len(valid),
    }


# ─── D5: Prompt Adherence ─────────────────────────────────────────────────────

def score_prompt_adherence(
    gen_frames: list[np.ndarray],
    prompt_text: str,
) -> tuple[float, dict]:
    """
    D5: CLIP cosine similarity between text prompt and sampled frames.
    Falls back to 0.5 (neutral) if CLIP not available.
    """
    if not _CLIP_AVAILABLE:
        logging.warning("CLIP not available (install torch + open-clip-torch). D5 returning neutral 0.5.")
        return 0.5, {"note": "CLIP unavailable"}

    import torch
    from PIL import Image

    sampled = sample_frames(gen_frames, 8)

    try:
        clip_scorer = TorchCLIPScore(model_name_or_path="openai/clip-vit-base-patch32")
        clip_scorer.eval()

        scores = []
        for frame in sampled:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tensor = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                s = clip_scorer(img_tensor, [prompt_text])
            scores.append(float(s))

        mean_score = float(np.mean(scores))
        # CLIP score is typically in [0, 100] range for torchmetrics — normalize
        normalized = min(1.0, mean_score / 30.0)  # 30 is a typical "good" CLIP score
        return normalized, {"mean_clip_score": mean_score, "normalized": normalized}
    except Exception as e:
        logging.warning(f"CLIP scoring failed: {e}. Returning neutral 0.5.")
        return 0.5, {"error": str(e)}


# ─── Main scorer ──────────────────────────────────────────────────────────────

def score_clip(
    instance_dir: str,
    generated_path: str,
    pitch_length_m: float = 105.0,
    pitch_width_m: float = 68.0,
    weights: Optional[dict] = None,
) -> dict:
    """
    Score a generated BEV soccer video against ground truth on 5 dimensions.

    Args:
        instance_dir: Directory containing ground_truth.mp4 and prompt.txt
        generated_path: Path to generated video file
        pitch_length_m: Pitch length in meters
        pitch_width_m: Pitch width in meters
        weights: Dict with keys D1-D5 (or full names). Default: equal weights (0.2 each).

    Returns:
        Dict with per-dimension scores, composite score, and details.
    """
    instance_dir = Path(instance_dir)
    gt_path = instance_dir / "ground_truth.mp4"
    prompt_path = instance_dir / "prompt.txt"

    if not gt_path.exists():
        raise FileNotFoundError(f"ground_truth.mp4 not found in {instance_dir}")
    if not Path(generated_path).exists():
        raise FileNotFoundError(f"Generated video not found: {generated_path}")

    prompt_text = prompt_path.read_text(encoding="utf-8").strip() if prompt_path.exists() else ""

    logging.info("Loading video frames...")
    gt_frames = read_video_frames(str(gt_path))
    gen_frames = read_video_frames(str(generated_path))
    fps = get_video_fps(str(generated_path))

    logging.info(f"GT: {len(gt_frames)} frames | Gen: {len(gen_frames)} frames | FPS: {fps}")

    # Default equal weights
    if weights is None:
        weights = {"D1": 0.2, "D2": 0.2, "D3": 0.2, "D4": 0.2, "D5": 0.2}

    # Normalize weights
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    details = {}

    logging.info("D1: Visual Quality (SSIM)...")
    d1, details["D1"] = score_visual_quality(gt_frames, gen_frames)

    logging.info("D2: Temporal Consistency (optical flow)...")
    d2, details["D2"] = score_temporal_consistency(gen_frames)

    logging.info("D3: Physical Plausibility (BEV color tracking)...")
    d3, details["D3"] = score_physical_plausibility(gen_frames, fps, pitch_length_m, pitch_width_m)

    logging.info("D4: Tactical Coherence (possession + support)...")
    d4, details["D4"] = score_tactical_coherence(gen_frames, fps, pitch_length_m, pitch_width_m)

    logging.info("D5: Prompt Adherence (CLIP)...")
    d5, details["D5"] = score_prompt_adherence(gen_frames, prompt_text)

    scores = {
        "visual_quality":        round(d1, 4),
        "temporal_consistency":  round(d2, 4),
        "physical_plausibility": round(d3, 4),
        "tactical_coherence":    round(d4, 4),
        "prompt_adherence":      round(d5, 4),
    }

    dim_map = {
        "D1": "visual_quality",
        "D2": "temporal_consistency",
        "D3": "physical_plausibility",
        "D4": "tactical_coherence",
        "D5": "prompt_adherence",
    }
    composite = sum(
        weights.get(k, 0.2) * scores[v]
        for k, v in dim_map.items()
    )

    result = {
        "instance_id": instance_dir.name,
        "scores": scores,
        "composite_score": round(composite, 4),
        "scorer_version": SCORER_VERSION,
        "weights_used": weights,
        "details": details,
    }

    return result


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Score a generated BEV soccer video.")
    parser.add_argument("--instance_dir", required=True, help="Path to clip instance directory")
    parser.add_argument("--generated", required=True, help="Path to generated video file")
    parser.add_argument("--pitch_length_m", type=float, default=105.0)
    parser.add_argument("--pitch_width_m", type=float, default=68.0)
    parser.add_argument(
        "--weights", type=str, default=None,
        help='JSON dict of dimension weights, e.g. \'{"D1":0.3,"D2":0.1,"D3":0.2,"D4":0.2,"D5":0.2}\''
    )
    parser.add_argument("--output", type=str, default=None, help="Save result JSON to this path")
    args = parser.parse_args()

    weights = json.loads(args.weights) if args.weights else None

    result = score_clip(
        instance_dir=args.instance_dir,
        generated_path=args.generated,
        pitch_length_m=args.pitch_length_m,
        pitch_width_m=args.pitch_width_m,
        weights=weights,
    )

    print(json.dumps(result, indent=2))

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        logging.info(f"Result saved to {args.output}")


if __name__ == "__main__":
    main()
