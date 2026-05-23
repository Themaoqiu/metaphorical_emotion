"""
Reward function for Spatio-Temporal Video Grounding (STVG).

Rewards:
  - format:   1.0 if response parses to a valid non-empty track dict, else 0.0
  - tiou:     mean temporal IoU across matched tracks
  - viou:     mean vIoU across matched tracks (spatial+temporal joint, DORO eval primary metric)
  - overall:  viou if format==1, else 0.0

Matching uses the same exact-optimal bitmask-DP assignment as DORO-STVG eval,
with vIoU as the matching criterion and divisor = max(num_gt, num_pred).

Ground truth is the raw assistant JSON string from the training data, e.g.:
  '{"the cat": {"0": [0.1, 0.2, 0.3, 0.4], "1": [0.1, 0.2, 0.3, 0.4]}}'
"""

import json
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

REWARD_NAME = "stvg"
REWARD_TYPE = "sequential"


# ---------------------------------------------------------------------------
# Geometry helpers — exact mirrors of DORO-STVG eval/utils/metrics.py
# ---------------------------------------------------------------------------

def _compute_tiou(gt_span: Optional[Tuple[int, int]], pred_span: Optional[Tuple[int, int]]) -> float:
    if gt_span is None or pred_span is None:
        return 0.0
    inter_start = max(gt_span[0], pred_span[0])
    inter_end = min(gt_span[1], pred_span[1])
    if inter_end < inter_start:
        return 0.0
    intersection = inter_end - inter_start + 1
    union = max(gt_span[1], pred_span[1]) - min(gt_span[0], pred_span[0]) + 1
    return intersection / union if union > 0 else 0.0


def _compute_siou(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _compute_viou(gt_bboxes: Dict[int, List[float]], pred_bboxes: Dict[int, List[float]]) -> float:
    """vIoU: mean box-IoU over the UNION of GT and pred frame sets (DORO eval primary metric)."""
    union_frames = set(gt_bboxes) | set(pred_bboxes)
    if not union_frames:
        return 0.0
    total = sum(
        _compute_siou(gt_bboxes.get(f, [0.0, 0.0, 0.0, 0.0]),
                      pred_bboxes.get(f, [0.0, 0.0, 0.0, 0.0]))
        for f in union_frames
    )
    return total / len(union_frames)


# ---------------------------------------------------------------------------
# Exact-optimal assignment — bitmask DP, identical to DORO-STVG eval
# ---------------------------------------------------------------------------

def _optimal_assignment(score_matrix: List[List[float]]) -> List[Tuple[int, int]]:
    num_rows = len(score_matrix)
    num_cols = len(score_matrix[0]) if score_matrix else 0
    if num_rows == 0 or num_cols == 0:
        return []

    size = max(num_rows, num_cols)
    padded = [[0.0] * size for _ in range(size)]
    for r in range(num_rows):
        for c in range(num_cols):
            padded[r][c] = score_matrix[r][c]

    @lru_cache(maxsize=None)
    def solve(row: int, used_mask: int) -> Tuple[float, Tuple[int, ...]]:
        if row == size:
            return 0.0, ()
        best_score, best_assign = -1.0, ()
        for col in range(size):
            if used_mask & (1 << col):
                continue
            next_score, next_assign = solve(row + 1, used_mask | (1 << col))
            total = padded[row][col] + next_score
            if total > best_score:
                best_score, best_assign = total, (col,) + next_assign
        return best_score, best_assign

    _, assignment = solve(0, 0)
    solve.cache_clear()

    pairs = []
    for row, col in enumerate(assignment):
        if row < num_rows and col < num_cols:
            pairs.append((row, col))
    return pairs


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Optional[str]:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    return text[start:end + 1]


def _parse_track_json(text: str) -> Optional[Dict[str, Any]]:
    candidate = _extract_json(text)
    if not candidate:
        return None
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _payload_to_tracks(payload: Dict[str, Any]) -> List[Dict]:
    tracks = []
    for description, frame_map in payload.items():
        if not isinstance(frame_map, dict):
            continue
        bboxes: Dict[int, List[float]] = {}
        for k, v in frame_map.items():
            try:
                fidx = int(str(k).strip())
            except ValueError:
                continue
            if not isinstance(v, list) or len(v) != 4:
                continue
            try:
                coords = [max(0.0, min(1.0, float(c))) for c in v]
            except (TypeError, ValueError):
                continue
            bboxes[fidx] = coords
        if not bboxes:
            continue
        frames = sorted(bboxes)
        tracks.append({
            "description": str(description).strip(),
            "temporal_span": (frames[0], frames[-1]),
            "spatial_bboxes": bboxes,
        })
    return tracks


# ---------------------------------------------------------------------------
# Multi-target metrics — exact match of DORO-STVG compute_multi_target_metrics
# ---------------------------------------------------------------------------

def _compute_multi_target_metrics(
    gt_tracks: List[Dict],
    pred_tracks: List[Dict],
) -> Tuple[float, float]:
    """Returns (mean_tIoU, mean_vIoU) with divisor = max(num_gt, num_pred)."""
    num_gt, num_pred = len(gt_tracks), len(pred_tracks)
    if num_gt == 0 and num_pred == 0:
        return 0.0, 0.0
    if num_gt == 0 or num_pred == 0:
        return 0.0, 0.0

    score_matrix = [
        [_compute_viou(gt["spatial_bboxes"], pred["spatial_bboxes"])
         for pred in pred_tracks]
        for gt in gt_tracks
    ]

    matches = _optimal_assignment(score_matrix)
    divisor = max(num_gt, num_pred)

    tiou_sum = viou_sum = 0.0
    for gt_idx, pred_idx in matches:
        tiou_sum += _compute_tiou(gt_tracks[gt_idx]["temporal_span"],
                                  pred_tracks[pred_idx]["temporal_span"])
        viou_sum += score_matrix[gt_idx][pred_idx]

    return tiou_sum / divisor, viou_sum / divisor


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_score(reward_input: Dict[str, Any]) -> Dict[str, float]:
    response: str = reward_input["response"]
    ground_truth: str = reward_input["ground_truth"]

    pred_payload = _parse_track_json(response)
    pred_tracks = _payload_to_tracks(pred_payload) if pred_payload else []
    fmt = 1.0 if pred_tracks else 0.0

    if fmt == 0.0:
        return {"overall": 0.0, "format": 0.0, "tiou": 0.0, "viou": 0.0}

    gt_payload = _parse_track_json(ground_truth)
    gt_tracks = _payload_to_tracks(gt_payload) if gt_payload else []

    tiou, viou = _compute_multi_target_metrics(gt_tracks, pred_tracks)

    return {"overall": viou, "format": fmt, "tiou": tiou, "viou": viou}
