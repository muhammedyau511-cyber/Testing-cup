import os
import zipfile

os.makedirs("temp/cup_state_refactored", exist_ok=True)
os.makedirs("generated", exist_ok=True)

utils_code = '''"""Shared configuration, video I/O helpers, and board-state checks."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json
import cv2
import numpy as np

DEFAULT_SPOT_X = (0.29, 0.51, 0.77)

@dataclass(frozen=True)
class DetectorConfig:
    spot_x: tuple[float, float, float] = DEFAULT_SPOT_X
    roi_x_range: tuple[float, float] = (0.08, 0.92)
    roi_y_range: tuple[float, float] = (0.36, 0.55)
    ball_hsv_low: tuple[int, int, int] = (0, 0, 155)
    ball_hsv_high: tuple[int, int, int] = (180, 105, 255)
    strict_ball_hsv_low: tuple[int, int, int] = (0, 0, 175)
    strict_ball_hsv_high: tuple[int, int, int] = (180, 85, 255)
    min_cups_for_board: int = 2
    min_cup_height_ratio: float = 0.08


def open_video_io(video_path: str, output_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {output_path}")
    return cap, writer, fps, width, height


def is_full_game_board(
    boxes: list[tuple[int, int, int, int]],
    frame_height: int,
    cfg: DetectorConfig = DetectorConfig(),
) -> bool:
    if len(boxes) < cfg.min_cups_for_board:
        return False
    median_height = float(np.median([b[3] for b in boxes]))
    return median_height >= cfg.min_cup_height_ratio * frame_height


def spot_from_x(
    ball_x: int,
    frame_width: int,
    spot_x: tuple[float, float, float] = DEFAULT_SPOT_X,
) -> int:
    normalized_x = ball_x / max(1, frame_width)
    return min(range(3), key=lambda i: abs(normalized_x - spot_x[i])) + 1


def save_json_report(report_path: str | Path, payload: dict) -> None:
    Path(report_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
'''

main_code = '''"""Visual shell-game observer with hysteresis ball tracking.

Reports a spot only when the ball is visible on the active game board.
Deliberately reports UNKNOWN during complete occlusion or non-game screens.
"""
from __future__ import annotations
import argparse
import csv
import itertools
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np
from utils import (
    DEFAULT_SPOT_X,
    DetectorConfig,
    is_full_game_board,
    open_video_io,
    spot_from_x,
)


@dataclass
class Cup:
    cup_id: int
    box: tuple[int, int, int, int]
    center: tuple[float, float]


def detect_cups(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, np.array([0, 65, 55]), np.array([14, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([160, 65, 55]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(red1, red2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = frame.shape[:2]
    found = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (0.0004 * w * h < area < 0.08 * w * h):
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bh < 1.05 * bw or bh < 24:
            continue
        found.append((x, y, bw, bh))
    found.sort(key=lambda b: b[2] * b[3], reverse=True)
    unique = []
    for box in found:
        cx = box[0] + box[2] / 2
        if all(abs(cx - (b[0] + b[2] / 2)) > 0.10 * w for b in unique):
            unique.append(box)
    return sorted(unique[:3], key=lambda b: b[0])


def update_cup_tracks(
    previous: dict[int, Cup], boxes: list[tuple[int, int, int, int]]
) -> dict[int, Cup]:
    if not previous:
        return {
            i: Cup(i, b, (b[0] + b[2] / 2, b[1] + b[3] / 2))
            for i, b in enumerate(boxes)
        }
    ids = sorted(previous)
    if len(boxes) < len(ids):
        return previous.copy()
    best, best_cost = None, float("inf")
    for perm in itertools.permutations(boxes, len(ids)):
        cost = 0.0
        for cup_id, box in zip(ids, perm):
            center = (box[0] + box[2] / 2, box[1] + box[3] / 2)
            cost += np.hypot(
                center[0] - previous[cup_id].center[0],
                center[1] - previous[cup_id].center[1],
            )
        if cost < best_cost:
            best, best_cost = perm, cost
    return {
        cup_id: Cup(cup_id, b, (b[0] + b[2] / 2, b[1] + b[3] / 2))
        for cup_id, b in zip(ids, best)
    }


def detect_ball(
    frame: np.ndarray,
    prev_xy: Optional[tuple[int, int]] = None,
    cfg: DetectorConfig = DetectorConfig(),
) -> Optional[tuple[int, int, int, float]]:
    """Detect the bright white ball on the wooden board using hysteresis."""
    h, w = frame.shape[:2]
    x1, x2 = int(cfg.roi_x_range[0] * w), int(cfg.roi_x_range[1] * w)
    y1, y2 = int(0.37 * h), int(0.52 * h)
    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Relax HSV slightly when continuing an active track near prev_xy
    if prev_xy is not None:
        mask = cv2.inRange(hsv, np.array(cfg.ball_hsv_low), np.array(cfg.ball_hsv_high))
    else:
        mask = cv2.inRange(
            hsv, np.array(cfg.strict_ball_hsv_low), np.array(cfg.strict_ball_hsv_high)
        )

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    scale = (w * h) / (720 * 1280)
    min_area, max_area = 200 * scale, 9000 * scale
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (min_area <= area <= max_area):
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        abs_x, abs_y = int(cx + x1), int(cy + y1)

        near_prev = (
            prev_xy is not None
            and np.hypot(abs_x - prev_xy[0], abs_y - prev_xy[1]) < 0.08 * w
        )
        min_circ = 0.48 if near_prev else 0.62
        if circularity < min_circ or radius < 8 * np.sqrt(scale):
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        ratio = bw / max(1, bh)
        if not (0.68 <= ratio <= 1.35):
            continue
        candidates.append((circularity, (abs_x, abs_y, int(radius), float(circularity))))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def run(
    video: str,
    output: str,
    log: str,
    debug: bool,
    spot_x: tuple[float, float, float],
) -> None:
    cap, writer, fps, width, height = open_video_io(video, output)
    tracks: dict[int, Cup] = {}
    rows = []
    frame_no = 0
    last_visible_spot: Optional[int] = None
    prev_ball_xy: Optional[tuple[int, int]] = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes = detect_cups(frame)
        if boxes:
            tracks = update_cup_tracks(tracks, boxes)

        full_board = is_full_game_board(boxes, height)
        ball = detect_ball(frame, prev_xy=prev_ball_xy) if full_board else None
        timestamp = frame_no / fps
        spot = None
        confidence = 0.0

        if ball:
            bx, by, radius, confidence = ball
            prev_ball_xy = (bx, by)
            spot = spot_from_x(bx, width, spot_x)
            last_visible_spot = spot
            cv2.circle(frame, (bx, by), radius, (0, 255, 0), 3)
            cv2.putText(
                frame,
                f"BALL VISIBLE - SPOT {spot}",
                (int(0.028 * width), int(0.043 * height)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            prev_ball_xy = None
            cv2.putText(
                frame,
                "BALL HIDDEN - LOCATION UNKNOWN",
                (int(0.028 * width), int(0.043 * height)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

        if debug:
            for cup_id, cup in tracks.items():
                x, y, bw, bh = cup.box
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 180, 0), 2)
                cv2.putText(
                    frame,
                    f"CUP {cup_id + 1}",
                    (x, max(20, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 180, 0),
                    1,
                )

        rows.append(
            [
                frame_no,
                f"{timestamp:.4f}",
                int(ball is not None),
                spot if spot is not None else "UNKNOWN",
                f"{confidence:.3f}",
            ]
        )
        writer.write(frame)
        frame_no += 1

    cap.release()
    writer.release()

    with open(log, "w", newline="", encoding="utf-8") as f:
        out = csv.writer(f)
        out.writerow(
            ["frame", "time_seconds", "ball_visible", "observed_spot", "circularity_confidence"]
        )
        out.writerows(rows)
        out.writerow([])
        out.writerow(
            ["last_observed_spot", last_visible_spot if last_visible_spot is not None else "UNKNOWN"]
        )
        out.writerow(
            ["warning", "Last observed spot is not a prediction during hidden shuffle."]
        )

    print(f"Frames processed: {frame_no}")
    print(f"Video saved: {output}")
    print(f"Frame log saved: {log}")
    print(f"Last observed spot: {last_visible_spot if last_visible_spot is not None else 'UNKNOWN'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Shell-game visual observer (reports UNKNOWN when ball is hidden)"
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="output_tracked.mp4")
    parser.add_argument("--log", default="ball_frames.csv")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--spot-x",
        nargs=3,
        type=float,
        default=DEFAULT_SPOT_X,
        metavar=("LEFT", "CENTER", "RIGHT"),
    )
    args = parser.parse_args()
    run(args.video, args.output, args.log, args.debug, tuple(args.spot_x))
'''

ball_interaction_code = '''"""Multi-candidate ball detector with bidirectional temporal corroboration.

Applies the full_game_board guard to prevent UI icon false positives on non-game
frames and corroborates candidates across both previous and next frames.
"""
from __future__ import annotations
import argparse
import csv
import json
import cv2
import numpy as np
from main import detect_cups
from utils import (
    DEFAULT_SPOT_X,
    DetectorConfig,
    is_full_game_board,
    open_video_io,
    save_json_report,
    spot_from_x,
)


def extract_candidates(
    frame: np.ndarray, cfg: DetectorConfig = DetectorConfig()
) -> list[dict]:
    h, w = frame.shape[:2]
    x1, x2 = int(cfg.roi_x_range[0] * w), int(cfg.roi_x_range[1] * w)
    y1, y2 = int(cfg.roi_y_range[0] * h), int(cfg.roi_y_range[1] * h)
    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(cfg.ball_hsv_low), np.array(cfg.ball_hsv_high))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scale = (w * h) / (720 * 1280)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0 or not (130 * scale <= area <= 5000 * scale):
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if min(bw, bh) < 10 * np.sqrt(scale) or max(bw, bh) / max(1, min(bw, bh)) > 1.45:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.48:
            continue
        hull = cv2.convexHull(c)
        hull_area = max(1, cv2.contourArea(hull))
        solidity = area / hull_area
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        shape = max(0.0, min(1.0, (circularity - 0.45) / 0.5))
        solid = max(0.0, min(1.0, (solidity - 0.65) / 0.35))
        ratio = min(bw, bh) / max(bw, bh)
        score = 0.55 * shape + 0.25 * solid + 0.20 * ratio
        out.append(
            {
                "x": int(cx + x1),
                "y": int(cy + y1),
                "radius": int(radius),
                "area": float(area),
                "circularity": float(circularity),
                "solidity": float(solidity),
                "score": float(score),
            }
        )
    return sorted(out, key=lambda z: z["score"], reverse=True)


def cup_relation(ball: dict, boxes: list[tuple[int, int, int, int]], width: int, height: int):
    if not boxes:
        return "CUP_REGIONS_NOT_DETECTED", None, None
    bx, by = ball["x"], ball["y"]
    ranked = []
    for i, (x, y, bw, bh) in enumerate(boxes):
        pad_x, pad_y = 0.22 * bw, 0.18 * bh
        inside = (x - pad_x <= bx <= x + bw + pad_x) and (y - pad_y <= by <= y + bh + pad_y)
        cx, cy = x + bw / 2, y + bh / 2
        d = float(np.hypot((bx - cx) / width, (by - cy) / height))
        ranked.append((d, i + 1, inside))
    d, i, inside = min(ranked)
    if inside:
        return "BALL_OVERLAPS_VISIBLE_CUP_REGION", i, d
    if d < 0.09:
        return "BALL_NEAR_CUP_REGION", i, d
    return "BALL_SEPARATE_FROM_DETECTED_CUPS", i, d


def run(
    video: str,
    output: str,
    log: str,
    report: str,
    spot_x: tuple[float, float, float] = DEFAULT_SPOT_X,
    min_score: float = 0.52,
) -> None:
    cap, writer, fps, w, h = open_video_io(video, output)
    # Pass 1: collect frame detections to enable bidirectional temporal validation
    frame_records = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes = detect_cups(frame)
        full_board = is_full_game_board(boxes, h)
        cands = extract_candidates(frame) if full_board else []
        best = cands[0] if cands and cands[0]["score"] >= min_score else None
        frame_records.append({"boxes": boxes, "cands": cands, "best": best, "full_board": full_board})

    # Rewind to render annotated output video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    prev_spot = None
    rows = []
    accepted = 0
    rejected = 0
    events = []

    for frame_no, rec in enumerate(frame_records):
        ok, frame = cap.read()
        if not ok:
            break
        best = rec["best"]
        boxes = rec["boxes"]
        cands = rec["cands"]
        prev_best = frame_records[frame_no - 1]["best"] if frame_no > 0 else None
        next_best = (
            frame_records[frame_no + 1]["best"]
            if frame_no + 1 < len(frame_records)
            else None
        )

        if best:
            near_prev = (
                prev_best is not None
                and np.hypot(best["x"] - prev_best["x"], best["y"] - prev_best["y"]) < 0.12 * w
            )
            near_next = (
                next_best is not None
                and np.hypot(best["x"] - next_best["x"], best["y"] - next_best["y"]) < 0.12 * w
            )
            temporal = "TEMPORALLY_CONTINUOUS" if (near_prev or near_next) else "SINGLE_FRAME_REVIEW"
            accepted += 1
            spot = spot_from_x(best["x"], w, tuple(spot_x))
            relation, cup_id, cup_dist = cup_relation(best, boxes, w, h)
            if prev_spot is not None and spot != prev_spot:
                events.append(
                    {
                        "frame": frame_no,
                        "time_seconds": round(frame_no / fps, 4),
                        "from_spot": prev_spot,
                        "to_spot": spot,
                        "event": "VISIBLE_SPOT_CHANGE_REQUIRES_REVIEW",
                        "temporal_status": temporal,
                    }
                )
            prev_spot = spot
            cv2.circle(frame, (best["x"], best["y"]), best["radius"], (0, 255, 0), 3)
            cv2.putText(
                frame,
                f"BALL CANDIDATE | SPOT {spot} | {temporal}",
                (int(0.02 * w), int(0.033 * h)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                relation,
                (int(0.02 * w), int(0.052 * h)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 220, 0),
                1,
                cv2.LINE_AA,
            )
            rows.append(
                [
                    frame_no,
                    round(frame_no / fps, 4),
                    1,
                    spot,
                    best["x"],
                    best["y"],
                    round(best["score"], 4),
                    round(best["circularity"], 4),
                    round(best["solidity"], 4),
                    temporal,
                    relation,
                    cup_id if cup_id else "",
                    round(cup_dist, 5) if cup_dist is not None else "",
                    len(cands),
                ]
            )
        else:
            rejected += 1
            status_text = (
                "BALL NOT CONFIRMED | LOCATION UNKNOWN"
                if rec["full_board"]
                else "NOT GAME BOARD | IGNORED"
            )
            cv2.putText(
                frame,
                status_text,
                (int(0.02 * w), int(0.033 * h)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
            rows.append(
                [
                    frame_no,
                    round(frame_no / fps, 4),
                    0,
                    "UNKNOWN",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "NO_ACCEPTED_CANDIDATE",
                    "UNKNOWN" if rec["full_board"] else "NOT_GAME_BOARD",
                    "",
                    "",
                    len(cands),
                ]
            )
        writer.write(frame)

    cap.release()
    writer.release()

    frame_no = len(frame_records)
    summary = {
        "input_video": str(video),
        "fps": fps,
        "frames_processed": frame_no,
        "accepted_candidate_frames": accepted,
        "unknown_frames": rejected,
        "candidate_acceptance_rate": round(accepted / max(1, frame_no), 4),
        "minimum_candidate_score": min_score,
        "visible_spot_change_events": events,
        "cup_interaction_counts": {},
        "interpretation": (
            "Candidate scores rank visual shape/brightness evidence; they are not probabilities. "
            "Non-game frames without the full cup board are excluded. Hidden intervals remain UNKNOWN."
        ),
    }
    for r in rows:
        rel = r[10]
        summary["cup_interaction_counts"][rel] = summary["cup_interaction_counts"].get(rel, 0) + 1

    with open(log, "w", newline="", encoding="utf-8") as f:
        cw = csv.writer(f)
        cw.writerow(
            [
                "frame",
                "time_seconds",
                "ball_candidate",
                "spot",
                "x",
                "y",
                "candidate_score",
                "circularity",
                "solidity",
                "temporal_status",
                "cup_relation",
                "nearest_cup_region",
                "normalized_cup_distance",
                "candidate_count",
            ]
        )
        cw.writerows(rows)

    save_json_report(report, summary)
    print(json.dumps(summary, indent=2))
    print(f"Annotated video: {output}\nCSV: {log}\nReport: {report}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Multi-candidate ball detector with cup-region interaction evidence"
    )
    p.add_argument("--video", required=True)
    p.add_argument("--output", default="ball_interaction_annotated.mp4")
    p.add_argument("--log", default="ball_interaction_frames.csv")
    p.add_argument("--report", default="ball_interaction_report.json")
    p.add_argument(
        "--spot-x",
        nargs=3,
        type=float,
        default=DEFAULT_SPOT_X,
        metavar=("LEFT", "CENTER", "RIGHT"),
    )
    p.add_argument("--min-score", type=float, default=0.52)
    a = p.parse_args()
    run(a.video, a.output, a.log, a.report, tuple(a.spot_x), a.min_score)
'''

ball_transfer_code = '''"""Evidence-based ball visibility and transfer reconstruction.

Enforces the full_game_board guard so phone UI overlays do not create false
ball reappearances after a shuffle finishes.
"""
from __future__ import annotations
import argparse
import csv
import json
from main import detect_ball, detect_cups
from utils import (
    DEFAULT_SPOT_X,
    is_full_game_board,
    open_video_io,
    save_json_report,
    spot_from_x,
)
import cv2


def run(
    video: str,
    output: str,
    log: str,
    report: str,
    spot_x: tuple[float, float, float] = DEFAULT_SPOT_X,
    min_gap_frames: int = 1,
) -> None:
    cap, writer, fps, width, height = open_video_io(video, output)
    rows = []
    observations = []
    frame_no = 0
    hidden_start = None
    previous = None
    events = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = frame_no / fps
        boxes = detect_cups(frame)
        full_board = is_full_game_board(boxes, height)
        ball = detect_ball(frame) if full_board else None

        if ball:
            bx, by, radius, quality = ball
            spot = spot_from_x(bx, width, tuple(spot_x))
            cv2.circle(frame, (bx, by), radius, (0, 255, 0), 3)
            cv2.putText(
                frame,
                f"BALL VISIBLE | SPOT {spot}",
                (int(0.025 * width), int(0.038 * height)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            obs = {
                "frame": frame_no,
                "time_seconds": round(ts, 4),
                "spot": spot,
                "x": bx,
                "y": by,
                "circularity": round(float(quality), 4),
            }
            observations.append(obs)
            if previous is not None and spot != previous["spot"]:
                gap = frame_no - previous["frame"] - 1
                if gap >= min_gap_frames:
                    events.append(
                        {
                            "event_id": len(events) + 1,
                            "from_spot": previous["spot"],
                            "to_spot": spot,
                            "last_seen_before": previous["time_seconds"],
                            "first_seen_after": round(ts, 4),
                            "hidden_frames_between": gap,
                            "classification": "OBSERVED_REAPPEARANCE_AT_DIFFERENT_SPOT_TRANSFER_PATH_UNKNOWN",
                            "confidence": "OBSERVED_ENDPOINTS_ONLY",
                            "note": (
                                "The ball was observed at two different spots with an intervening hidden interval. "
                                "The video does not show the transfer path or prove which cup carried the ball."
                            ),
                        }
                    )
                else:
                    events.append(
                        {
                            "event_id": len(events) + 1,
                            "from_spot": previous["spot"],
                            "to_spot": spot,
                            "last_seen_before": previous["time_seconds"],
                            "first_seen_after": round(ts, 4),
                            "hidden_frames_between": gap,
                            "classification": "VISIBLE_POSITION_CHANGE_REVIEW",
                            "confidence": "DETECTOR_OBSERVATION",
                            "note": "Spot assignment changed between adjacent observations; review detections.",
                        }
                    )
            previous = obs
            hidden_start = None
            rows.append([frame_no, round(ts, 4), 1, spot, bx, by, round(float(quality), 4), "OBSERVED"])
        else:
            if hidden_start is None:
                hidden_start = frame_no
            label = "BALL HIDDEN | LOCATION UNKNOWN" if full_board else "NOT GAME BOARD | IGNORED"
            cv2.putText(
                frame,
                label,
                (int(0.025 * width), int(0.038 * height)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
            rows.append(
                [frame_no, round(ts, 4), 0, "UNKNOWN", "", "", "", "UNKNOWN" if full_board else "NOT_GAME_BOARD"]
            )
        writer.write(frame)
        frame_no += 1

    cap.release()
    writer.release()

    hidden = []
    start = None
    for r in rows:
        if r[2] == 0 and start is None:
            start = r[0]
        if r[2] == 1 and start is not None:
            end = r[0] - 1
            hidden.append(
                {
                    "start_frame": start,
                    "end_frame": end,
                    "start_seconds": round(start / fps, 4),
                    "end_seconds": round(end / fps, 4),
                    "duration_seconds": round((end - start + 1) / fps, 4),
                }
            )
            start = None
    if start is not None:
        end = rows[-1][0]
        hidden.append(
            {
                "start_frame": start,
                "end_frame": end,
                "start_seconds": round(start / fps, 4),
                "end_seconds": round(end / fps, 4),
                "duration_seconds": round((end - start + 1) / fps, 4),
            }
        )

    first = observations[0] if observations else None
    final = observations[-1] if observations else None
    summary = {
        "input_video": str(video),
        "fps": fps,
        "frames_processed": frame_no,
        "visible_observations": len(observations),
        "hidden_frames": sum(1 for r in rows if r[2] == 0),
        "hidden_intervals": hidden,
        "first_observed_ball": first,
        "last_observed_ball": final,
        "observed_endpoint_spot_changed": bool(first and final and first["spot"] != final["spot"]),
        "ball_transfer_events": events,
        "interpretation": (
            "Only visible ball detections on the active game board are reported as observed locations. "
            "During every hidden interval the ball location is UNKNOWN."
        ),
    }

    with open(log, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_seconds", "ball_visible", "observed_spot", "ball_x", "ball_y", "circularity", "state"])
        w.writerows(rows)

    save_json_report(report, summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("hidden_intervals", "ball_transfer_events")}, indent=2))
    print("Ball events:")
    for e in events:
        print(f"  {e['classification']}: {e['from_spot']} -> {e['to_spot']} ({e['last_seen_before']}-{e['first_seen_after']} s)")
    print(f"Annotated video: {output}\nCSV: {log}\nReport: {report}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Reconstruct only visually supported ball locations and endpoint changes")
    p.add_argument("--video", required=True)
    p.add_argument("--output", default="ball_transfer_annotated.mp4")
    p.add_argument("--log", default="ball_transfer_frames.csv")
    p.add_argument("--report", default="ball_transfer_report.json")
    p.add_argument("--spot-x", nargs=3, type=float, default=DEFAULT_SPOT_X, metavar=("LEFT", "CENTER", "RIGHT"))
    p.add_argument("--min-gap-frames", type=int, default=1)
    a = p.parse_args()
    run(a.video, a.output, a.log, a.report, tuple(a.spot_x), a.min_gap_frames)
'''

cup_tracker_code = '''"""Conservative cup-identity tracking with pyramidal Lucas-Kanade optical flow.

Improvements:
- Fixes median_lk_error to compute over filtered inlier points (lk_err[keep]).
- Eliminates redundant second video seek on initialization frame.
- Adds --auto-init option using detect_cups() when three cups are visible.
- Replenishes Shi-Tomasi features during clean TRACKED frames if point count drops.
- Scales HUD text coordinates relative to frame resolution.
"""
from __future__ import annotations
import argparse
import csv
import json
import cv2
import numpy as np
from main import detect_cups
from utils import open_video_io, save_json_report

DEFAULT_BOXES = (
    (0.195, 0.315, 0.375, 0.445),
    (0.425, 0.325, 0.595, 0.455),
    (0.640, 0.315, 0.810, 0.445),
)


def red_feature_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    a = cv2.inRange(hsv, np.array([0, 75, 55]), np.array([15, 255, 255]))
    b = cv2.inRange(hsv, np.array([160, 75, 55]), np.array([180, 255, 255]))
    return cv2.bitwise_or(a, b)


def box_pixels(norm_box, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = norm_box
    return (
        max(0, int(x1 * w)),
        max(0, int(y1 * h)),
        min(w, int(x2 * w)),
        min(h, int(y2 * h)),
    )


def init_points_from_pixel_box(
    gray: np.ndarray, red: np.ndarray, px_box: tuple[int, int, int, int]
) -> np.ndarray:
    x1, y1, x2, y2 = px_box
    roi_mask = np.zeros_like(gray)
    roi_mask[y1:y2, x1:x2] = red[y1:y2, x1:x2]
    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=100,
        qualityLevel=0.008,
        minDistance=3,
        mask=roi_mask,
        blockSize=5,
        useHarrisDetector=False,
    )
    if pts is None:
        return np.empty((0, 1, 2), np.float32)
    return pts.astype(np.float32)


def pairwise_ambiguous(centers, valid, cup_w: float) -> list[bool]:
    flags = [False] * len(centers)
    for i in range(len(centers)):
        if not valid[i]:
            continue
        for j in range(i + 1, len(centers)):
            if not valid[j]:
                continue
            if np.linalg.norm(np.array(centers[i]) - np.array(centers[j])) < 0.72 * cup_w:
                flags[i] = flags[j] = True
    return flags


def run(
    video: str,
    output: str,
    log: str,
    report: str,
    start: float,
    end: float,
    boxes_norm,
    auto_init: bool = False,
) -> None:
    cap, writer, fps, w, h = open_video_io(video, output)
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    ok, first = cap.read()
    if not ok:
        cap.release()
        writer.release()
        raise RuntimeError("Could not read initialization frame")

    gray_prev = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    red_prev = red_feature_mask(first)

    if auto_init:
        detected = detect_cups(first)
        if len(detected) != 3:
            cap.release()
            writer.release()
            raise RuntimeError(
                f"--auto-init expected 3 separated cups at t={start}s, found {len(detected)}"
            )
        px_boxes = [(x, y, x + bw, y + bh) for (x, y, bw, bh) in detected]
    else:
        px_boxes = [box_pixels(b, w, h) for b in boxes_norm]

    points = [init_points_from_pixel_box(gray_prev, red_prev, b) for b in px_boxes]
    states = []
    for i, p in enumerate(points):
        states.append(
            {
                "id": i + 1,
                "points": p,
                "center": np.mean(p[:, 0, :], axis=0) if len(p) else np.array([np.nan, np.nan]),
                "lost": len(p) < 6,
            }
        )

    rows = []
    frame_no = round(start * fps)
    prev_order = None
    swap_candidates = []
    cup_w = 0.175 * w
    cup_h = 0.130 * h

    # Process initialization frame + subsequent frames without seeking twice
    current_frame = first
    while True:
        ts = frame_no / fps
        if ts > end:
            break
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        red_curr = red_feature_mask(current_frame)
        centers = []
        valid = []
        details = []

        if frame_no == round(start * fps):
            for st in states:
                c = st["center"]
                v = not st["lost"]
                centers.append((float(c[0]), float(c[1])))
                valid.append(v)
                details.append((len(st["points"]), 0.0 if v else 999.0, v))
        else:
            for st in states:
                p = st["points"]
                if st["lost"] or len(p) < 6:
                    centers.append(tuple(st["center"]))
                    valid.append(False)
                    details.append((0, 999.0, False))
                    continue
                nxt, status, err = cv2.calcOpticalFlowPyrLK(
                    gray_prev,
                    gray,
                    p,
                    None,
                    winSize=(41, 41),
                    maxLevel=5,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                )
                if nxt is None or status is None:
                    st["lost"] = True
                    centers.append(tuple(st["center"]))
                    valid.append(False)
                    details.append((0, 999.0, False))
                    continue
                good = status.reshape(-1) == 1
                new = nxt[good]
                lk_err = err.reshape(-1)[good] if err is not None and np.any(good) else np.array([])
                if len(lk_err):
                    keep = lk_err < 60.0
                    new = new[keep]
                    kept_err = lk_err[keep]
                else:
                    kept_err = np.array([])

                if len(new) < 5:
                    st["lost"] = True
                    centers.append(tuple(st["center"]))
                    valid.append(False)
                    details.append((len(new), float(np.median(kept_err)) if len(kept_err) else 999.0, False))
                    continue

                center = np.median(new[:, 0, :], axis=0)
                st["center"] = center
                st["points"] = new.astype(np.float32)
                centers.append((float(center[0]), float(center[1])))
                valid.append(True)
                mederr = float(np.median(kept_err)) if len(kept_err) else 999.0
                details.append((len(new), mederr, True))

        overlap = pairwise_ambiguous(centers, valid, cup_w)

        # Replenish features when a track is clean and unambiguous but running low on corners
        for st, v, amb in zip(states, valid, overlap):
            if v and not amb and len(st["points"]) < 25:
                cx, cy = st["center"]
                rx1 = max(0, int(cx - 0.5 * cup_w))
                ry1 = max(0, int(cy - 0.5 * cup_h))
                rx2 = min(w, int(cx + 0.5 * cup_w))
                ry2 = min(h, int(cy + 0.5 * cup_h))
                fresh = init_points_from_pixel_box(gray, red_curr, (rx1, ry1, rx2, ry2))
                if len(fresh) >= len(st["points"]):
                    st["points"] = fresh

        order = None
        if all(valid) and not any(overlap):
            order = tuple(int(x) for x in np.argsort([c[0] for c in centers]) + 1)
            if prev_order and order != prev_order:
                swap_candidates.append(
                    {
                        "time_seconds": round(ts, 4),
                        "previous_left_to_right": list(prev_order),
                        "new_left_to_right": list(order),
                        "status": "CANDIDATE_ORDER_CHANGE_REQUIRES_REVIEW",
                    }
                )
            prev_order = order

        for i, (st, c, v, d, amb) in enumerate(zip(states, centers, valid, details, overlap)):
            x, y = c
            npts, err_val, _ = d
            status = "LOST" if not v else ("OVERLAP_AMBIGUOUS" if amb else "TRACKED")
            if v:
                color = (0, 165, 255) if amb else (0, 255, 0)
                cv2.circle(current_frame, (int(x), int(y)), max(5, int(0.011 * w)), color, -1)
                cv2.putText(
                    current_frame,
                    f"CUP {st['id']}: {status} ({npts})",
                    (max(5, int(x - 0.075 * w)), max(20, int(y - 0.066 * h))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.43,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    current_frame,
                    f"CUP {st['id']}: LOST",
                    (int(0.028 * w), int((0.17 + 0.02 * i) * h)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
            rows.append(
                [
                    frame_no,
                    round(ts, 4),
                    st["id"],
                    round(float(x), 2) if np.isfinite(x) else "",
                    round(float(y), 2) if np.isfinite(y) else "",
                    npts,
                    round(err_val, 3),
                    status,
                ]
            )

        headline = "IDENTITY TRACKING | " + (
            "ALL TRACKS AMBIGUOUS/LOST"
            if not all(valid) or any(overlap)
            else "FEATURE TRACKS VALID"
        )
        cv2.rectangle(
            current_frame,
            (int(0.09 * w), int(0.117 * h)),
            (int(0.91 * w), int(0.144 * h)),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            current_frame,
            headline,
            (int(0.10 * w), int(0.136 * h)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        writer.write(current_frame)
        gray_prev = gray
        frame_no += 1
        ok, current_frame = cap.read()
        if not ok:
            break

    cap.release()
    writer.release()

    with open(log, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(
            [
                "frame",
                "time_seconds",
                "cup_id",
                "feature_center_x",
                "feature_center_y",
                "tracked_features",
                "median_lk_error",
                "identity_state",
            ]
        )
        wr.writerows(rows)

    counts = {}
    for r in rows:
        counts[r[-1]] = counts.get(r[-1], 0) + 1

    summary = {
        "input_video": str(video),
        "fps": fps,
        "shuffle_window_seconds": [start, end],
        "frames_processed": len(set(r[0] for r in rows)),
        "track_rows": len(rows),
        "initial_feature_counts": {f"cup_{i + 1}": int(len(p)) for i, p in enumerate(points)},
        "identity_state_counts_by_cup_frame": counts,
        "candidate_order_changes": swap_candidates,
        "interpretation": (
            "Lucas-Kanade tracks image features, not semantic cup identities. "
            "OVERLAP_AMBIGUOUS and LOST are explicit. This module does not infer ball ownership."
        ),
    }
    save_json_report(report, summary)
    print(json.dumps(summary, indent=2))
    print(f"Annotated video: {output}\nCSV: {log}\nReport: {report}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Conservative persistent cup feature tracking")
    p.add_argument("--video", required=True)
    p.add_argument("--output", default="cup_identity_annotated.mp4")
    p.add_argument("--log", default="cup_identity_frames.csv")
    p.add_argument("--report", default="cup_identity_report.json")
    p.add_argument("--start", type=float, default=2.0)
    p.add_argument("--end", type=float, default=11.5)
    p.add_argument("--auto-init", action="store_true", help="Auto-detect 3 cup boxes at --start")
    p.add_argument(
        "--boxes",
        nargs=12,
        type=float,
        default=None,
        help="12 normalized values: x1 y1 x2 y2 for cup 1, cup 2, cup 3",
    )
    a = p.parse_args()
    b = DEFAULT_BOXES if a.boxes is None else tuple(tuple(a.boxes[i : i + 4]) for i in (0, 4, 8))
    run(a.video, a.output, a.log, a.report, a.start, a.end, b, a.auto_init)
'''

eval_code = '''#!/usr/bin/env python3
"""Evaluate detector output CSV against manually reviewed labels (stdlib only).

Computes precision, recall, visible spot accuracy, not_game exclusion count,
and supports optional score-threshold sweeps when candidate_score is present.
"""
import argparse
import csv
import json


def evaluate_rows(labels: list[dict], detections: dict[int, dict], min_score: float | None = None) -> dict:
    visible = [r for r in labels if r["ground_truth_state"] == "visible"]
    occluded = [r for r in labels if r["ground_truth_state"] == "occluded"]
    not_game = [r for r in labels if r["ground_truth_state"] == "not_game"]
    uncertain = [r for r in labels if r["ground_truth_state"] == "uncertain"]

    def is_positive(det: dict | None) -> bool:
        if not det:
            return False
        raw = int(det.get("ball_candidate", det.get("ball_visible", 0)))
        if not raw:
            return False
        if min_score is not None and det.get("candidate_score"):
            return float(det["candidate_score"]) >= min_score
        return True

    tp = 0
    spot_correct = 0
    for r in visible:
        det = detections.get(int(r["frame"]))
        if is_positive(det):
            tp += 1
            pred_spot = str(det.get("spot", det.get("observed_spot", ""))).strip()
            gt_spot = str(r.get("ground_truth_spot", "")).strip()
            if gt_spot and pred_spot == gt_spot:
                spot_correct += 1

    fn = len(visible) - tp
    fp_occluded = sum(1 for r in occluded if is_positive(detections.get(int(r["frame"]))))
    tn_occluded = len(occluded) - fp_occluded
    fp_not_game = sum(1 for r in not_game if is_positive(detections.get(int(r["frame"]))))

    total_fp = fp_occluded + fp_not_game
    return {
        "sampled_game_frames": len(visible) + len(occluded),
        "ground_truth_visible_frames": len(visible),
        "ground_truth_occluded_frames": len(occluded),
        "not_game_frames_excluded": len(not_game),
        "uncertain_frames_excluded": len(uncertain),
        "true_positive": tp,
        "false_negative": fn,
        "false_positive_on_occluded": fp_occluded,
        "false_positive_on_not_game": fp_not_game,
        "true_negative": tn_occluded,
        "precision": round(tp / (tp + total_fp), 4) if (tp + total_fp) else None,
        "recall": round(tp / len(visible), 4) if visible else None,
        "visible_spot_correct_count": spot_correct,
        "visible_spot_accuracy": round(spot_correct / len(visible), 4) if visible else None,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default="data/manual_ball_labels.csv")
    p.add_argument("--detections", required=True)
    p.add_argument("--sweep", action="store_true", help="Sweep candidate_score from 0.40 to 0.80")
    a = p.parse_args()

    with open(a.labels, newline="", encoding="utf-8") as f:
        labels = list(csv.DictReader(f))
    with open(a.detections, newline="", encoding="utf-8") as f:
        detections = {int(r["frame"]): r for r in csv.DictReader(f)}

    result = evaluate_rows(labels, detections)
    if a.sweep:
        result["threshold_sweep"] = {
            f"{t:.2f}": evaluate_rows(labels, detections, min_score=t)
            for t in [0.40, 0.45, 0.50, 0.52, 0.55, 0.60, 0.65, 0.70, 0.75]
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
'''

readme_changes = '''# Refactored Modules & Improvements Summary

This archive contains the updated Python modules implementing the code review recommendations for `Cup-state-updated`:

## Included Files
1. **`utils.py`**
   - Centralizes `DetectorConfig`, `open_video_io()`, `is_full_game_board()`, `spot_from_x()`, and `save_json_report()` to eliminate duplicate boilerplate across scripts.
2. **`main.py`**
   - Adds hysteresis ball detection (`detect_ball(..., prev_xy=...)`) to recover motion-blurred visible ball frames without lowering initial detection precision.
   - Scales overlay text coordinates proportionally to video resolution.
3. **`ball_interaction_analysis.py`**
   - Applies `is_full_game_board()` so bright UI icons on non-game phone screens are ignored.
   - Implements bidirectional temporal corroboration (`near_prev or near_next`) so the first frame of a valid multi-frame appearance is marked `TEMPORALLY_CONTINUOUS`.
4. **`ball_transfer_reconstruction.py`**
   - Enforces `is_full_game_board()` before running `detect_ball()`, preventing post-game UI overlays from triggering false ball reappearances.
5. **`cup_identity_tracker.py`**
   - Fixes `median_lk_error` so it is calculated strictly over filtered inlier features (`lk_err[keep]`).
   - Removes the duplicate `cap.set(cv2.CAP_PROP_POS_MSEC, ...)` seek call.
   - Adds `--auto-init` to initialize cup ROIs automatically via `detect_cups()` when 3 separated cups are visible at `--start`.
   - Adds mid-track feature replenishment during unambiguous `TRACKED` frames when feature count drops below 25.
   - Scales HUD coordinates relative to frame width and height.
6. **`evaluate_manual_labels.py`**
   - Expands evaluation output to include `visible_spot_accuracy`, `visible_spot_correct_count`, `not_game_frames_excluded`, `false_positive_on_not_game`, and an optional `--sweep` flag for threshold calibration.
'''

files = {
    "utils.py": utils_code,
    "main.py": main_code,
    "ball_interaction_analysis.py": ball_interaction_code,
    "ball_transfer_reconstruction.py": ball_transfer_code,
    "cup_identity_tracker.py": cup_tracker_code,
    "evaluate_manual_labels.py": eval_code,
    "REFACTORING_NOTES.md": readme_changes,
}

for fname, content in files.items():
    with open(os.path.join("temp/cup_state_refactored", fname), "w", encoding="utf-8") as f:
        f.write(content)

zip_path = "generated/Cup-state-updated-refactored.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in files:
        zf.write(
            os.path.join("temp/cup_state_refactored", fname),
            arcname=f"Cup-state-updated-refactored/{fname}",
        )

print("Created:", zip_path, os.path.getsize(zip_path))
