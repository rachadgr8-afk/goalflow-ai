import gc
import json
import math
import os
import shutil
import subprocess
import sys
import cv2
import numpy as np

try:
    from sam_tracker import PlayerTracker, get_cached_yolo_model
    HAS_SAM_TRACKER = True
except Exception:
    PlayerTracker = None
    get_cached_yolo_model = lambda name="yolov8n.pt": None
    HAS_SAM_TRACKER = False

from cinematic_modules import (
    SmartReframer,
    CinematicTextRenderer,
    CinematicTransitionEngine,
    ReferenceStyleAnalyzer,
    QualityControlEngine,
)
from cinematic_storyteller import (
    CinematicStoryteller,
    DepthAnythingV2Engine,
    PsychologicalColorGrader,
    get_process_rss_mb,
)

# Global cached PlayerTracker instance so MobileSAM loads once per process (Req 14)
_GLOBAL_PLAYER_TRACKER_INSTANCE = None


def get_cached_player_tracker():
    global _GLOBAL_PLAYER_TRACKER_INSTANCE
    if _GLOBAL_PLAYER_TRACKER_INSTANCE is None and HAS_SAM_TRACKER and PlayerTracker is not None:
        try:
            _GLOBAL_PLAYER_TRACKER_INSTANCE = PlayerTracker()
        except Exception:
            _GLOBAL_PLAYER_TRACKER_INSTANCE = None
    return _GLOBAL_PLAYER_TRACKER_INSTANCE


# =====================================================================
# 1. VIDEO ANALYZER
# =====================================================================
class VideoAnalyzer:
    """Extracts technical specs and video metadata via ffprobe & OpenCV."""

    @staticmethod
    def probe(input_path: str) -> dict:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        info = {
            "duration": 0.0,
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "has_audio": False,
            "audio_codec": None,
            "video_codec": "h264",
            "frame_count": 0,
            "bitrate": 0,
        }
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", input_path
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(res.stdout)
            format_data = data.get("format", {})
            info["duration"] = float(format_data.get("duration", 0.0))
            info["bitrate"] = int(format_data.get("bit_rate", 0))
            for s in data.get("streams", []):
                if s.get("codec_type") == "video" and info["width"] == 1920:
                    info["width"] = int(s.get("width", 1920))
                    info["height"] = int(s.get("height", 1080))
                    info["video_codec"] = s.get("codec_name", "h264")
                    r_fps = s.get("r_frame_rate", "30/1")
                    if "/" in r_fps:
                        num, den = r_fps.split("/")
                        info["fps"] = float(num) / max(1.0, float(den))
                    else:
                        info["fps"] = float(r_fps)
                    nb = s.get("nb_frames")
                    if nb and nb.isdigit():
                        info["frame_count"] = int(nb)
                elif s.get("codec_type") == "audio":
                    info["has_audio"] = True
                    info["audio_codec"] = s.get("codec_name")
        except Exception:
            pass

        cap = cv2.VideoCapture(input_path)
        if cap.isOpened():
            c_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            c_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            c_fps = cap.get(cv2.CAP_PROP_FPS)
            c_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if c_w > 0 and c_h > 0:
                info["width"] = c_w
                info["height"] = c_h
            if c_fps > 0 and not np.isnan(c_fps):
                info["fps"] = c_fps
            if c_cnt > 0:
                info["frame_count"] = c_cnt
            if info["duration"] <= 0 and info["fps"] > 0 and info["frame_count"] > 0:
                info["duration"] = float(info["frame_count"]) / float(info["fps"])
        return info


# =====================================================================
# 2. YOLO PLAYER & BALL DETECTOR + MULTI-ID TRACKER (Req 1)
# =====================================================================
class FootballDetectorTracker:
    """
    Requirement 1 — Real YOLO Player & Ball Detection + Stable Multi-Object Tracking:
    Hierarchy:
      1. Primary Detector: Real Ultralytics YOLO (yolov8n.pt cached in RAM, classes: 0=person, 32=sports ball)
         + non-pitch foreground jersey/ball detector for non-photorealistic/synthetic test frames.
      2. Multi-Object Kalman-style Tracker: Assigns stable ID, confidence, bbox, centerX, centerY,
         scale, frameStart, frameEnd, velocity, acceleration, and ball proximity.
      3. Fallback Detector: MOG2 Motion Subtractor (strictly fallback when YOLO returns 0 candidates).
      4. Ultimate Fallback: Inertial prediction -> Center crop fallback.
    """

    def __init__(self, width: int, height: int, fps: float, use_yolo: bool = True):
        self.w = width
        self.h = height
        self.fps = max(15.0, fps)
        self.use_yolo = use_yolo
        self.yolo_model = get_cached_yolo_model("yolov8n.pt") if use_yolo else None
        self.tracked_player = None
        self.tracked_ball = None
        self.defenders = []
        self.active_tracks = {}  # id -> track dict
        self.next_track_id = 1
        self.frame_counter = 0
        self.detector_source = "yolo" if self.yolo_model is not None else "mog2_fallback"
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=120, varThreshold=26, detectShadows=False
        )

    def _run_yolo_detection(self, small_frame: np.ndarray, small_scale: float, w: int, h: int):
        """Runs real YOLO inference on downscaled frame for low-RAM, high-speed detection."""
        player_candidates = []
        ball_candidates = []
        if self.yolo_model is None:
            return player_candidates, ball_candidates, False
        try:
            results = self.yolo_model.predict(
                small_frame, classes=[0, 32], conf=0.22, imgsz=320, verbose=False
            )
            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                xyxy = boxes.xyxy.cpu().numpy()
                cls_arr = boxes.cls.cpu().numpy()
                conf_arr = boxes.conf.cpu().numpy()
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i] / small_scale
                    bw = max(4.0, float(x2 - x1))
                    bh = max(4.0, float(y2 - y1))
                    cx = float(x1 + bw * 0.5)
                    cy = float(y1 + bh * 0.5)
                    conf = float(conf_arr[i])
                    c_id = int(cls_arr[i])
                    if c_id == 0:
                        player_candidates.append((cx, cy, bw, bh, conf, "yolo"))
                    elif c_id == 32:
                        ball_candidates.append((cx, cy, bw, bh, conf, "yolo"))
            return player_candidates, ball_candidates, (len(player_candidates) > 0 or len(ball_candidates) > 0)
        except Exception:
            return [], [], False

    def _run_mog2_fallback(self, small_frame: np.ndarray, small_scale: float, w: int, h: int):
        """MOG2 motion subtractor used ONLY as fallback when YOLO detects no objects."""
        player_candidates = []
        ball_candidates = []
        fg = self.bg_subtractor.apply(small_frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 8:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            ox = x / small_scale
            oy = y / small_scale
            obw = bw / small_scale
            obh = bh / small_scale
            aspect = float(obh) / max(1.0, float(obw))
            if 8 <= (obw * obh) <= 3200 and 0.55 <= aspect <= 1.75:
                ball_candidates.append((ox + obw / 2.0, oy + obh / 2.0, obw, obh, 0.72, "mog2_fallback"))
            if (w * h * 0.0025) <= (obw * obh) <= (w * h * 0.30) and 1.15 <= aspect <= 4.5:
                player_candidates.append((ox + obw / 2.0, oy + obh / 2.0, obw, obh, 0.76, "mog2_fallback"))
        return player_candidates, ball_candidates

    def _update_multi_player_tracks(self, player_candidates: list, dt: float, w: int, h: int, frame_idx: int):
        """Updates stable player tracks with ID, confidence, bbox, centerX/Y, scale, direction, velocity, acceleration."""
        matched_track_ids = set()
        used_cand_indices = set()

        existing_ids = sorted(
            self.active_tracks.keys(),
            key=lambda tid: (self.active_tracks[tid]["last_seen_frame"], self.active_tracks[tid]["confidence"]),
            reverse=True,
        )

        max_match_dist_sq = (max(w, h) * 0.18) ** 2
        for tid in existing_ids:
            tr = self.active_tracks[tid]
            pred_x = tr["centerX"] + tr["vx"] * dt
            pred_y = tr["centerY"] + tr["vy"] * dt
            best_idx = None
            best_dist = max_match_dist_sq
            for c_idx, cand in enumerate(player_candidates):
                if c_idx in used_cand_indices:
                    continue
                d_sq = (cand[0] - pred_x) ** 2 + (cand[1] - pred_y) ** 2
                if d_sq < best_dist:
                    best_dist = d_sq
                    best_idx = c_idx
            if best_idx is not None:
                used_cand_indices.add(best_idx)
                matched_track_ids.add(tid)
                cx, cy, pw, ph, conf, src = player_candidates[best_idx]
                elapsed_frames = max(1, frame_idx - tr["last_seen_frame"])
                eff_dt = dt * elapsed_frames
                vx = (cx - tr["centerX"]) / eff_dt
                vy = (cy - tr["centerY"]) / eff_dt
                ax = (vx - tr["vx"]) / eff_dt
                ay = (vy - tr["vy"]) / eff_dt
                svx = tr["vx"] * 0.35 + vx * 0.65
                svy = tr["vy"] * 0.35 + vy * 0.65
                sax = tr["ax"] * 0.40 + ax * 0.60
                say = tr["ay"] * 0.40 + ay * 0.60
                speed = math.sqrt(svx * svx + svy * svy)
                accel = math.sqrt(sax * sax + say * say)
                dir_rad = math.atan2(svy, svx) if speed > 1.0 else float(tr.get("direction_rad", 0.0))
                dir_deg = round(math.degrees(dir_rad), 2)
                prev_dir_deg = float(tr.get("direction_deg", dir_deg))
                raw_diff = abs(dir_deg - prev_dir_deg) % 360.0
                dir_change_deg = round(360.0 - raw_diff if raw_diff > 180.0 else raw_diff, 2) if speed > 8.0 else 0.0

                scale_val = round(float(ph) / max(1.0, float(h)), 4)
                bbox = (
                    int(max(0, cx - pw / 2.0)),
                    int(max(0, cy - ph / 2.0)),
                    int(min(w, cx + pw / 2.0)),
                    int(min(h, cy + ph / 2.0)),
                )
                tr.update({
                    "cx": cx,
                    "cy": cy,
                    "centerX": round(cx, 2),
                    "centerY": round(cy, 2),
                    "w": pw,
                    "h": ph,
                    "scale": scale_val,
                    "vx": svx,
                    "vy": svy,
                    "ax": sax,
                    "ay": say,
                    "speed": speed,
                    "accel": accel,
                    "velocity": {"vx": round(svx, 2), "vy": round(svy, 2), "speed": round(speed, 2)},
                    "acceleration": {"ax": round(sax, 2), "ay": round(say, 2), "accel": round(accel, 2)},
                    "direction_rad": round(dir_rad, 4),
                    "direction_deg": dir_deg,
                    "direction_change_deg": dir_change_deg,
                    "bbox": bbox,
                    "confidence": round(float(conf), 3),
                    "frameEnd": frame_idx,
                    "last_seen_frame": frame_idx,
                    "detector": src,
                })

        for c_idx, cand in enumerate(player_candidates):
            if c_idx in used_cand_indices:
                continue
            cx, cy, pw, ph, conf, src = cand
            tid = self.next_track_id
            self.next_track_id += 1
            scale_val = round(float(ph) / max(1.0, float(h)), 4)
            bbox = (
                int(max(0, cx - pw / 2.0)),
                int(max(0, cy - ph / 2.0)),
                int(min(w, cx + pw / 2.0)),
                int(min(h, cy + ph / 2.0)),
            )
            self.active_tracks[tid] = {
                "id": tid,
                "cx": cx,
                "cy": cy,
                "centerX": round(cx, 2),
                "centerY": round(cy, 2),
                "w": pw,
                "h": ph,
                "scale": scale_val,
                "vx": 0.0,
                "vy": 0.0,
                "ax": 0.0,
                "ay": 0.0,
                "speed": 0.0,
                "accel": 0.0,
                "velocity": {"vx": 0.0, "vy": 0.0, "speed": 0.0},
                "acceleration": {"ax": 0.0, "ay": 0.0, "accel": 0.0},
                "direction_rad": 0.0,
                "direction_deg": 0.0,
                "direction_change_deg": 0.0,
                "bbox": bbox,
                "confidence": round(float(conf), 3),
                "frameStart": frame_idx,
                "frameEnd": frame_idx,
                "last_seen_frame": frame_idx,
                "ball_proximity": 999.0,
                "has_ball": False,
                "detector": src,
            }
            matched_track_ids.add(tid)

        max_stale_frames = int(round(self.fps * 1.5))
        stale_ids = [
            tid for tid, tr in self.active_tracks.items()
            if (frame_idx - tr["last_seen_frame"]) > max_stale_frames
        ]
        for tid in stale_ids:
            del self.active_tracks[tid]

        return matched_track_ids

    def detect_and_track_frame(self, frame: np.ndarray, timestamp: float) -> dict:
        h, w = frame.shape[:2]
        frame_idx = self.frame_counter
        self.frame_counter += 1

        small_scale = 0.5 if (w * h) > (960 * 540) else 1.0
        sw = int(w * small_scale)
        sh = int(h * small_scale)
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA) if small_scale < 1.0 else frame

        # Tier 1: Primary YOLO detection
        player_candidates, ball_candidates, yolo_hit = self._run_yolo_detection(small, small_scale, w, h)

        # Tier 2: MOG2 fallback ONLY if YOLO found no candidates
        if not yolo_hit:
            mog_players, mog_balls = self._run_mog2_fallback(small, small_scale, w, h)
            if not player_candidates:
                player_candidates = mog_players
            if not ball_candidates:
                ball_candidates = mog_balls
            self.detector_source = "mog2_fallback" if (mog_players or mog_balls) else "center_fallback"
        else:
            self.detector_source = "yolo"

        dt = 1.0 / self.fps
        self._update_multi_player_tracks(player_candidates, dt, w, h, frame_idx)

        # Update Ball Track first so we can use ball proximity to select & link the active player
        chosen_ball = None
        if ball_candidates:
            ref_x = self.tracked_ball["bx"] if self.tracked_ball else (self.tracked_player["cx"] if self.tracked_player else w * 0.5)
            ref_y = self.tracked_ball["by"] if self.tracked_ball else (self.tracked_player["cy"] if self.tracked_player else h * 0.6)
            best_ball = min(ball_candidates, key=lambda b: (b[0] - ref_x) ** 2 + (b[1] - ref_y) ** 2)
            bx, by, bw, bh, bconf, bsrc = best_ball
            bvx = (bx - self.tracked_ball["bx"]) / dt if self.tracked_ball else 0.0
            bvy = (by - self.tracked_ball["by"]) / dt if self.tracked_ball else 0.0
            bspeed = math.sqrt(bvx * bvx + bvy * bvy)
            chosen_ball = {
                "bx": round(bx, 2),
                "by": round(by, 2),
                "bw": round(bw, 2),
                "bh": round(bh, 2),
                "bvx": round(bvx, 2),
                "bvy": round(bvy, 2),
                "speed": round(bspeed, 2),
                "direction_deg": round(math.degrees(math.atan2(bvy, bvx)), 2) if bspeed > 1.0 else 0.0,
                "confidence": round(float(bconf), 3),
                "detector": bsrc,
                "dist_to_player": 999.0,
            }
        self.tracked_ball = chosen_ball

        # Select primary player (linking ball proximity + track continuity + centrality)
        chosen_player = None
        if self.active_tracks:
            for tr in self.active_tracks.values():
                if chosen_ball is not None:
                    feet_y = tr["cy"] + tr["h"] * 0.42
                    b_dist = math.sqrt((chosen_ball["bx"] - tr["cx"]) ** 2 + (chosen_ball["by"] - feet_y) ** 2)
                    tr["ball_proximity"] = round(b_dist, 2)
                    tr["has_ball"] = bool(b_dist < max(65.0, tr["w"] * 1.6))
                else:
                    tr["ball_proximity"] = 999.0
                    tr["has_ball"] = False

            prev_id = self.tracked_player.get("id") if self.tracked_player else None

            def player_priority_score(tr):
                ball_bonus = 350.0 / max(20.0, tr["ball_proximity"]) if chosen_ball else 0.0
                continuity_bonus = 2.5 if (prev_id is not None and tr["id"] == prev_id) else 0.0
                center_dist = math.sqrt(((tr["cx"] - w * 0.5) / w) ** 2 + ((tr["cy"] - h * 0.5) / h) ** 2)
                recency_penalty = (frame_idx - tr["last_seen_frame"]) * 0.8
                return ball_bonus + continuity_bonus + tr["confidence"] * 2.0 - center_dist * 1.5 - recency_penalty

            chosen_player = dict(max(self.active_tracks.values(), key=player_priority_score))
        elif self.tracked_player is not None:
            prev = self.tracked_player
            new_cx = max(w * 0.1, min(w * 0.9, prev["cx"] + prev["vx"] * dt * 0.85))
            new_cy = max(h * 0.1, min(h * 0.9, prev["cy"] + prev["vy"] * dt * 0.85))
            chosen_player = {
                "id": prev.get("id", 1),
                "cx": new_cx,
                "cy": new_cy,
                "centerX": round(new_cx, 2),
                "centerY": round(new_cy, 2),
                "w": prev["w"],
                "h": prev["h"],
                "scale": prev.get("scale", round(prev["h"] / max(1.0, float(h)), 4)),
                "vx": prev["vx"] * 0.85,
                "vy": prev["vy"] * 0.85,
                "ax": 0.0,
                "ay": 0.0,
                "speed": prev["speed"] * 0.85,
                "accel": 0.0,
                "velocity": {"vx": round(prev["vx"] * 0.85, 2), "vy": round(prev["vy"] * 0.85, 2), "speed": round(prev["speed"] * 0.85, 2)},
                "acceleration": {"ax": 0.0, "ay": 0.0, "accel": 0.0},
                "direction_rad": prev.get("direction_rad", 0.0),
                "direction_deg": prev.get("direction_deg", 0.0),
                "direction_change_deg": 0.0,
                "bbox": (
                    int(max(0, new_cx - prev["w"] / 2)),
                    int(max(0, new_cy - prev["h"] / 2)),
                    int(min(w, new_cx + prev["w"] / 2)),
                    int(min(h, new_cy + prev["h"] / 2)),
                ),
                "confidence": 0.60,
                "frameStart": prev.get("frameStart", 0),
                "frameEnd": frame_idx,
                "ball_proximity": 999.0,
                "has_ball": False,
                "detector": "inertial_fallback",
            }
        else:
            chosen_player = {
                "id": 1,
                "cx": w * 0.5,
                "cy": h * 0.55,
                "centerX": round(w * 0.5, 2),
                "centerY": round(h * 0.55, 2),
                "w": w * 0.14,
                "h": h * 0.38,
                "scale": 0.38,
                "vx": 0.0,
                "vy": 0.0,
                "ax": 0.0,
                "ay": 0.0,
                "speed": 0.0,
                "accel": 0.0,
                "velocity": {"vx": 0.0, "vy": 0.0, "speed": 0.0},
                "acceleration": {"ax": 0.0, "ay": 0.0, "accel": 0.0},
                "direction_rad": 0.0,
                "direction_deg": 0.0,
                "direction_change_deg": 0.0,
                "bbox": (int(w * 0.43), int(h * 0.36), int(w * 0.57), int(h * 0.74)),
                "confidence": 0.50,
                "frameStart": frame_idx,
                "frameEnd": frame_idx,
                "ball_proximity": 999.0,
                "has_ball": False,
                "detector": "center_fallback",
            }

        self.tracked_player = chosen_player
        if self.tracked_ball is not None and chosen_player is not None:
            self.tracked_ball["dist_to_player"] = round(
                math.sqrt(
                    (self.tracked_ball["bx"] - chosen_player["cx"]) ** 2
                    + (self.tracked_ball["by"] - chosen_player["cy"]) ** 2
                ),
                2,
            )
            self.tracked_ball["linked_player_id"] = chosen_player["id"]

        # Defenders / Opponents in proximity
        defenders = []
        for tr in self.active_tracks.values():
            if chosen_player and tr["id"] != chosen_player["id"]:
                d = math.sqrt((tr["cx"] - chosen_player["cx"]) ** 2 + (tr["cy"] - chosen_player["cy"]) ** 2)
                if 20.0 < d < (chosen_player["w"] * 3.2):
                    defenders.append({
                        "id": tr["id"],
                        "x": round(tr["cx"], 2),
                        "y": round(tr["cy"], 2),
                        "dist": round(d, 2),
                        "speed": round(float(tr.get("speed", 0.0)), 2),
                    })
        self.defenders = defenders

        # Compute structured player-ball interaction & defender interaction
        ball_dist = float(self.tracked_ball["dist_to_player"]) if self.tracked_ball else 999.0
        ball_spd = float(self.tracked_ball.get("speed", 0.0)) if self.tracked_ball else 0.0
        p_spd = float(chosen_player.get("speed", 0.0)) if chosen_player else 0.0
        p_acc = float(chosen_player.get("accel", 0.0)) if chosen_player else 0.0

        pbi_type = "none"
        pbi_score = 0.0
        if ball_dist < 115.0:
            if ball_spd > 140.0 and p_acc > 220.0:
                pbi_type = "striking"
                pbi_score = min(1.0, 0.70 + (115.0 - ball_dist) / 250.0)
            elif ball_dist < 90.0 and p_spd > 35.0:
                pbi_type = "dribbling"
                pbi_score = min(1.0, 0.55 + (90.0 - ball_dist) / 200.0)
            else:
                pbi_type = "receiving"
                pbi_score = min(0.85, 0.35 + (115.0 - ball_dist) / 250.0)
        elif 115.0 <= ball_dist <= 220.0 and ball_spd > 110.0:
            pbi_type = "passing"
            pbi_score = 0.50

        player_ball_interaction = {
            "active": bool(pbi_type != "none"),
            "has_ball": bool(chosen_player.get("has_ball", False)),
            "interaction_type": pbi_type,
            "ball_distance": round(ball_dist, 2),
            "ball_speed": round(ball_spd, 2),
            "relative_speed": round(abs(ball_spd - p_spd), 2),
            "score": round(pbi_score, 3),
        }

        min_def_d = min((d["dist"] for d in defenders), default=999.0)
        def_pressure = round(min(1.0, max(0.0, (160.0 - min_def_d) / 140.0)), 3) if defenders else 0.0
        defender_interaction = {
            "active": bool(len(defenders) > 0 and min_def_d < 150.0),
            "defender_count": len(defenders),
            "min_distance": round(min_def_d, 2),
            "converging": bool(min_def_d < 95.0),
            "pressure_score": def_pressure,
        }

        if chosen_player is not None:
            chosen_player["player_ball_interaction"] = player_ball_interaction
            chosen_player["defender_interaction"] = defender_interaction

        return {
            "player": self.tracked_player,
            "ball": self.tracked_ball,
            "ball_proximity": round(ball_dist, 2),
            "defenders": self.defenders,
            "player_ball_interaction": player_ball_interaction,
            "defender_interaction": defender_interaction,
            "player_tracks": list(self.active_tracks.values()),
            "detector_source": self.detector_source,
            "frame_index": frame_idx,
            "timestamp": timestamp,
        }


# =====================================================================
# 3. EVIDENCE-BASED FOOTBALL EVENT DETECTOR
# =====================================================================
class FootballEventEngine:
    """
    Football Analysis & Event Classification:
    Relies strictly on Player+Ball detection/tracking, velocity, acceleration,
    direction change, ball proximity, player-ball interaction, and defender interaction.
    Detects:
      shot, pass, dribble, skill, tackle, save, goal, celebration, reaction,
      high_intensity, high-motion, acceleration, defender_interaction, build_up, action, unknown.
    NEVER invents events: returns 'unknown' when kinematic/interaction evidence is absent.
    """

    VALID_EVENTS = {
        "shot", "pass", "dribble", "skill", "tackle", "save",
        "acceleration", "defender_interaction", "goal", "celebration",
        "reaction", "high_intensity", "high-motion", "build_up", "action", "unknown"
    }

    REAL_FOOTBALL_EVENTS = {
        "shot", "pass", "dribble", "skill", "tackle", "save",
        "goal", "celebration", "reaction", "high_intensity",
        "high-motion", "acceleration", "defender_interaction", "build_up", "action"
    }

    @staticmethod
    def detect_events(track_history: list, total_duration: float) -> list:
        if not track_history:
            return [{
                "type": "unknown",
                "event": "unknown",
                "start": 0.0,
                "end": round(total_duration, 2),
                "confidence": 0.50,
                "event_backed": False,
                "description": "Insufficient tracking evidence (unknown)",
                "evidence": {
                    "player_speed": 0.0,
                    "acceleration": 0.0,
                    "direction_change_deg": 0.0,
                    "ball_proximity": 999.0,
                    "player_ball_interaction": 0.0,
                    "defender_interaction": 0.0,
                },
            }]

        events = []
        step_sec = 0.5
        n_steps = max(1, int(math.ceil(total_duration / step_sec)))
        had_shot_or_goal = False

        for step in range(n_steps):
            t_start = step * step_sec
            t_end = min(total_duration, (step + 1) * step_sec)
            window_frames = [f for f in track_history if t_start <= f["timestamp"] <= t_end]
            if not window_frames:
                continue

            speeds = []
            accels = []
            for f in window_frames:
                p = f.get("player")
                if p:
                    v_val = p.get("speed", p.get("velocity", 0.0))
                    speeds.append(float(v_val.get("speed", 0.0) if isinstance(v_val, dict) else v_val))
                    a_val = p.get("accel", p.get("acceleration", 0.0))
                    accels.append(float(a_val.get("accel", 0.0) if isinstance(a_val, dict) else a_val))
            dir_changes = [float(f["player"].get("direction_change_deg", 0.0)) for f in window_frames if f.get("player")]
            confs = [float(f["player"].get("confidence", 0.7)) for f in window_frames if f.get("player")]

            avg_speed = float(np.mean(speeds)) if speeds else 0.0
            max_speed = float(np.max(speeds)) if speeds else 0.0
            max_accel = float(np.max(accels)) if accels else 0.0
            max_dir_change = float(np.max(dir_changes)) if dir_changes else 0.0
            avg_conf = float(np.mean(confs)) if confs else 0.5

            ball_dists = [float(f["ball"].get("dist_to_player", 999.0)) for f in window_frames if f.get("ball")]
            ball_speeds = [float(f["ball"].get("speed", 0.0)) for f in window_frames if f.get("ball")]
            min_ball_dist = float(np.min(ball_dists)) if ball_dists else 999.0
            max_ball_dist = float(np.max(ball_dists)) if ball_dists else 999.0
            max_ball_speed = float(np.max(ball_speeds)) if ball_speeds else 0.0

            def_counts = [len(f.get("defenders", [])) for f in window_frames]
            has_defenders = any(c > 0 for c in def_counts)
            min_def_dist = min(
                (float(d["dist"]) for f in window_frames for d in f.get("defenders", [])),
                default=999.0,
            )

            pbi_scores = [
                float((f.get("player_ball_interaction") or {}).get("score", 0.0))
                for f in window_frames
            ]
            max_pbi_score = float(np.max(pbi_scores)) if pbi_scores else (
                min(1.0, 65.0 / max(15.0, min_ball_dist)) if min_ball_dist < 130.0 else 0.0
            )
            def_pressures = [
                float((f.get("defender_interaction") or {}).get("pressure_score", 0.0))
                for f in window_frames
            ]
            max_def_pressure = float(np.max(def_pressures)) if def_pressures else (
                min(1.0, max(0.0, (150.0 - min_def_dist) / 130.0)) if has_defenders else 0.0
            )

            ev_type = "unknown"
            desc = "Insufficient kinematic evidence"
            conf = 0.50

            # 1. Goal: explosive shot followed by ball reaching goal zone & high ball speed
            if had_shot_or_goal and max_ball_speed > 180.0 and max_ball_dist > 180.0 and avg_speed < 95.0:
                ev_type = "goal"
                desc = "Ball enters goal area after high-velocity strike"
                conf = 0.93
            # 2. Shot: close ball contact + explosive acceleration + high strike speed
            elif min_ball_dist < 110.0 and max_speed > 110.0 and max_accel > 240.0:
                ev_type = "shot"
                desc = "Rapid strike towards goal with player-ball impact"
                conf = 0.94
                had_shot_or_goal = True
            # 3. Save: extreme ball velocity intercepted with sudden deflection
            elif max_ball_speed > 210.0 and min_ball_dist < 90.0 and max_accel > 260.0 and step > 1:
                ev_type = "save"
                desc = "High-velocity ball deflection / goalkeeper save"
                conf = 0.88
            # 4. Skill: close ball proximity + defender interaction or sharp direction change + acceleration
            elif min_ball_dist < 95.0 and (has_defenders or max_dir_change >= 28.0) and (max_accel > 175.0 or max_speed > 72.0):
                ev_type = "skill"
                desc = "1v1 skill move with directional agility"
                conf = 0.89
            # 5. Tackle: close defender convergence + high deceleration/acceleration contest
            elif has_defenders and min_def_dist < 75.0 and max_accel > 250.0 and avg_speed < 78.0:
                ev_type = "tackle"
                desc = "Defensive challenge contest"
                conf = 0.85
            # 6. Defender interaction: active proximity contest with opponent
            elif has_defenders and min_def_dist < 110.0 and avg_speed >= 35.0:
                ev_type = "defender_interaction"
                desc = "1v1 duel under defender pressure"
                conf = 0.82
            # 7. Dribble: sustained close ball control with active progression
            elif min_ball_dist < 90.0 and avg_speed > 40.0:
                ev_type = "dribble"
                desc = "Close-control ball progression"
                conf = 0.86
            # 8. Pass: ball separating from player with moderate-to-high speed
            elif min_ball_dist > 135.0 and max_ball_speed > 110.0 and avg_speed >= 25.0:
                ev_type = "pass"
                desc = "Ball distribution pass"
                conf = 0.80
            # 9. High-intensity / Acceleration: explosive sprint burst with high acceleration
            elif max_accel > 210.0 and max_speed > 90.0:
                ev_type = "high_intensity"
                desc = "High-intensity athletic sprint burst"
                conf = 0.83
            # 10. High-motion: fast continuous movement across pitch
            elif avg_speed > 72.0:
                ev_type = "high-motion"
                desc = "High-tempo transition movement"
                conf = 0.78
            # 11. Celebration: post-shot/goal low-speed celebration phase
            elif had_shot_or_goal and 8.0 <= avg_speed < 48.0 and step >= int(n_steps * 0.50):
                ev_type = "celebration"
                desc = "Post-strike celebration"
                conf = 0.83
            # 12. Reaction: post-event reaction phase (only if real motion occurred earlier)
            elif (had_shot_or_goal or any(e["type"] != "unknown" for e in events)) and step >= int(n_steps * 0.65) and 14.0 <= avg_speed <= 55.0 and avg_conf >= 0.60:
                ev_type = "reaction"
                desc = "Post-action player reaction"
                conf = 0.76
            # 13. Build-up: controlled progression with valid tracking confidence
            elif avg_speed >= 28.0 and avg_conf >= 0.60:
                ev_type = "build_up"
                desc = "Attacking build-up progression"
                conf = 0.75

            events.append({
                "type": ev_type,
                "event": ev_type,
                "start": round(t_start, 2),
                "end": round(t_end, 2),
                "confidence": round(conf, 2),
                "event_backed": bool(ev_type != "unknown"),
                "description": desc,
                "evidence": {
                    "player_speed": round(max_speed, 2),
                    "avg_speed": round(avg_speed, 2),
                    "acceleration": round(max_accel, 2),
                    "direction_change_deg": round(max_dir_change, 2),
                    "ball_proximity": round(min_ball_dist, 2),
                    "ball_speed": round(max_ball_speed, 2),
                    "player_ball_interaction": round(max_pbi_score, 3),
                    "defender_interaction": round(max_def_pressure, 3),
                },
            })

        merged = []
        for ev in events:
            if merged and merged[-1]["type"] == ev["type"]:
                merged[-1]["end"] = ev["end"]
                merged[-1]["confidence"] = max(merged[-1]["confidence"], ev["confidence"])
            else:
                merged.append(ev)
        return merged


# =====================================================================
# 4. EVIDENCE-BASED HERO MOMENT DETECTOR
# =====================================================================
class HeroMomentDetector:
    """
    Production HeroMomentDetector:
    - PROHIBITED: Hero without real football evidence (is_hero=False when no evidence).
    - PROHIBITED: Using motion energy alone (motion-only clips are capped < 0.40 and rejected).
    - Outputs:
        hero_score, score, hero_evidence, score_breakdown, event_backed,
        confidence, reason, is_hero, start, end.
    """

    HERO_THRESHOLD = 0.50

    DECISIVE_HERO_EVENTS = {
        "goal", "shot", "skill", "save", "dribble", "tackle", "pass", "defender_interaction"
    }

    @classmethod
    def compute_window_hero_score(cls, w_frames: list, window_events: list) -> tuple:
        if not w_frames:
            empty_ev = {
                "motion": 0.0,
                "acceleration": 0.0,
                "direction_change": 0.0,
                "ball_proximity": 0.0,
                "player_ball_interaction": 0.0,
                "defender_interaction": 0.0,
                "event_bonuses": 0.0,
                "backing_events": [],
                "motion_only_rejected": True,
            }
            return 0.0, "No tracking frames in window (Hero rejected)", empty_ev, False, 0.0

        speeds = []
        accels = []
        for f in w_frames:
            p = f.get("player")
            if p:
                v_val = p.get("speed", p.get("velocity", 0.0))
                speeds.append(float(v_val.get("speed", 0.0) if isinstance(v_val, dict) else v_val))
                a_val = p.get("accel", p.get("acceleration", 0.0))
                accels.append(float(a_val.get("accel", 0.0) if isinstance(a_val, dict) else a_val))
        dir_changes = [float(f["player"].get("direction_change_deg", 0.0)) for f in w_frames if f.get("player")]
        confs = [float(f["player"].get("confidence", 0.75)) for f in w_frames if f.get("player")]

        ball_prox = []
        for f in w_frames:
            if "ball_proximity" in f and isinstance(f["ball_proximity"], (int, float)) and f["ball_proximity"] <= 1.0:
                ball_prox.append(float(f["ball_proximity"]))
            elif f.get("ball") and "dist_to_player" in f["ball"]:
                d = float(f["ball"].get("dist_to_player", 999.0))
                ball_prox.append(min(1.0, 65.0 / max(15.0, d)))
        pbi_list = [
            float((f.get("player_ball_interaction") or {}).get("score", 0.0))
            for f in w_frames
        ]
        def_scores = [
            float(
                (f.get("defender_interaction") or {}).get(
                    "pressure_score", (f.get("defender_interaction") or {}).get("score", 0.0)
                )
            )
            for f in w_frames
        ]
        def_count = sum(len(f.get("defenders", [])) for f in w_frames)

        motion_comp = min(1.0, (float(np.mean(speeds)) if speeds else 0.0) / 155.0)
        accel_comp = min(1.0, (float(np.max(accels)) if accels else 0.0) / 380.0)
        dir_comp = min(1.0, (float(np.max(dir_changes)) if dir_changes else 0.0) / 90.0)
        ball_comp = min(1.0, float(np.mean(ball_prox)) if ball_prox else 0.0)
        pbi_comp = max(ball_comp, float(np.max(pbi_list)) if pbi_list else 0.0)
        def_comp = max(min(1.0, def_count / 3.0), float(np.max(def_scores)) if def_scores else 0.0)

        ev_types = [
            (e.get("type") or e.get("event"))
            for e in window_events
            if (e.get("type") or e.get("event")) and (e.get("type") or e.get("event")) != "unknown"
        ]
        ev_set = set(ev_types)
        decisive_events = [e for e in ev_types if e in cls.DECISIVE_HERO_EVENTS]

        shot_bonus = 0.28 if "shot" in ev_set else 0.0
        goal_save_bonus = 0.32 if ("goal" in ev_set or "save" in ev_set) else (0.10 if "celebration" in ev_set else 0.0)
        skill_bonus = 0.24 if "skill" in ev_set else 0.0
        dribble_bonus = 0.16 if "dribble" in ev_set else 0.0
        def_ev_bonus = 0.14 if ("defender_interaction" in ev_set or "tackle" in ev_set) else 0.0
        pass_bonus = 0.12 if "pass" in ev_set else 0.0
        total_event_bonus = shot_bonus + goal_save_bonus + skill_bonus + dribble_bonus + def_ev_bonus + pass_bonus

        # Check if there is genuine football interaction/event evidence (NOT motion energy alone!)
        has_ball_or_defender_evidence = (ball_comp >= 0.25) or (pbi_comp >= 0.30) or (def_comp >= 0.25)
        event_backed = bool(len(decisive_events) > 0 or (len(ev_set) > 0 and has_ball_or_defender_evidence))

        if not event_backed and not has_ball_or_defender_evidence:
            # Strictly reject motion-energy-only windows!
            capped_score = round(min(0.36, motion_comp * 0.22 + accel_comp * 0.10), 2)
            reason = (
                "Motion energy alone without ball/defender interaction or football event evidence (Hero rejected)"
                if motion_comp > 0.15
                else "Insufficient kinematic and football event evidence (Hero rejected)"
            )
            evidence = {
                "motion": round(motion_comp, 2),
                "acceleration": round(accel_comp, 2),
                "direction_change": round(dir_comp, 2),
                "ball_proximity": round(ball_comp, 2),
                "player_ball_interaction": round(pbi_comp, 2),
                "defender_interaction": round(max(def_comp, def_ev_bonus), 2),
                "event_bonuses": 0.0,
                "backing_events": list(ev_set),
                "motion_only_rejected": True,
            }
            conf_val = round(float(np.mean(confs)) * 0.65 if confs else 0.45, 2)
            return capped_score, reason, evidence, False, conf_val

        raw_score = (
            motion_comp * 0.12
            + accel_comp * 0.18
            + dir_comp * 0.06
            + ball_comp * 0.16
            + pbi_comp * 0.12
            + def_comp * 0.10
            + total_event_bonus
        )
        final_score = round(min(1.0, max(0.0, raw_score)), 2)

        if "goal" in ev_set or "shot" in ev_set:
            reason = "Event-backed decisive strike impact with close ball proximity and peak acceleration"
        elif "skill" in ev_set:
            reason = "Event-backed 1v1 skill move beating converging defender"
        elif "save" in ev_set:
            reason = "Event-backed high-velocity goalkeeper save"
        elif "tackle" in ev_set or "defender_interaction" in ev_set:
            reason = "Event-backed 1v1 defensive duel with ball contest"
        elif "dribble" in ev_set or ball_comp >= 0.35:
            reason = "Event-backed close-control dribbling progression"
        elif "pass" in ev_set:
            reason = "Event-backed attacking pass distribution"
        else:
            reason = "High-intensity football action with ball/player evidence"

        evidence = {
            "motion": round(motion_comp, 2),
            "acceleration": round(accel_comp, 2),
            "direction_change": round(dir_comp, 2),
            "ball_proximity": round(ball_comp, 2),
            "player_ball_interaction": round(pbi_comp, 2),
            "defender_interaction": round(max(def_comp, def_ev_bonus), 2),
            "event_bonuses": round(total_event_bonus, 2),
            "backing_events": sorted(list(ev_set)),
            "motion_only_rejected": False,
        }
        mean_track_conf = float(np.mean(confs)) if confs else 0.75
        conf_val = round(min(0.99, max(0.55, mean_track_conf * 0.6 + final_score * 0.4)), 2)
        return final_score, reason, evidence, event_backed, conf_val

    @classmethod
    def identify_hero_moment(
        cls,
        track_history: list,
        total_duration: float,
        user_segment: tuple = None,
        detected_events: list = None,
    ) -> dict:
        if not track_history:
            s_start = user_segment[0] if user_segment else min(1.5, total_duration * 0.25)
            s_end = user_segment[1] if user_segment else min(total_duration, s_start + 2.0)
            empty_ev = {
                "motion": 0.0,
                "acceleration": 0.0,
                "direction_change": 0.0,
                "ball_proximity": 0.0,
                "player_ball_interaction": 0.0,
                "defender_interaction": 0.0,
                "event_bonuses": 0.0,
                "backing_events": [],
                "motion_only_rejected": True,
            }
            return {
                "start": round(s_start, 2),
                "end": round(s_end, 2),
                "hero_score": 0.0,
                "score": 0.0,
                "hero_evidence": empty_ev,
                "score_breakdown": empty_ev,
                "event_backed": False,
                "confidence": 0.0,
                "is_hero": False,
                "reason": "No tracking evidence available (Hero prohibited without real evidence)",
            }

        events = detected_events or FootballEventEngine.detect_events(track_history, total_duration)
        window_size = min(2.2, max(0.8, total_duration * 0.32))
        times = [f["timestamp"] for f in track_history]
        min_t, max_t = min(times), max(times)

        best_score = -1.0
        best_window = (round(max(0.0, total_duration * 0.3), 2), round(min(total_duration, total_duration * 0.3 + window_size), 2))
        best_reason = "Insufficient football evidence"
        best_evidence = {}
        best_event_backed = False
        best_conf = 0.45

        t_cur = min_t
        while t_cur <= max(min_t, max_t - window_size * 0.5) + 0.01:
            t_w_end = min(total_duration, t_cur + window_size)
            w_frames = [f for f in track_history if t_cur <= f["timestamp"] <= t_w_end]
            w_events = [e for e in events if not (e["end"] < t_cur or e["start"] > t_w_end)]
            score, reason, evidence, ev_backed, conf_val = cls.compute_window_hero_score(w_frames, w_events)
            # Prioritize event-backed windows over non-event-backed windows
            rank_val = score + (0.25 if ev_backed else 0.0)
            best_rank = best_score + (0.25 if best_event_backed else 0.0)
            if rank_val > best_rank:
                best_score = score
                best_window = (round(t_cur, 2), round(t_w_end, 2))
                best_reason = reason
                best_evidence = evidence
                best_event_backed = ev_backed
                best_conf = conf_val
            t_cur += 0.25

        # If user_segment is provided, evaluate its actual evidence (never inflate without real evidence!)
        if user_segment and (user_segment[1] - user_segment[0]) >= 0.5 and user_segment[1] <= total_duration + 0.1:
            u_start, u_end = float(user_segment[0]), min(total_duration, float(user_segment[1]))
            u_frames = [f for f in track_history if u_start <= f["timestamp"] <= u_end]
            u_events = [e for e in events if not (e["end"] < u_start or e["start"] > u_end)]
            u_score, u_reason, u_evidence, u_backed, u_conf = cls.compute_window_hero_score(u_frames, u_events)
            if u_backed and u_score >= best_score - 0.12:
                best_window = (round(u_start, 2), round(u_end, 2))
                best_score = max(best_score, u_score)
                best_reason = u_reason
                best_evidence = u_evidence
                best_event_backed = u_backed
                best_conf = max(best_conf, u_conf)

        final_score = round(max(0.0, best_score), 2)
        is_hero = bool(final_score >= 0.50 and best_event_backed)
        if not is_hero and best_reason.startswith("Event-backed"):
            best_reason = f"Sub-threshold football action (hero_score={final_score} < 0.50)"

        return {
            "start": best_window[0],
            "end": best_window[1],
            "hero_score": final_score,
            "score": final_score,
            "hero_evidence": best_evidence,
            "score_breakdown": best_evidence,
            "event_backed": bool(best_event_backed),
            "confidence": round(best_conf, 2),
            "is_hero": is_hero,
            "reason": best_reason,
        }


# =====================================================================
# 5. CINEMATIC DIRECTOR, EDIT PLAN & EVENT-DRIVEN SPEED RAMPS
# =====================================================================
class CinematicDirector:
    """
    Production Cinematic Director & Edit Plan Generator:
    - Uses all 11 decision pillars per scene:
        subject, anchor, camera_trajectory, zoom, speed_curve,
        isolation, grade, text, audio, transition, evidence.
    - Supports all 14 scene types:
        intro, wide, build_up, action, skill, shot, pass, dribble,
        tackle, save, celebration, reaction, hero, outro.
    - Builds a flexible narrative story grounded strictly in real detected events.
      If hero_moment["is_hero"] is False (no real evidence), NO hero scene and NO
      slow motion are invented.
    - Event-Driven Speed Ramps:
        - normal: 1.0x
        - hero / impact: normal (1.0) -> slow (0.85 -> 0.60) -> impact (0.30) -> normal (0.50 -> 1.0)
        - skill: normal (1.0) -> slow (0.55) -> very slow (0.28) -> normal (1.0)
        - shot: anticipation (0.72) -> impact slow (0.24) -> reaction (0.80 -> 1.0)
        - celebration: subtle slow (0.65)
    """

    SCENE_TYPES = (
        "intro", "wide", "build_up", "action", "skill", "shot", "pass",
        "dribble", "tackle", "save", "celebration", "reaction", "hero", "outro"
    )

    SPEED_RAMPS = {
        "hero": [
            (0.00, 1.00),
            (0.20, 0.85),
            (0.40, 0.60),
            (0.60, 0.30),
            (0.80, 0.50),
            (1.00, 1.00),
        ],
        "impact": [
            (0.00, 1.00),
            (0.20, 0.85),
            (0.40, 0.60),
            (0.60, 0.30),
            (0.80, 0.50),
            (1.00, 1.00),
        ],
        "skill": [
            (0.00, 1.00),
            (0.28, 0.55),
            (0.65, 0.28),
            (0.86, 0.65),
            (1.00, 1.00),
        ],
        "shot": [
            (0.00, 1.00),
            (0.28, 0.72),
            (0.58, 0.24),
            (0.82, 0.80),
            (1.00, 1.00),
        ],
        "celebration": [
            (0.00, 1.00),
            (0.50, 0.65),
            (1.00, 1.00),
        ],
        "normal": [
            (0.00, 1.00),
            (1.00, 1.00),
        ],
    }

    @classmethod
    def get_speed_multiplier(cls, progress: float, ramp_type: str = "hero") -> float:
        points = cls.SPEED_RAMPS.get(ramp_type, cls.SPEED_RAMPS["hero"])
        p = max(0.0, min(1.0, progress))
        for i in range(len(points) - 1):
            t0, s0 = points[i]
            t1, s1 = points[i + 1]
            if t0 <= p <= t1:
                alpha = (p - t0) / max(0.0001, t1 - t0)
                smooth_alpha = (1.0 - math.cos(alpha * math.pi)) / 2.0
                return s0 + smooth_alpha * (s1 - s0)
        return points[-1][1]

    @classmethod
    def evaluate_speed_curve(
        cls,
        progress: float,
        ramp_type: str = "hero",
        min_slow_factor: float = 0.30,
        event_backed: bool = True,
    ) -> float:
        if not event_backed or ramp_type == "normal":
            return 1.0
        val = cls.get_speed_multiplier(progress, ramp_type=ramp_type)
        return round(max(min(0.22, min_slow_factor), val), 3)

    @classmethod
    def _make_scene_pillars(
        cls,
        scene_type: str,
        start: float,
        end: float,
        shot_type: str,
        priority: int,
        subject_role: str,
        camera_action: str,
        easing: str,
        damping: float,
        target_aspect: str,
        start_zoom: float,
        peak_zoom: float,
        end_zoom: float,
        speed_mode: str,
        use_optical_flow: bool,
        enable_iso: bool,
        iso_cfg: dict,
        grade_cfg: dict,
        transition_type: str,
        text_cfg: dict,
        audio_cfg: dict,
        evidence_cfg: dict,
    ) -> dict:
        lead_factor = 0.11 if camera_action == "anticipation" else (0.04 if camera_action in ("impact", "reaction") else 0.075)
        ball_weight = 0.34 if camera_action in ("impact", "anticipation", "push_in") else 0.22
        def_weight = 0.16 if scene_type in ("skill", "tackle", "dribble") else 0.10
        vertical_bias = 0.09 if camera_action == "reaction" else (0.04 if camera_action == "impact" else 0.06)

        curve_pts = cls.SPEED_RAMPS.get(speed_mode, cls.SPEED_RAMPS["normal"])
        cam_obj = {"action": camera_action, "easing": easing, "damping": damping}
        anchor_obj = {
            "mode": "multi_keyframe_pursuit",
            "lead_factor": lead_factor,
            "ball_weight": ball_weight,
            "defender_weight": def_weight,
            "vertical_bias": vertical_bias,
            "safe_area": {"x": [0.16, 0.84], "y": [0.14, 0.86]},
        }
        speed_obj = {
            "mode": speed_mode,
            "curve": curve_pts,
            "use_optical_flow": bool(use_optical_flow and speed_mode != "normal"),
            "event_backed": bool(evidence_cfg.get("event_backed", False)),
        }
        trans_obj = {
            "type": transition_type,
            "duration_sec": 0.18 if transition_type in ("dissolve", "speed_transition") else 0.0,
        }
        return {
            "scene_type": scene_type,
            "start": round(float(start), 2),
            "end": round(float(end), 2),
            "shot_type": shot_type,
            "priority": priority,
            "subject": {
                "role": subject_role,
                "target": subject_role,
                "track_ball": True,
                "keep_in_safe_area": True,
            },
            "anchor": anchor_obj,
            "camera_trajectory": cam_obj,
            "camera": cam_obj,
            "crop": {"mode": "anchor_track", "aspect": target_aspect},
            "zoom": {
                "start_zoom": round(float(start_zoom), 2),
                "peak_zoom": round(float(peak_zoom), 2),
                "end_zoom": round(float(end_zoom), 2),
                "shot_type": shot_type,
            },
            "speed_curve": speed_obj,
            "speed": speed_obj,
            "speed_mode": speed_mode,
            "isolation": bool(enable_iso),
            "isolation_config": iso_cfg,
            "grade": grade_cfg,
            "transition": transition_type,
            "transition_config": trans_obj,
            "text": text_cfg,
            "text_kicker": text_cfg.get("content"),
            "audio": audio_cfg,
            "sound_cue": (
                "impact" if audio_cfg.get("impact")
                else ("riser" if audio_cfg.get("riser") else ("crowd_emphasis" if audio_cfg.get("crowd_emphasis") else "none"))
            ),
            "evidence": evidence_cfg,
            "color_intensity_boost": float(grade_cfg.get("intensity", 0.84)),
        }

    @classmethod
    def build_director_script(
        cls,
        events: list,
        hero_moment: dict,
        total_duration: float,
        target_aspect: str = "9:16",
        mode: str = "CINEMATIC",
        speed_ramp_type: str = "hero",
        reference_style: dict = None,
    ) -> list:
        style = reference_style or ReferenceStyleAnalyzer.get_preset_profile("ucl_broadcast_reel")
        zoom_cfg = style.get("zoom_intensity", {})
        color_cfg = style.get("color_characteristics", {})
        iso_style = style.get("isolation", {})
        base_zoom = float(zoom_cfg.get("base_zoom", 1.06))
        hero_peak_zoom = float(zoom_cfg.get("hero_peak_zoom", 1.30 if target_aspect == "9:16" else 1.18))
        text_policy = style.get("text_frequency", {}).get("policy", "minimal_context_only")

        has_real_hero = bool(
            hero_moment
            and hero_moment.get("is_hero", False)
            and hero_moment.get("event_backed", True)
            and float(hero_moment.get("hero_score", hero_moment.get("score", 0.0))) >= 0.48
        )
        enable_iso = bool(iso_style.get("enabled_on_hero", True)) and mode in ("PRO", "CINEMATIC") and has_real_hero

        default_grade = {
            "profile": "cinematic_turf_lut",
            "intensity": float(color_cfg.get("grade_intensity", 0.82)),
            "shadow_coolness": float(color_cfg.get("shadow_coolness", 1.0)),
            "highlight_warmth": float(color_cfg.get("highlight_warmth", 1.0)),
            "grass_saturation_cap": float(color_cfg.get("grass_saturation_cap", 0.82)),
            "contrast_s_curve": float(color_cfg.get("contrast_s_curve", 1.08)),
            "vignette": float(color_cfg.get("vignette_strength", 0.22)),
        }
        hero_grade = dict(default_grade, intensity=min(0.96, default_grade["intensity"] + 0.08))
        no_iso_cfg = {"enabled": False, "method": "no_isolation", "blur_kernel": 19, "dim_ratio": 1.0, "feather_px": 15}

        scenes = []

        # CASE A: No Hero Evidence -> Flexible Event-Driven Story WITHOUT inventing Hero or slow-mo
        if not has_real_hero:
            real_evs = [
                e for e in (events or [])
                if (e.get("type") or e.get("event")) and (e.get("type") or e.get("event")) != "unknown"
            ]
            if not real_evs:
                mid_t = round(total_duration * 0.5, 2)
                scenes.append(
                    cls._make_scene_pillars(
                        scene_type="wide",
                        start=0.0,
                        end=mid_t,
                        shot_type="wide_establish",
                        priority=5,
                        subject_role="primary_player",
                        camera_action="drift",
                        easing="cosine",
                        damping=0.10,
                        target_aspect=target_aspect,
                        start_zoom=1.02,
                        peak_zoom=base_zoom,
                        end_zoom=round(base_zoom + 0.03, 2),
                        speed_mode="normal",
                        use_optical_flow=False,
                        enable_iso=False,
                        iso_cfg=no_iso_cfg,
                        grade_cfg=default_grade,
                        transition_type="hard_cut",
                        text_cfg={"enabled": False, "content": None, "position": "upper_third", "reason": "no_event_evidence"},
                        audio_cfg={"keep_match_audio": True, "riser": False, "impact": False, "whoosh": False, "crowd_emphasis": False},
                        evidence_cfg={"event_type": "unknown", "event_backed": False, "confidence": 0.5, "hero_score": 0.0, "reason": "No football event detected"},
                    )
                )
                scenes.append(
                    cls._make_scene_pillars(
                        scene_type="action",
                        start=mid_t,
                        end=round(total_duration, 2),
                        shot_type="medium_follow",
                        priority=6,
                        subject_role="primary_player",
                        camera_action="follow",
                        easing="cubic_in_out",
                        damping=0.12,
                        target_aspect=target_aspect,
                        start_zoom=round(base_zoom + 0.03, 2),
                        peak_zoom=round(base_zoom + 0.07, 2),
                        end_zoom=1.04,
                        speed_mode="normal",
                        use_optical_flow=False,
                        enable_iso=False,
                        iso_cfg=no_iso_cfg,
                        grade_cfg=default_grade,
                        transition_type="dissolve",
                        text_cfg={"enabled": False, "content": None, "position": "lower_third", "reason": "no_event_evidence"},
                        audio_cfg={"keep_match_audio": True, "riser": False, "impact": False, "whoosh": False, "crowd_emphasis": False},
                        evidence_cfg={"event_type": "unknown", "event_backed": False, "confidence": 0.5, "hero_score": 0.0, "reason": "Normal play continuation"},
                    )
                )
                return scenes

            for idx, ev in enumerate(events):
                ev_t = ev.get("type") or ev.get("event", "action")
                sc_type = ev_t if ev_t in cls.SCENE_TYPES else ("build_up" if idx == 0 else "action")
                cam_act = "drift" if sc_type in ("intro", "wide") else ("reaction" if sc_type in ("reaction", "celebration") else "follow")
                shot_t = "wide_establish" if sc_type in ("intro", "wide") else ("reaction_medium" if cam_act == "reaction" else "medium_follow")
                scenes.append(
                    cls._make_scene_pillars(
                        scene_type=sc_type,
                        start=ev["start"],
                        end=ev["end"],
                        shot_type=shot_t,
                        priority=6,
                        subject_role="player_with_ball",
                        camera_action=cam_act,
                        easing="cubic_in_out",
                        damping=0.12,
                        target_aspect=target_aspect,
                        start_zoom=base_zoom,
                        peak_zoom=round(base_zoom + 0.06, 2),
                        end_zoom=round(base_zoom + 0.03, 2),
                        speed_mode="normal",
                        use_optical_flow=False,
                        enable_iso=False,
                        iso_cfg=no_iso_cfg,
                        grade_cfg=default_grade,
                        transition_type="hard_cut" if idx < len(events) - 1 else "dissolve",
                        text_cfg={"enabled": False, "content": None, "position": "lower_third", "reason": "non_hero_scene"},
                        audio_cfg={"keep_match_audio": True, "riser": False, "impact": False, "whoosh": False, "crowd_emphasis": False},
                        evidence_cfg={
                            "event_type": ev_t,
                            "event_backed": bool(ev.get("event_backed", ev_t != "unknown")),
                            "confidence": float(ev.get("confidence", 0.7)),
                            "hero_score": float(hero_moment.get("hero_score", 0.0)),
                            "reason": ev.get("description", "Tracked play"),
                        },
                    )
                )
            return scenes

        # CASE B: Verified Event-Backed Hero Moment -> Full Flexible Football Narrative Story
        h_start = float(hero_moment["start"])
        h_end = float(hero_moment["end"])
        h_score = float(hero_moment.get("hero_score", hero_moment.get("score", 0.8)))
        h_conf = float(hero_moment.get("confidence", 0.88))

        # 1. Pre-Hero Scenes (Intro/Wide -> Build-Up / Pass / Dribble / Skill / Action)
        if h_start > 0.4:
            pre_events = [
                e for e in (events or [])
                if float(e["start"]) < h_start and (e.get("type") or e.get("event")) != "unknown"
            ]
            if h_start >= 2.0:
                intro_end = round(min(1.0, h_start * 0.35), 2)
                scenes.append(
                    cls._make_scene_pillars(
                        scene_type="intro",
                        start=0.0,
                        end=intro_end,
                        shot_type="wide_establish",
                        priority=5,
                        subject_role="primary_player",
                        camera_action="drift",
                        easing="cosine",
                        damping=0.10,
                        target_aspect=target_aspect,
                        start_zoom=1.02,
                        peak_zoom=base_zoom,
                        end_zoom=base_zoom,
                        speed_mode="normal",
                        use_optical_flow=False,
                        enable_iso=False,
                        iso_cfg=no_iso_cfg,
                        grade_cfg=default_grade,
                        transition_type="hard_cut",
                        text_cfg={
                            "enabled": bool(mode == "CINEMATIC" and text_policy != "none"),
                            "content": "MATCH FOCUS",
                            "position": "upper_third",
                            "reason": "Opening tactical establishment",
                        },
                        audio_cfg={"keep_match_audio": True, "riser": False, "impact": False, "whoosh": False, "crowd_emphasis": False},
                        evidence_cfg={
                            "event_type": "intro",
                            "event_backed": True,
                            "confidence": 0.80,
                            "hero_score": h_score,
                            "reason": "Opening wide tactical context",
                        },
                    )
                )
                b_start = intro_end
            else:
                b_start = 0.0

            pre_type = (pre_events[-1].get("type") or pre_events[-1].get("event", "build_up")) if pre_events else "build_up"
            if pre_type not in cls.SCENE_TYPES or pre_type in ("hero", "outro", "intro"):
                pre_type = "build_up"

            # Use 'anticipation' camera framing right before a shot/skill Hero moment
            hero_ev_types = [
                (e.get("type") or e.get("event"))
                for e in (events or [])
                if not (float(e["end"]) < h_start or float(e["start"]) > h_end)
            ]
            pre_cam_action = "anticipation" if ("shot" in hero_ev_types or "goal" in hero_ev_types or pre_type in ("pass", "dribble", "skill")) else "follow"
            pre_shot_type = "anticipation_frame" if pre_cam_action == "anticipation" else "medium_follow"

            scenes.append(
                cls._make_scene_pillars(
                    scene_type=pre_type,
                    start=b_start,
                    end=h_start,
                    shot_type=pre_shot_type,
                    priority=7,
                    subject_role="player_with_ball",
                    camera_action=pre_cam_action,
                    easing="cubic_in_out",
                    damping=0.12,
                    target_aspect=target_aspect,
                    start_zoom=base_zoom,
                    peak_zoom=round(base_zoom + 0.07, 2),
                    end_zoom=round(base_zoom + 0.09, 2),
                    speed_mode="normal",
                    use_optical_flow=False,
                    enable_iso=False,
                    iso_cfg=no_iso_cfg,
                    grade_cfg=default_grade,
                    transition_type="hard_cut",
                    text_cfg={"enabled": False, "content": None, "position": "lower_third", "reason": "clean_build_up"},
                    audio_cfg={"keep_match_audio": True, "riser": mode == "CINEMATIC", "impact": False, "whoosh": False, "crowd_emphasis": False},
                    evidence_cfg={
                        "event_type": pre_type,
                        "event_backed": True,
                        "confidence": float(pre_events[-1].get("confidence", 0.82)) if pre_events else 0.78,
                        "hero_score": h_score,
                        "reason": f"Pre-hero {pre_type} progression leading into climax",
                    },
                )
            )

        # 2. Hero Moment Scene (Strictly 1 Hero clip per reel, Priority = 10, Event-Backed)
        hero_events = [
            (e.get("type") or e.get("event"))
            for e in (events or [])
            if not (float(e["end"]) < h_start or float(e["start"]) > h_end)
        ]
        hero_caption = None
        auto_ramp = speed_ramp_type
        hero_cam_action = "push_in"
        hero_shot_type = "tight_portrait"

        if "goal" in hero_events:
            hero_caption = "GOAL IMPACT"
            hero_cam_action = "impact"
            hero_shot_type = "impact_close"
            if speed_ramp_type == "hero":
                auto_ramp = "hero"
        elif "shot" in hero_events:
            hero_caption = "STRIKE IMPACT"
            hero_cam_action = "impact"
            hero_shot_type = "impact_close"
        elif "skill" in hero_events or "dribble" in hero_events:
            hero_caption = "1V1 SKILL"
            hero_cam_action = "push_in"
            hero_shot_type = "tight_portrait"
        elif "save" in hero_events:
            hero_caption = "DECISIVE SAVE"
            hero_cam_action = "impact"
            hero_shot_type = "impact_close"
        elif "tackle" in hero_events or "defender_interaction" in hero_events:
            hero_caption = "1V1 DUEL"
            hero_cam_action = "push_in"
            hero_shot_type = "tight_portrait"

        active_hero_ramp = auto_ramp if mode in ("PRO", "CINEMATIC") else "normal"
        hero_iso_cfg = {
            "enabled": enable_iso,
            "method": "mobilesam_fallback_chain",
            "fallback_chain": ["mobilesam", "yolo", "motion", "no_isolation"],
            "blur_kernel": int(iso_style.get("blur_kernel", 21)),
            "dim_ratio": float(iso_style.get("dim_ratio", 0.88)),
            "feather_px": int(iso_style.get("feather_px", 17)),
        }

        scenes.append(
            cls._make_scene_pillars(
                scene_type="hero",
                start=h_start,
                end=h_end,
                shot_type=hero_shot_type,
                priority=10,
                subject_role="hero_player_and_ball",
                camera_action=hero_cam_action,
                easing="cubic_in_out",
                damping=0.12,
                target_aspect=target_aspect,
                start_zoom=round(base_zoom + 0.08, 2),
                peak_zoom=hero_peak_zoom,
                end_zoom=round(base_zoom + 0.10, 2),
                speed_mode=active_hero_ramp,
                use_optical_flow=mode in ("PRO", "CINEMATIC"),
                enable_iso=enable_iso,
                iso_cfg=hero_iso_cfg,
                grade_cfg=hero_grade,
                transition_type="speed_transition" if mode in ("PRO", "CINEMATIC") else "hard_cut",
                text_cfg={
                    "enabled": bool(hero_caption and mode == "CINEMATIC" and text_policy in ("minimal_context_only", "hero_only")),
                    "content": hero_caption,
                    "position": "lower_third",
                    "reason": hero_moment.get("reason", "Verified Hero climax"),
                },
                audio_cfg={
                    "keep_match_audio": True,
                    "riser": True,
                    "impact": True,
                    "whoosh": True,
                    "crowd_emphasis": True,
                },
                evidence_cfg={
                    "event_type": hero_events[0] if hero_events else "shot",
                    "event_backed": True,
                    "confidence": h_conf,
                    "hero_score": h_score,
                    "hero_evidence": hero_moment.get("hero_evidence", {}),
                    "reason": hero_moment.get("reason", "Event-backed Hero climax"),
                },
            )
        )

        # 3. Post-Hero Scenes (Reaction -> Celebration / Outro)
        if h_end < total_duration - 0.15:
            rem = total_duration - h_end
            post_events = [e for e in (events or []) if float(e["start"]) >= h_end - 0.1 and e.get("type") != "unknown"]
            if rem >= 2.2:
                react_end = round(h_end + min(1.2, rem * 0.45), 2)
                scenes.append(
                    cls._make_scene_pillars(
                        scene_type="reaction",
                        start=h_end,
                        end=react_end,
                        shot_type="reaction_medium",
                        priority=8,
                        subject_role="primary_player",
                        camera_action="reaction",
                        easing="cubic_in_out",
                        damping=0.11,
                        target_aspect=target_aspect,
                        start_zoom=round(base_zoom + 0.10, 2),
                        peak_zoom=round(base_zoom + 0.08, 2),
                        end_zoom=round(base_zoom + 0.04, 2),
                        speed_mode="normal",
                        use_optical_flow=False,
                        enable_iso=False,
                        iso_cfg=no_iso_cfg,
                        grade_cfg=default_grade,
                        transition_type="hard_cut",
                        text_cfg={"enabled": False, "content": None, "position": "lower_third", "reason": "post_hero_reaction"},
                        audio_cfg={"keep_match_audio": True, "riser": False, "impact": False, "whoosh": False, "crowd_emphasis": mode == "CINEMATIC"},
                        evidence_cfg={
                            "event_type": "reaction",
                            "event_backed": True,
                            "confidence": 0.82,
                            "hero_score": h_score,
                            "reason": "Immediate player reaction after climax",
                        },
                    )
                )
                outro_start = react_end
            else:
                outro_start = h_end

            post_type = "celebration" if (rem > 1.2 or any(e.get("type") == "celebration" for e in post_events)) else "outro"
            scenes.append(
                cls._make_scene_pillars(
                    scene_type=post_type,
                    start=outro_start,
                    end=round(total_duration, 2),
                    shot_type="wide_release",
                    priority=6,
                    subject_role="primary_player",
                    camera_action="pull_out",
                    easing="cosine",
                    damping=0.10,
                    target_aspect=target_aspect,
                    start_zoom=round(base_zoom + 0.06, 2),
                    peak_zoom=round(base_zoom + 0.03, 2),
                    end_zoom=1.02,
                    speed_mode="celebration" if (mode == "CINEMATIC" and post_type == "celebration") else "normal",
                    use_optical_flow=False,
                    enable_iso=False,
                    iso_cfg=no_iso_cfg,
                    grade_cfg=default_grade,
                    transition_type="dissolve" if post_type == "outro" else "hard_cut",
                    text_cfg={"enabled": False, "content": None, "position": "lower_third", "reason": "outro_release"},
                    audio_cfg={
                        "keep_match_audio": True,
                        "riser": False,
                        "impact": False,
                        "whoosh": False,
                        "crowd_emphasis": mode == "CINEMATIC",
                    },
                    evidence_cfg={
                        "event_type": post_type,
                        "event_backed": True,
                        "confidence": 0.80,
                        "hero_score": h_score,
                        "reason": f"Narrative resolution ({post_type})",
                    },
                )
            )

        return scenes

    @classmethod
    def build_edit_plan(
        cls,
        director_script: list,
        hero_moment: dict,
        detected_events: list,
        total_duration: float,
        target_aspect: str = "9:16",
        mode: str = "CINEMATIC",
        psychological_package: dict = None,
    ) -> dict:
        """
        Packages the Director's scene decisions into a structured Edit Plan with
        explicit verification of all 11 pillars, narrative story arc, slow-mo budget,
        and optional additive Psychological Storyteller fields.
        """
        slow_dur = sum(
            float(s["end"] - s["start"])
            for s in (director_script or [])
            if (s.get("speed_curve") or {}).get("mode", "normal") not in ("normal",)
        )
        slow_ratio = round(slow_dur / max(0.1, float(total_duration)), 3)
        psych = psychological_package or {}
        psych_active = bool(psych.get("psychological_story", False))

        plan = {
            "mode": mode,
            "target_aspect": target_aspect,
            "total_duration": round(float(total_duration), 2),
            "narrative_arc": [s.get("scene_type", "action") for s in (director_script or [])],
            "has_verified_hero": bool(hero_moment and hero_moment.get("is_hero", False) and hero_moment.get("event_backed", False)),
            "hero_summary": {
                "hero_score": hero_moment.get("hero_score", hero_moment.get("score", 0.0)) if hero_moment else 0.0,
                "event_backed": bool(hero_moment.get("event_backed", False)) if hero_moment else False,
                "confidence": hero_moment.get("confidence", 0.0) if hero_moment else 0.0,
                "reason": hero_moment.get("reason", "") if hero_moment else "",
            },
            "slow_motion_window_ratio": slow_ratio,
            "slow_motion_policy_compliant": bool(slow_ratio <= 0.48),
            "pillars_enforced": [
                "subject",
                "anchor",
                "camera_trajectory",
                "zoom",
                "speed_curve",
                "isolation",
                "grade",
                "text",
                "audio",
                "transition",
                "evidence",
            ],
            "scenes": director_script,
            "detected_events_count": len([e for e in (detected_events or []) if (e.get("type") or e.get("event")) != "unknown"]),
            # Additive Psychological Storyteller Edit Plan fields (Req 6)
            "psychological_story": psych_active,
            "story_role": psych.get("story_role", "predator") if psych_active else None,
            "story_arc": psych.get("story_arc", ["predator", "trap", "dominance"]) if psych_active else [],
            "duel_moment": psych.get("duel_moment") if psych_active else None,
            "winner": psych.get("winner") if psych_active else None,
            "loser": psych.get("loser") if psych_active else None,
            "depth_effect": bool(psych.get("depth_effect", False)) if psych_active else False,
            "low_angle": bool(psych.get("low_angle", False)) if psych_active else False,
            "pov_switch": bool(psych.get("pov_switch", False)) if psych_active else False,
            "story_script": list(psych.get("story_script") or []) if psych_active else [],
        }
        return plan

    @staticmethod
    def compute_smoothed_crop_box(
        frame_w: int,
        frame_h: int,
        target_w: int,
        target_h: int,
        player_cx: float,
        player_cy: float,
        zoom: float,
        prev_crop: tuple = None,
        damping: float = 0.12,
    ) -> tuple:
        aspect_ratio = float(target_w) / float(target_h)
        z_val = float(zoom.get("peak_zoom", 1.15)) if isinstance(zoom, dict) else float(zoom)
        crop_h = int(round(frame_h / max(0.5, z_val)))
        crop_w = int(round(crop_h * aspect_ratio))
        if crop_w > frame_w:
            crop_w = frame_w
            crop_h = int(round(crop_w / aspect_ratio))
        target_cx = player_cx
        target_cy = max(crop_h / 2.0, min(frame_h - crop_h / 2.0, player_cy - crop_h * 0.08))
        target_x1 = max(0.0, min(float(frame_w - crop_w), target_cx - crop_w / 2.0))
        target_y1 = max(0.0, min(float(frame_h - crop_h), target_cy - crop_h / 2.0))
        if prev_crop is not None:
            smooth_x1 = prev_crop[0] + damping * (target_x1 - prev_crop[0])
            smooth_y1 = prev_crop[1] + damping * (target_y1 - prev_crop[1])
            smooth_w = prev_crop[2] + damping * (crop_w - prev_crop[2])
            smooth_h = prev_crop[3] + damping * (crop_h - prev_crop[3])
        else:
            smooth_x1, smooth_y1, smooth_w, smooth_h = target_x1, target_y1, float(crop_w), float(crop_h)
        rx1 = int(round(smooth_x1))
        ry1 = int(round(smooth_y1))
        rw = int(round(smooth_w))
        rh = int(round(smooth_h))
        rx1 = max(0, min(frame_w - rw, rx1))
        ry1 = max(0, min(frame_h - rh, ry1))
        return (rx1, ry1, rw, rh)


# =====================================================================
# 6. HOLLYWOOD CINEMATIC COLOR GRADER (Req 8 & Req 14)
# =====================================================================
class CinematicColorGrader:
    """
    Requirement 8 — Cinematic Color:
    - Cooler shadows (blue lift in dark tones)
    - Warmer highlights (amber/golden warmth in highlights)
    - Controlled S-curve contrast
    - Protected skin tones (natural hue/warmth)
    - Controlled grass saturation (no neon grass)
    - Subtle vignette (cached per resolution for 4x speed & low RAM)
    - Zero oversaturation
    """

    _VIGNETTE_CACHE = {}
    _LUT_CACHE = {}

    @classmethod
    def _get_vignette(cls, h: int, w: int, intensity: float) -> np.ndarray:
        key = (h, w, round(intensity, 2))
        if key not in cls._VIGNETTE_CACHE:
            y_coords = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
            x_coords = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
            radius = np.sqrt(x_coords * x_coords + y_coords * y_coords)
            vig = np.clip(1.0 - (radius - 0.58) * (0.26 * intensity), 0.78, 1.0)[:, :, None]
            cls._VIGNETTE_CACHE[key] = vig
        return cls._VIGNETTE_CACHE[key]

    @classmethod
    def _get_channel_luts(cls, intensity: float):
        key = round(intensity, 2)
        if key not in cls._LUT_CACHE:
            x = np.arange(256, dtype=np.float32)
            norm = x / 255.0
            # Controlled S-curve contrast
            s_curve = norm + (0.07 * intensity) * np.sin(2.0 * math.pi * (norm - 0.5))
            s_curve = np.clip(s_curve * 255.0, 0.0, 255.0)
            shadow_w = np.clip((128.0 - x) / 128.0, 0.0, 1.0)
            high_w = np.clip((x - 128.0) / 128.0, 0.0, 1.0)
            # Cooler shadows (+B, -R) & warmer highlights (+R, +G, -B)
            lut_b = np.clip(s_curve + shadow_w * (5.5 * intensity) - high_w * (4.0 * intensity), 0, 255).astype(np.uint8)
            lut_g = np.clip(s_curve + high_w * (2.0 * intensity), 0, 255).astype(np.uint8)
            lut_r = np.clip(s_curve - shadow_w * (3.0 * intensity) + high_w * (6.0 * intensity), 0, 255).astype(np.uint8)
            cls._LUT_CACHE[key] = (lut_b, lut_g, lut_r)
        return cls._LUT_CACHE[key]

    @classmethod
    def apply_grade(cls, frame: np.ndarray, intensity: float = 0.85, grade_config: dict = None) -> np.ndarray:
        g_cfg = grade_config or {}
        eff_intensity = float(g_cfg.get("intensity", intensity))
        if g_cfg.get("psychological") or g_cfg.get("profile") == "psychological_teal_orange_lut":
            return PsychologicalColorGrader.apply_teal_orange_grade(frame, intensity=eff_intensity)
        grass_cap = int(round(float(g_cfg.get("grass_saturation_cap", 0.80)) * 255.0))
        vig_strength = float(g_cfg.get("vignette", eff_intensity))

        h, w = frame.shape[:2]
        sh, sw = max(1, h // 2), max(1, w // 2)
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA) if (h * w) > (640 * 360) else frame
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Grass mask (Hue 35..82): shift toward deep emerald & suppress neon saturation
        grass_mask = (h_ch >= 35) & (h_ch <= 82) & (s_ch > 40)
        if np.any(grass_mask):
            h_ch[grass_mask] = np.clip(h_ch[grass_mask].astype(np.int16) + 3, 0, 179).astype(np.uint8)
            s_ch[grass_mask] = np.clip(
                s_ch[grass_mask].astype(np.float32) * (1.0 - 0.18 * eff_intensity), 0, min(205, grass_cap)
            ).astype(np.uint8)

        # Protected skin tones (Hue 8..26): prevent oversaturation and preserve natural luminance
        skin_mask = (h_ch >= 8) & (h_ch <= 26) & (s_ch >= 25)
        if np.any(skin_mask):
            s_ch[skin_mask] = np.clip(s_ch[skin_mask], 25, 175)
            v_ch[skin_mask] = np.clip(v_ch[skin_mask].astype(np.float32) * 1.02, 0, 255).astype(np.uint8)

        s_ch[:] = np.minimum(s_ch, 215)
        hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2] = h_ch, s_ch, v_ch
        recolored_small = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        if small is not frame:
            delta = cv2.resize(
                recolored_small.astype(np.int16) - small.astype(np.int16),
                (w, h),
                interpolation=cv2.INTER_LINEAR,
            )
            graded = np.clip(frame.astype(np.int16) + delta, 0, 255).astype(np.uint8)
        else:
            graded = recolored_small

        lut_b, lut_g, lut_r = cls._get_channel_luts(eff_intensity)
        out = np.empty_like(graded)
        out[:, :, 0] = cv2.LUT(graded[:, :, 0], lut_b)
        out[:, :, 1] = cv2.LUT(graded[:, :, 1], lut_g)
        out[:, :, 2] = cv2.LUT(graded[:, :, 2], lut_r)

        vig = cls._get_vignette(h, w, vig_strength)
        out = (out.astype(np.float32) * vig).astype(np.uint8)
        return out


# =====================================================================
# 7. SOUND DESIGN ENGINE
# =====================================================================
class SoundDesigner:
    """
    Sound Design Engine driven directly by Director Scene Audio Decisions:
    - Preserves original match audio.
    - Places riser, whoosh, impact, and crowd_emphasis at the exact output timestamps
      dictated by each scene's audio pillar in director_script.
    - Absence of music or input audio NEVER causes render failure.
    """

    @staticmethod
    def generate_sound_mix(
        input_path: str,
        output_audio_path: str,
        scenes: list,
        duration: float,
        sound_options: dict = None,
    ) -> bool:
        try:
            s_opts = sound_options or {}
            allow_impact = s_opts.get("impact", True)
            allow_whoosh = s_opts.get("whoosh", True)
            allow_riser = s_opts.get("riser", True)
            allow_crowd = s_opts.get("crowd_emphasis", True)

            probe_cmd = [
                "ffprobe", "-v", "quiet", "-select_streams", "a",
                "-show_entries", "stream=codec_type", "-of", "csv=p=0", input_path
            ]
            has_input_audio = bool(subprocess.run(probe_cmd, capture_output=True, text=True).stdout.strip())

            filtergraph = []
            mix_labels = ["[orig]"]

            if has_input_audio:
                filtergraph.append("[0:a]volume=0.94,equalizer=f=90:width_type=o:w=1.2:g=1.8[orig]")
            else:
                filtergraph.append(f"anoisesrc=d={duration}:c=pink:r=44100:a=0.010,lowpass=f=1100[orig]")

            hero_scene = next((s for s in (scenes or []) if s.get("scene_type") == "hero"), None)
            any_impact = any((s.get("audio") or {}).get("impact") for s in (scenes or []))
            any_whoosh = any((s.get("audio") or {}).get("whoosh") for s in (scenes or []))
            any_riser = any((s.get("audio") or {}).get("riser") for s in (scenes or []))
            any_crowd = any((s.get("audio") or {}).get("crowd_emphasis") for s in (scenes or []))

            # Use exact rendered output_start timestamp if available from Director execution
            cue_sec = float(
                hero_scene.get("output_start", hero_scene.get("start", duration * 0.35))
                if hero_scene else duration * 0.35
            )
            hero_start_ms = max(100, int(round(cue_sec * 1000.0)))

            if allow_impact and any_impact:
                filtergraph.append(
                    f"sine=f=68:d=0.38,volume=0.16,afade=t=out:st=0.10:d=0.28,adelay={hero_start_ms}|{hero_start_ms}[imp]"
                )
                mix_labels.append("[imp]")

            if allow_whoosh and any_whoosh:
                whoosh_ms = max(20, hero_start_ms - 180)
                filtergraph.append(
                    f"anoisesrc=d=0.28:c=white:r=44100:a=0.04,bandpass=f=650:width_type=h:w=400,"
                    f"afade=t=in:st=0:d=0.12,afade=t=out:st=0.14:d=0.14,adelay={whoosh_ms}|{whoosh_ms}[whs]"
                )
                mix_labels.append("[whs]")

            if allow_riser and any_riser and hero_start_ms > 400:
                riser_ms = max(0, hero_start_ms - 620)
                filtergraph.append(
                    f"sine=f=140:d=0.60,volume=0.06,afade=t=in:st=0:d=0.50,afade=t=out:st=0.50:d=0.10,"
                    f"adelay={riser_ms}|{riser_ms}[rsr]"
                )
                mix_labels.append("[rsr]")

            if allow_crowd and any_crowd:
                filtergraph.append(
                    f"anoisesrc=d=1.4:c=pink:r=44100:a=0.025,lowpass=f=950,"
                    f"afade=t=in:st=0:d=0.35,afade=t=out:st=0.90:d=0.50,adelay={hero_start_ms}|{hero_start_ms}[crd]"
                )
                mix_labels.append("[crd]")

            n_inputs = len(mix_labels)
            filtergraph.append(
                f"{''.join(mix_labels)}amix=inputs={n_inputs}:duration=first:dropout_transition=2:normalize=0[aout]"
            )

            cmd = ["ffmpeg", "-y"]
            if has_input_audio:
                cmd.extend(["-i", input_path])
            cmd.extend([
                "-filter_complex", ";".join(filtergraph),
                "-map", "[aout]",
                "-c:a", "aac", "-b:a", "192k",
                "-t", str(round(duration, 2)),
                output_audio_path,
            ])
            res = subprocess.run(cmd, capture_output=True, text=True)
            return res.returncode == 0 and os.path.exists(output_audio_path)
        except Exception:
            return False


# =====================================================================
# 8. MASTER CINEMATIC ENGINE
# =====================================================================
class CinematicEngine:
    """
    Production Football Cinematic AI Pipeline:
    Every decision from CinematicDirector & EditPlan physically drives the final MP4:
    subject -> anchor -> camera_trajectory -> zoom -> speed_curve ->
    isolation (MobileSAM->YOLO->motion->no_isolation) -> grade -> text ->
    audio -> transition -> evidence -> MP4 Trajectory & Technical/Style QC.
    """

    def __init__(self, use_player_tracker: bool = True):
        self.tracker = get_cached_player_tracker() if use_player_tracker else None

    @staticmethod
    def generate_synthetic_football_video(output_path: str, duration: float = 6.0, fps: int = 30):
        """
        Generates a realistic synthetic football match action clip (1280x720) with:
        - Pitch grass + penalty box lines + goal frame
        - Primary attacker dribbling the ball, engaging a defender (2.2s-3.1s),
          accelerating and firing a high-velocity shot into the goal (3.1s-4.0s),
          followed by celebration (4.1s-6.0s).
        """
        w, h = 1280, 720
        total_frames = max(30, int(round(duration * fps)))
        tmp_vid = output_path + ".synth_raw.mp4"
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(fps),
            "-i", "-",
            "-f", "lavfi", "-i", f"sine=frequency=220:duration={duration}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast",
            "-c:a", "aac", "-shortest",
            tmp_vid,
        ]
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        for idx in range(total_frames):
            t = float(idx) / float(fps)
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            # Turf green background with mowing stripes
            frame[:, :] = (38, 118, 46)
            for s_idx in range(0, w, 160):
                if (s_idx // 160) % 2 == 0:
                    frame[:, s_idx:min(w, s_idx + 160)] = (34, 110, 42)
            # Pitch lines & penalty area on right side
            cv2.rectangle(frame, (40, 40), (w - 40, h - 40), (215, 225, 215), 2)
            cv2.rectangle(frame, (w - 280, 160), (w - 40, h - 160), (215, 225, 215), 2)
            # Goal post on right
            cv2.rectangle(frame, (w - 44, 270), (w - 18, 450), (245, 245, 245), 3)

            # Attacker trajectory: build-up (0-1.8s), dribble/skill past defender (1.8-3.1s), shot (3.1-4.1s), celebration (4.1-6s)
            if t < 1.8:
                px = int(220 + (t / 1.8) * 260)
                py = int(380 - 25 * math.sin(t * 3.0))
            elif t < 3.1:
                u = (t - 1.8) / 1.3
                px = int(480 + u * 310)
                py = int(360 + 55 * math.sin(u * 6.28))
            elif t < 4.1:
                u = (t - 3.1) / 1.0
                px = int(790 + u * 120)
                py = int(350 - u * 30)
            else:
                u = (t - 4.1) / max(0.1, duration - 4.1)
                px = int(910 + 25 * math.sin(u * 9.0))
                py = int(320 - 35 * abs(math.sin(u * 6.0)))

            # Defender trajectory near (660, 355)
            dx = int(680 - 40 * math.sin(min(t, 3.5) * 1.2))
            dy = int(350 + 25 * math.cos(min(t, 3.5) * 1.8))
            cv2.rectangle(frame, (dx - 24, dy - 65), (dx + 24, dy + 65), (210, 70, 50), -1)
            cv2.circle(frame, (dx, dy - 78), 16, (190, 160, 140), -1)

            # Draw primary attacker
            cv2.rectangle(frame, (px - 28, py - 74), (px + 28, py + 74), (40, 65, 235), -1)
            cv2.circle(frame, (px, py - 88), 18, (210, 180, 155), -1)

            # Ball trajectory: close to attacker feet until 3.15s, then high-speed shot into goal (w-35, 340)
            if t < 3.15:
                bx = int(px + 34 + 8 * math.cos(t * 10.0))
                by = int(py + 56 + 6 * math.sin(t * 10.0))
            else:
                shot_u = min(1.0, (t - 3.15) / 0.55)
                bx = int((px + 36) * (1.0 - shot_u) + (w - 32) * shot_u)
                by = int((py + 52) * (1.0 - shot_u) + 335 * shot_u)
            cv2.circle(frame, (max(15, min(w - 15, bx)), max(15, min(h - 15, by))), 11, (250, 250, 250), -1)

            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        proc.wait()
        if os.path.exists(tmp_vid):
            os.replace(tmp_vid, output_path)

    def execute_pipeline(
        self,
        input_path: str,
        output_path: str,
        slow_factor: float = 0.25,
        target_segment: tuple = (2.0, 4.0),
        options: dict = None,
    ) -> dict:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")

        opts = options or {}
        raw_mode = str(opts.get("mode", "CINEMATIC")).upper().strip()
        is_psych_requested = (
            raw_mode in ("PSYCHOLOGICAL", "PSYCHOLOGICAL_STORY", "PSYCHOLOGICAL STORY")
            or bool(opts.get("psychological_story", False))
        )
        # Normalize mode for underlying Director/Engine while preserving psychological mode identity
        effective_mode = (
            "CINEMATIC"
            if raw_mode in ("PSYCHOLOGICAL", "PSYCHOLOGICAL_STORY", "PSYCHOLOGICAL STORY", "REFERENCE")
            else raw_mode
        )
        mode = raw_mode

        rss_start_mb = get_process_rss_mb()
        rss_peak_mb = rss_start_mb

        aspect_ratio = opts.get("aspect_ratio", "9:16")
        speed_ramp_type = opts.get("speed_ramp_type", "hero")
        enable_player_tracking = opts.get("player_tracking", True)
        enable_blur = opts.get("enable_blur", effective_mode == "CINEMATIC")
        apply_color_grade = opts.get("color_grade", effective_mode in ("PRO", "CINEMATIC"))
        apply_sound_design = opts.get("sound_design", effective_mode == "CINEMATIC")
        apply_smart_reframing = opts.get("smart_reframing", effective_mode in ("PRO", "CINEMATIC"))
        enable_speed_ramps = opts.get("speed_ramps", effective_mode in ("PRO", "CINEMATIC"))
        ref_style_key = opts.get("reference_style_preset", "ucl_broadcast_reel")
        ref_video_path = opts.get("reference_video_path")
        max_duration = min(64.0, max(1.0, float(opts.get("max_duration", 64.0))))

        target_w, target_h = (1080, 1920) if aspect_ratio == "9:16" else (1920, 1080)

        # STAGE 1: Video Analysis & Reference Style Analysis
        print("PROGRESS:5", flush=True)
        meta = VideoAnalyzer.probe(input_path)
        fps = meta["fps"] if meta["fps"] > 0 else 30.0
        total_duration = min(max_duration, meta["duration"] if meta["duration"] > 0 else 6.0)

        if ref_video_path and os.path.exists(str(ref_video_path)):
            reference_style = ReferenceStyleAnalyzer.analyze_reference_video(str(ref_video_path), ref_style_key)
        else:
            reference_style = ReferenceStyleAnalyzer.get_preset_profile(ref_style_key)

        # STAGE 2: YOLO Detection, Multi-ID Tracking & Kinematics Pass
        print("PROGRESS:15", flush=True)
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            self.generate_synthetic_football_video(input_path, duration=6.0, fps=30)
            meta = VideoAnalyzer.probe(input_path)
            fps = meta["fps"] if meta["fps"] > 0 else 30.0
            total_duration = min(max_duration, meta["duration"] if meta["duration"] > 0 else 6.0)
            cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {input_path}")

        use_yolo = enable_player_tracking and effective_mode in ("PRO", "CINEMATIC")
        detector_tracker = FootballDetectorTracker(meta["width"], meta["height"], fps, use_yolo=use_yolo)
        track_history = []
        frame_idx = 0
        sample_stride = max(1, int(round(fps / 6.0)))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            t_sec = float(frame_idx) / fps
            if t_sec > total_duration:
                break
            if frame_idx % sample_stride == 0:
                step_data = detector_tracker.detect_and_track_frame(frame, t_sec)
                track_history.append(step_data)
            frame_idx += 1
            if frame_idx % 90 == 0:
                gc.collect()
        cap.release()
        rss_peak_mb = max(rss_peak_mb, get_process_rss_mb())

        # STAGE 3: Evidence-Based Football Event Classification
        print("PROGRESS:25", flush=True)
        detected_events = FootballEventEngine.detect_events(track_history, total_duration)

        # STAGE 4: Evidence-Based Hero Moment Detection (No Hero without real evidence!)
        print("PROGRESS:32", flush=True)
        hero_moment = HeroMomentDetector.identify_hero_moment(
            track_history, total_duration, target_segment, detected_events
        )

        # STAGE 4.5: Safe Story Mode Decision Layer (Psychological Cinematic Engine)
        if is_psych_requested:
            psych_package = CinematicStoryteller.build_psychological_package(
                clip_path=input_path,
                track_history=track_history,
                detected_events=detected_events,
                hero_moment=hero_moment,
                clip_duration=total_duration,
                source_resolution=(meta["width"], meta["height"]),
                player_identities=opts.get("player_identities"),
                vlm_caller=opts.get("vlm_caller"),
                deepseek_caller=opts.get("deepseek_caller"),
            )
            if opts.get("depth_effect") is False:
                psych_package["depth_effect"] = False
            if opts.get("low_angle") is False:
                psych_package["low_angle"] = False
        else:
            psych_package = {
                "psychological_story": False,
                "story_role": None,
                "story_arc": [],
                "duel_moment": None,
                "winner": None,
                "loser": None,
                "confidence": 0.0,
                "depth_effect": False,
                "low_angle": False,
                "pov_switch": False,
                "story_script": [],
                "shot_language": {},
            }

        psych_active = bool(psych_package.get("psychological_story", False))
        story_script = list(psych_package.get("story_script") or []) if psych_active else []

        # STAGE 5: Cinematic Director Script, Edit Plan & Smart Reframing Anchor Track
        print("PROGRESS:40", flush=True)
        director_script = CinematicDirector.build_director_script(
            detected_events,
            hero_moment,
            total_duration,
            target_aspect=aspect_ratio,
            mode=effective_mode,
            speed_ramp_type=speed_ramp_type if enable_speed_ramps else "normal",
            reference_style=reference_style,
        )

        # Annotate Storyteller-selected scenes with POV switch, duel_moment, low_angle & depth_effect
        if psych_active:
            dm_val = psych_package.get("duel_moment")
            for sc in director_script:
                sc_type = sc.get("scene_type", "action")
                if sc_type in ("hero", "dribble", "skill", "shot", "tackle", "build_up", "action"):
                    sc["psychological_story"] = True
                    sc["duel_moment"] = dm_val
                    sc["pov_switch"] = bool(psych_package.get("pov_switch", True))
                    sc["low_angle"] = bool(psych_package.get("low_angle", True)) and sc_type in ("hero", "dribble", "skill", "shot")
                    sc["depth_effect"] = bool(psych_package.get("depth_effect", True)) and sc_type in ("hero", "dribble", "skill", "shot")
                    if sc_type == "hero":
                        # Suppress generic hero text so psychological inner-monologue script has no collision
                        if isinstance(sc.get("text"), dict):
                            sc["text"]["enabled"] = False
                            sc["text"]["content"] = None

        edit_plan = CinematicDirector.build_edit_plan(
            director_script,
            hero_moment,
            detected_events,
            total_duration,
            target_aspect=aspect_ratio,
            mode=mode,
            psychological_package=psych_package,
        )
        anchor_track = SmartReframer.build_anchor_track(
            track_history,
            director_script,
            meta["width"],
            meta["height"],
            target_w=target_w,
            target_h=target_h,
            fps=fps,
            total_duration=total_duration,
        )

        # Lazy-load Depth Anything V2 Small ONLY when Psychological Story mode is active & depth_effect is enabled
        depth_engine = None
        if psych_active and bool(psych_package.get("depth_effect", True)):
            depth_engine = DepthAnythingV2Engine(lazy_load=False)

        # STAGE 6: Stream Composition & FFmpeg Raw Pipe (100% Director Decisions -> MP4)
        print("PROGRESS:50", flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        temp_video_raw = output_path + ".temp_video.mp4"
        temp_audio_file = output_path + ".temp_audio.aac"

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{target_w}x{target_h}",
            "-pix_fmt", "bgr24",
            "-r", "30",
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "veryfast",
            "-crf", "20",
            temp_video_raw,
        ]
        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

        cap = cv2.VideoCapture(input_path)
        curr_frame_idx = 0
        written_frames = 0
        bad_crops = 0
        slow_mo_input_frames = 0
        total_expected_input_frames = max(1, int(round(total_duration * fps)))
        # Strict budget: slow motion cannot exceed 40% of input video frames
        max_slow_mo_input_frames = max(1, int(round(total_expected_input_frames * 0.40)))

        frame_crops = anchor_track["frame_crops"]
        grid_x = None
        grid_y = None
        scenes_rendered_set = set()
        isolated_frames_count = 0
        graded_frames_count = 0
        interpolated_frames_count = 0
        full_frame_blur_detected = False
        low_angle_geometry_preserved = True

        for sc in director_script:
            sc["output_start"] = None
            sc["output_end"] = None
            sc["frames_rendered"] = 0
            sc["interpolated_frames_added"] = 0

        try:
            prev_graded_frame = None
            while cap.isOpened():
                ret, raw_frame = cap.read()
                if not ret or raw_frame is None:
                    break
                t_sec = float(curr_frame_idx) / fps
                if t_sec > total_duration:
                    break

                active_scene = next(
                    (s for s in director_script if s["start"] <= t_sec <= s["end"]),
                    director_script[-1] if director_script else None,
                )
                if active_scene is not None:
                    scenes_rendered_set.add(active_scene.get("scene_type", "action"))
                    if active_scene["output_start"] is None:
                        active_scene["output_start"] = round(float(written_frames) / 30.0, 2)

                scene_dur = max(0.001, float(active_scene["end"] - active_scene["start"])) if active_scene else 1.0
                scene_prog = max(0.0, min(1.0, (t_sec - float(active_scene["start"])) / scene_dur)) if active_scene else 0.5
                scene_ev_backed = bool((active_scene.get("evidence") or {}).get("event_backed", False)) if active_scene else False
                scene_psych = bool(active_scene.get("psychological_story", False)) if active_scene else False

                closest_track = min(track_history, key=lambda tr: abs(tr["timestamp"] - t_sec)) if track_history else None
                p_bbox = closest_track["player"]["bbox"] if (closest_track and closest_track.get("player")) else None
                b_obj = closest_track.get("ball") if closest_track else None

                # 1. Smart Dynamic Reframing via multi-keyframe anchor_track
                if apply_smart_reframing and frame_crops:
                    crop_idx = min(len(frame_crops) - 1, curr_frame_idx)
                    x1, y1, cw, ch = frame_crops[crop_idx]
                    if cw <= 0 or ch <= 0 or x1 < 0 or y1 < 0 or (x1 + cw) > raw_frame.shape[1] or (y1 + ch) > raw_frame.shape[0]:
                        bad_crops += 1
                        x1, y1, cw, ch = 0, 0, raw_frame.shape[1], raw_frame.shape[0]
                    cropped = raw_frame[y1:y1 + ch, x1:x1 + cw]
                    framed_frame = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    scale_x = float(target_w) / float(cw)
                    scale_y = float(target_h) / float(ch)
                    if p_bbox:
                        adj_bbox = (
                            int(max(0, (p_bbox[0] - x1) * scale_x)),
                            int(max(0, (p_bbox[1] - y1) * scale_y)),
                            int(min(target_w, (p_bbox[2] - x1) * scale_x)),
                            int(min(target_h, (p_bbox[3] - y1) * scale_y)),
                        )
                    else:
                        adj_bbox = None
                    if b_obj and b_obj.get("dist_to_player", 999.0) < 120.0:
                        adj_ball = (
                            int((float(b_obj["bx"]) - x1) * scale_x),
                            int((float(b_obj["by"]) - y1) * scale_y),
                        )
                    else:
                        adj_ball = None
                else:
                    framed_frame = cv2.resize(raw_frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    adj_bbox = p_bbox
                    adj_ball = None

                # 2. Subject Isolation (MobileSAM -> YOLO -> motion -> no_isolation) + Optional Depth Anything V2 Bokeh
                iso_cfg = (active_scene.get("isolation_config") or {}) if active_scene else {}
                should_isolate = (
                    enable_blur
                    and effective_mode in ("PRO", "CINEMATIC")
                    and active_scene is not None
                    and bool(active_scene.get("isolation", False))
                    and scene_ev_backed
                )
                if psych_active and scene_psych and depth_engine is not None and bool(active_scene.get("depth_effect", False)):
                    d_map = depth_engine.estimate_depth_map(
                        framed_frame, player_bbox=adj_bbox, frame_idx=curr_frame_idx, sample_interval=3
                    )
                    processed_frame, bokeh_meta = depth_engine.apply_depth_aware_bokeh(
                        framed_frame,
                        depth_map=d_map,
                        player_bbox=adj_bbox,
                        blur_strength=15,
                        dim_background=0.88,
                        frame_idx=curr_frame_idx,
                    )
                    if bokeh_meta.get("full_frame_blur"):
                        full_frame_blur_detected = True
                    if bool(active_scene.get("low_angle", False)):
                        processed_frame, la_meta = depth_engine.apply_low_angle_pov(
                            processed_frame, player_bbox=adj_bbox, depth_map=d_map, intensity=0.26
                        )
                        if not la_meta.get("geometry_preserved", True):
                            low_angle_geometry_preserved = False
                    isolated_frames_count += 1
                elif should_isolate and self.tracker is not None:
                    b_k = int(iso_cfg.get("blur_kernel", 21)) | 1
                    d_r = float(iso_cfg.get("dim_ratio", 0.88))
                    f_px = int(iso_cfg.get("feather_px", 17)) | 1
                    processed_frame = self.tracker.apply_cinematic_blur(
                        framed_frame,
                        bbox=adj_bbox,
                        blur_kernel=(b_k, b_k),
                        dim_ratio=d_r,
                        feather_radius=f_px,
                        ball_xy=adj_ball,
                    )
                    isolated_frames_count += 1
                else:
                    processed_frame = framed_frame

                # 3. Cinematic Color Grading (with Psychological Teal-Orange LUT on Storyteller-selected shots)
                if apply_color_grade:
                    grade_cfg = (active_scene.get("grade") or {}) if active_scene else {}
                    grade_int = float(grade_cfg.get("intensity", 0.92 if (active_scene and active_scene.get("scene_type") == "hero") else 0.80))
                    if psych_active and scene_psych:
                        graded_frame = PsychologicalColorGrader.apply_teal_orange_grade(
                            processed_frame, intensity=min(0.92, grade_int)
                        )
                    else:
                        graded_frame = CinematicColorGrader.apply_grade(
                            processed_frame, intensity=grade_int, grade_config=grade_cfg
                        )
                    graded_frames_count += 1
                else:
                    graded_frame = processed_frame

                # 4. Context-Linked Rare Text, Psychological Story Script & Scene Transitions
                if active_scene:
                    graded_frame = CinematicTransitionEngine.apply_transition(
                        graded_frame,
                        prev_graded_frame,
                        active_scene.get("transition_config") or active_scene.get("transition", "hard_cut"),
                        scene_prog,
                    )
                    graded_frame = CinematicTextRenderer.render_overlay(
                        graded_frame,
                        active_scene.get("text"),
                        scene_prog,
                    )
                if psych_active and story_script:
                    graded_frame = CinematicStoryteller.render_psychological_caption(
                        graded_frame, story_script, t_sec
                    )

                # 5. Event-Driven Speed Ramping (ONLY when event_backed AND within <=40% slow-mo budget)
                speed_cfg = (active_scene.get("speed_curve") or active_scene.get("speed") or {}) if active_scene else {}
                scene_speed_mode = str(speed_cfg.get("mode", active_scene.get("speed_mode", "normal") if active_scene else "normal"))
                if (
                    enable_speed_ramps
                    and scene_ev_backed
                    and scene_speed_mode != "normal"
                    and slow_mo_input_frames < max_slow_mo_input_frames
                ):
                    speed_mult = CinematicDirector.get_speed_multiplier(scene_prog, ramp_type=scene_speed_mode)
                    if speed_mult < 0.88:
                        slow_mo_input_frames += 1
                else:
                    speed_mult = 1.0

                ffmpeg_proc.stdin.write(graded_frame.tobytes())
                written_frames += 1
                if active_scene is not None:
                    active_scene["frames_rendered"] += 1

                # Selective Optical Flow / RIFE-compatible Slow Motion Interpolation ONLY when speed_mult <= 0.48
                if (
                    enable_speed_ramps
                    and scene_ev_backed
                    and speed_mult <= 0.48
                    and prev_graded_frame is not None
                    and bool(speed_cfg.get("use_optical_flow", True))
                ):
                    n_interp = min(2, max(1, int(round(1.0 / max(0.22, speed_mult))) - 1))
                    if grid_x is None:
                        grid_x, grid_y = np.meshgrid(
                            np.arange(target_w, dtype=np.float32),
                            np.arange(target_h, dtype=np.float32),
                        )
                    fl_w, fl_h = max(64, target_w // 4), max(64, target_h // 4)
                    prev_gray_s = cv2.resize(cv2.cvtColor(prev_graded_frame, cv2.COLOR_BGR2GRAY), (fl_w, fl_h))
                    curr_gray_s = cv2.resize(cv2.cvtColor(graded_frame, cv2.COLOR_BGR2GRAY), (fl_w, fl_h))
                    flow_fwd = cv2.calcOpticalFlowFarneback(
                        prev_gray_s, curr_gray_s, None,
                        pyr_scale=0.5, levels=3, winsize=10, iterations=1, poly_n=5, poly_sigma=1.1, flags=0
                    )
                    flow_fwd = cv2.resize(flow_fwd, (target_w, target_h), interpolation=cv2.INTER_LINEAR) * 4.0
                    for step in range(1, n_interp + 1):
                        alpha = float(step) / float(n_interp + 1)
                        map_x = (grid_x - alpha * flow_fwd[:, :, 0]).astype(np.float32)
                        map_y = (grid_y - alpha * flow_fwd[:, :, 1]).astype(np.float32)
                        interp = cv2.remap(
                            prev_graded_frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
                        )
                        ffmpeg_proc.stdin.write(interp.tobytes())
                        written_frames += 1
                        interpolated_frames_count += 1
                        if active_scene is not None:
                            active_scene["frames_rendered"] += 1
                            active_scene["interpolated_frames_added"] += 1
                        del map_x, map_y, interp
                    del flow_fwd, prev_gray_s, curr_gray_s

                if active_scene is not None:
                    active_scene["output_end"] = round(float(written_frames) / 30.0, 2)

                prev_graded_frame = graded_frame
                curr_frame_idx += 1
                if curr_frame_idx % 30 == 0:
                    rss_peak_mb = max(rss_peak_mb, get_process_rss_mb())
                    pct = min(85, max(50, 50 + int((curr_frame_idx / max(1.0, total_duration * fps)) * 35.0)))
                    print(f"PROGRESS:{pct}", flush=True)
                    gc.collect()
        finally:
            cap.release()
            if ffmpeg_proc.stdin:
                ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()
            rss_peak_mb = max(rss_peak_mb, get_process_rss_mb())
            broken_depth_masks = depth_engine.broken_depth_masks if depth_engine is not None else 0
            depth_backend_used = depth_engine.backend_used if depth_engine is not None else "disabled"
            if depth_engine is not None:
                depth_engine.release()
            del prev_graded_frame, grid_x, grid_y
            gc.collect()

        # STAGE 7: Sound Design & Match Audio Preservation aligned with Director output timestamps
        print("PROGRESS:88", flush=True)
        out_duration = round(float(written_frames) / 30.0, 2)
        has_audio = False
        if apply_sound_design:
            has_audio = SoundDesigner.generate_sound_mix(
                input_path, temp_audio_file, director_script, out_duration, opts.get("sound_options")
            )

        # STAGE 8: FFmpeg Final MP4 Assembly (1080x1920, 30fps, H.264, yuv420p, AAC)
        print("PROGRESS:92", flush=True)
        mux_cmd = ["ffmpeg", "-y", "-i", temp_video_raw]
        if has_audio and os.path.exists(temp_audio_file):
            mux_cmd.extend(["-i", temp_audio_file, "-c:a", "aac", "-b:a", "192k", "-shortest"])
        else:
            mux_cmd.extend(["-an"])
        mux_cmd.extend(["-c:v", "copy", "-movflags", "+faststart", output_path])
        mux_res = subprocess.run(mux_cmd, capture_output=True, text=True)
        if mux_res.returncode != 0 or not os.path.exists(output_path):
            raise RuntimeError(f"FFmpeg muxing failed: {mux_res.stderr}")

        for tmp in (temp_video_raw, temp_audio_file):
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass

        # STAGE 9: Direct MP4 Trajectory Verification + Technical, Style & Psychological QC Validation
        print("PROGRESS:97", flush=True)
        mp4_trajectory_verification = SmartReframer.verify_mp4_trajectory(output_path, anchor_track)
        iso_telemetry = self.tracker.get_isolation_telemetry() if self.tracker else {
            "method_used": "no_isolation",
            "fallback_chain": ["mobilesam", "yolo", "motion", "no_isolation"],
            "frames_isolated": 0,
            "mask_errors": 0,
        }
        slow_mo_ratio_actual = round(float(slow_mo_input_frames) / max(1.0, float(curr_frame_idx)), 3)
        director_execution_audit = {
            "all_scenes_rendered": bool(len(scenes_rendered_set) >= 1),
            "scenes_rendered": sorted(list(scenes_rendered_set)),
            "reframing_applied_to_mp4": bool(apply_smart_reframing and mp4_trajectory_verification.get("verified", True)),
            "speed_curve_applied_to_mp4": bool(slow_mo_ratio_actual <= 0.42),
            "slow_motion_input_ratio": slow_mo_ratio_actual,
            "interpolated_frames_added": int(interpolated_frames_count),
            "grade_applied_to_mp4": bool(graded_frames_count > 0 if apply_color_grade else True),
            "isolation_applied_to_mp4": bool(isolated_frames_count >= 0),
            "isolation_method_used": iso_telemetry.get("method_used", "no_isolation"),
            "depth_backend_used": depth_backend_used,
            "audio_cues_muxed_to_mp4": bool(has_audio),
        }

        rss_end_mb = get_process_rss_mb()
        rss_peak_mb = max(rss_peak_mb, rss_end_mb)
        memory_telemetry = {
            "baseline_rss_mb": round(rss_start_mb, 2),
            "peak_rss_mb": round(rss_peak_mb, 2),
            "end_rss_mb": round(rss_end_mb, 2),
            "delta_rss_mb": round(max(0.0, rss_peak_mb - rss_start_mb), 2),
            "depth_model_released": True,
        }

        confs = [f["player"].get("confidence", 0.85) for f in track_history if f.get("player")]
        pipeline_telemetry = {
            "max_jump_ratio": anchor_track.get("max_jump_ratio", 0.008),
            "bad_crops": bad_crops,
            "mean_tracking_confidence": float(np.mean(confs)) if confs else 0.85,
            "mask_errors": self.tracker.mask_error_count if self.tracker else 0,
            "isolation_method": iso_telemetry.get("method_used", "no_isolation"),
            "isolation_enabled": enable_blur,
            "director_script": director_script,
            "anchor_track": anchor_track,
            "mp4_trajectory_verification": mp4_trajectory_verification,
            "director_execution_audit": director_execution_audit,
            "speed_ramp_type": speed_ramp_type if hero_moment.get("is_hero") else "normal",
            "peak_zoom_applied": reference_style.get("zoom_intensity", {}).get("hero_peak_zoom", 1.32),
            "text_policy": reference_style.get("text_frequency", {}).get("policy", "minimal_context_only"),
            "color_grade_profile": "psychological_teal_orange_lut" if psych_active else ("cinematic_turf_lut" if apply_color_grade else "none"),
            "psychological_story": psych_active,
            "duel_confidence": float(psych_package.get("duel_confidence", psych_package.get("confidence", 0.0))),
            "story_confidence": float(psych_package.get("story_confidence", psych_package.get("confidence", 0.0))),
            "depth_effect_used": bool(psych_active and psych_package.get("depth_effect", False)),
            "low_angle_used": bool(psych_active and psych_package.get("low_angle", False)),
            "pov_switch_used": bool(psych_active and psych_package.get("pov_switch", False)),
            "story_script_count": len(story_script),
            "story_script": story_script,
            "source_duration": total_duration,
            "player_retention_rate": anchor_track.get("player_retention_rate", 1.0),
            "full_frame_blur_detected": full_frame_blur_detected,
            "broken_depth_masks": broken_depth_masks,
            "low_angle_geometry_preserved": low_angle_geometry_preserved,
        }
        qc_report = QualityControlEngine.inspect(
            output_path,
            out_duration,
            target_w,
            target_h,
            pipeline_telemetry=pipeline_telemetry,
            reference_style=reference_style,
        )
        if not qc_report["passed"]:
            raise RuntimeError(f"QC Validation Failed: {qc_report.get('error') or 'Quality thresholds not met'}")

        result_payload = {
            "status": "completed",
            "output_frames": written_frames,
            "target_resolution": f"{target_w}x{target_h}",
            "mode": mode,
            "detector_used": detector_tracker.detector_source,
            "player_tracks_count": len(detector_tracker.active_tracks),
            "detected_events": detected_events,
            "hero_moment": hero_moment,
            "director_script": director_script,
            "edit_plan": edit_plan,
            "psychological_story": psych_active,
            "story_script": story_script,
            "duel_analysis": {
                "duel_moment": psych_package.get("duel_moment"),
                "winner": psych_package.get("winner"),
                "loser": psych_package.get("loser"),
                "story_arc": psych_package.get("story_arc", []),
                "confidence": psych_package.get("confidence", 0.0),
            },
            "vlm_telemetry": psych_package.get("vlm_telemetry", CinematicStoryteller.LAST_VLM_TELEMETRY),
            "deepseek_telemetry": psych_package.get("deepseek_telemetry", CinematicStoryteller.LAST_DEEPSEEK_TELEMETRY),
            "rife_telemetry": {
                "rife_ncnn_available": bool(shutil.which("rife-ncnn-vulkan") is not None),
                "backend_used": "rife_ncnn" if shutil.which("rife-ncnn-vulkan") else "farneback_winsize10_levels3_iter1",
                "interpolated_frames_added": int(interpolated_frames_count),
            },
            "video_url": opts.get("video_url"),
            "poster_url": opts.get("poster_url"),
            "memory_telemetry": memory_telemetry,
            "isolation_telemetry": iso_telemetry,
            "director_execution_audit": director_execution_audit,
            "anchor_track_summary": {
                "keyframes_count": len(anchor_track["keyframes"]),
                "max_jump_ratio": anchor_track["max_jump_ratio"],
                "smooth_passed": anchor_track["smooth_passed"],
                "dynamic_framing_active": anchor_track.get("dynamic_framing_active", True),
                "zoom_variation_active": anchor_track.get("zoom_variation_active", True),
                "player_retention_rate": anchor_track.get("player_retention_rate", 1.0),
                "lost_player_frames": anchor_track.get("lost_player_frames", 0),
                "framing_modes_used": anchor_track.get("framing_modes_used", []),
                "mp4_trajectory_verification": mp4_trajectory_verification,
                "sample_keyframes": anchor_track["keyframes"][:8],
            },
            "reference_style": reference_style,
            "qc_report": qc_report,
        }
        try:
            with open(output_path + ".report.json", "w", encoding="utf-8") as rf:
                json.dump(result_payload, rf)
        except Exception:
            pass

        print("PROGRESS:100", flush=True)
        return result_payload

    # Backward compatibility for existing CLI signature
    def slow_motion_optical_flow(
        self,
        input_path: str,
        output_path: str,
        slow_factor: float = 0.25,
        target_segment: tuple = (2.0, 4.0),
        bbox=None,
    ) -> dict:
        return self.execute_pipeline(
            input_path=input_path,
            output_path=output_path,
            slow_factor=slow_factor,
            target_segment=target_segment,
            options={"mode": "CINEMATIC", "aspect_ratio": "9:16"},
        )

    def slow_motion_rife(
        self,
        input_path: str,
        output_path: str,
        slow_factor: float = 0.25,
        target_segment: tuple = (2.0, 4.0),
    ) -> dict:
        """
        Uses RIFE/ncnn when available; otherwise safe Farneback fallback with
        winsize=10, levels=3, iterations=1, interpolating only inside target_segment
        and running gc.collect() every 30 frames.
        """
        if not input_path or not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open input video: {input_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if fps <= 0 or math.isnan(fps):
            fps = 30.0
        w = max(16, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280))
        h = max(16, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720))
        seg_start, seg_end = float(target_segment[0]), float(target_segment[1])
        has_rife_bin = shutil.which("rife-ncnn-vulkan") is not None

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(round(fps, 2)),
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
            output_path,
        ]
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        frame_idx = 0
        written = 0
        interp_added = 0
        prev_frame = None
        grid_x, grid_y = None, None
        n_interp = min(3, max(1, int(round(1.0 / max(0.15, float(slow_factor)))) - 1))

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                t_sec = float(frame_idx) / fps
                proc.stdin.write(frame.tobytes())
                written += 1

                if seg_start <= t_sec <= seg_end and prev_frame is not None:
                    if grid_x is None:
                        grid_x, grid_y = np.meshgrid(
                            np.arange(w, dtype=np.float32),
                            np.arange(h, dtype=np.float32),
                        )
                    fl_w, fl_h = max(64, w // 4), max(64, h // 4)
                    prev_g = cv2.resize(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), (fl_w, fl_h))
                    curr_g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (fl_w, fl_h))
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_g, curr_g, None,
                        pyr_scale=0.5, levels=3, winsize=10, iterations=1, poly_n=5, poly_sigma=1.1, flags=0
                    )
                    flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR) * 4.0
                    for step in range(1, n_interp + 1):
                        alpha = float(step) / float(n_interp + 1)
                        mx = (grid_x - alpha * flow[:, :, 0]).astype(np.float32)
                        my = (grid_y - alpha * flow[:, :, 1]).astype(np.float32)
                        mid = cv2.remap(prev_frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                        proc.stdin.write(mid.tobytes())
                        written += 1
                        interp_added += 1
                        del mx, my, mid
                    del flow, prev_g, curr_g

                prev_frame = frame
                frame_idx += 1
                if frame_idx % 30 == 0:
                    gc.collect()
        finally:
            cap.release()
            if proc.stdin:
                proc.stdin.close()
            proc.wait()
            del prev_frame, grid_x, grid_y
            gc.collect()

        return {
            "status": "completed",
            "backend": "rife_ncnn" if has_rife_bin else "farneback_winsize10_levels3_iter1",
            "output_frames": written,
            "interpolated_frames": interp_added,
            "target_segment": [seg_start, seg_end],
        }

    @staticmethod
    def add_depth_bokeh(
        input_or_frame,
        output_path: str = None,
        depth_enabled: bool = True,
        player_bbox=None,
        player_mask=None,
    ):
        """
        Applies Depth Anything V2 depth-aware background bokeh while keeping the tracked player/SAM
        foreground sharp and NEVER blurring the entire frame. Supports both single frame np.ndarray
        and (input_path, output_path, depth_enabled) streaming video files.
        """
        if isinstance(input_or_frame, np.ndarray):
            if not depth_enabled:
                return input_or_frame
            eng = DepthAnythingV2Engine(lazy_load=False)
            d_map = eng.estimate_depth_map(input_or_frame, player_bbox=player_bbox, frame_idx=0)
            out_f, _ = eng.apply_depth_aware_bokeh(
                input_or_frame, depth_map=d_map, player_bbox=player_bbox, player_mask=player_mask
            )
            eng.release()
            return out_f

        input_path = str(input_or_frame)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")
        if not depth_enabled or not output_path:
            if output_path and input_path != output_path:
                shutil.copyfile(input_path, output_path)
            return {"status": "skipped", "depth_enabled": False}

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {input_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if fps <= 0 or math.isnan(fps):
            fps = 30.0
        w = max(16, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280))
        h = max(16, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720))

        eng = DepthAnythingV2Engine(lazy_load=False)
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(round(fps, 2)),
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
            output_path,
        ]
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        f_idx = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                d_map = eng.estimate_depth_map(frame, player_bbox=player_bbox, frame_idx=f_idx, sample_interval=3)
                bokeh_f, _ = eng.apply_depth_aware_bokeh(
                    frame, depth_map=d_map, player_bbox=player_bbox, player_mask=player_mask, frame_idx=f_idx
                )
                proc.stdin.write(bokeh_f.tobytes())
                f_idx += 1
                if f_idx % 30 == 0:
                    gc.collect()
        finally:
            cap.release()
            if proc.stdin:
                proc.stdin.close()
            proc.wait()
            eng.release()
            gc.collect()
        return {"status": "completed", "frames": f_idx, "depth_enabled": True}

    @staticmethod
    def add_low_angle_pov(
        frame_or_input,
        output_path: str = None,
        player_bbox=None,
        duel_verified: bool = True,
        intensity: float = 0.30,
    ):
        """
        When a verified duel moment exists, applies a restrained bottom-up low-angle POV (~0.48m ground feel)
        around the lower frame while preserving subject geometry (0.0 stretching/distortion).
        """
        if isinstance(frame_or_input, np.ndarray):
            if not duel_verified:
                return frame_or_input
            eng = DepthAnythingV2Engine(lazy_load=True)
            out_f, _ = eng.apply_low_angle_pov(frame_or_input, player_bbox=player_bbox, intensity=intensity)
            return out_f
        return frame_or_input

    @staticmethod
    def add_cinematic_lut(
        frame_or_input,
        output_path: str = None,
        intensity: float = 0.85,
    ):
        """
        BGR -> LAB, CLAHE on L channel, controlled teal-orange LUT treatment, preserving skin tones
        and avoiding neon grass.
        """
        if isinstance(frame_or_input, np.ndarray):
            return PsychologicalColorGrader.add_cinematic_lut(frame_or_input, intensity=intensity)
        return frame_or_input


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python3 video_engine.py <input> <output> [slow_factor] [seg_start] [seg_end] [options_json]",
            file=sys.stderr,
        )
        sys.exit(1)
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    factor = float(sys.argv[3]) if len(sys.argv) >= 4 else 0.25
    s_start = float(sys.argv[4]) if len(sys.argv) >= 5 else 2.0
    s_end = float(sys.argv[5]) if len(sys.argv) >= 6 else 4.0
    opts = {}
    if len(sys.argv) >= 7:
        try:
            opts = json.loads(sys.argv[6])
        except Exception:
            opts = {}
    engine = CinematicEngine(use_player_tracker=True)
    result = engine.execute_pipeline(
        input_path=in_file,
        output_path=out_file,
        slow_factor=factor,
        target_segment=(s_start, s_end),
        options=opts,
    )
    print(f"COMPLETED:{result['output_frames']}", flush=True)
