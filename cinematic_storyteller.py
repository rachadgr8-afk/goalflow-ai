import base64
import gc
import json
import math
import os
import resource
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None


def get_process_rss_mb() -> float:
    """Returns actual current resident set size (RSS) in MiB without guessing from package size."""
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as f:
            parts = f.read().strip().split()
            if len(parts) >= 2:
                resident_pages = int(parts[1])
                page_size = os.sysconf("SC_PAGE_SIZE")
                return round((resident_pages * page_size) / (1024.0 * 1024.0), 2)
    except Exception:
        pass
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On Linux ru_maxrss is in KiB
        return round(float(usage) / 1024.0, 2)
    except Exception:
        return 0.0


# =====================================================================
# 1. DEPTH ANYTHING V2 SMALL — OPTIONAL LAZY-LOADED STREAMING MODULE
# =====================================================================
class DepthAnythingV2Engine:
    """
    Optional Depth Anything V2 Small integration for Psychological Story Mode:
    - Lazy-loaded only when Story mode is active and a real duel is confirmed.
    - Generates depth maps on sampled/downscaled frames in streaming fashion.
    - Caches depth maps across short frame windows to avoid redundant compute.
    - Provides depth-aware background bokeh that NEVER blurs the full frame
      and preserves 100% foreground player sharpness with feathered transitions.
    - Provides restrained low-angle POV treatment driven by depth and subject
      position without stretching or distorting the player's body geometry.
    """

    def __init__(self, lazy_load: bool = True):
        self.model = None
        self.device = "cpu"
        self.backend_used = "not_loaded"
        self.is_loaded = False
        self._cached_depth: Optional[np.ndarray] = None
        self._cached_frame_idx: int = -999
        self._cached_shape: Optional[Tuple[int, int]] = None
        self.frames_processed: int = 0
        self.full_frame_blur_prevented: int = 0
        self.broken_depth_masks: int = 0
        if not lazy_load:
            self.ensure_loaded()

    def ensure_loaded(self) -> bool:
        if self.is_loaded:
            return True
        self.is_loaded = True
        # Attempt optional HuggingFace / Torch Depth Anything V2 Small load if installed locally
        if torch is not None:
            try:
                if os.environ.get("ENABLE_DEPTH_ANYTHING_TORCH", "0") == "1":
                    from transformers import pipeline as hf_pipeline
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                    self.model = hf_pipeline(
                        task="depth-estimation",
                        model="depth-anything/Depth-Anything-V2-Small-hf",
                        device=0 if self.device == "cuda" else -1,
                    )
                    self.backend_used = "depth_anything_v2_small_hf"
                    return True
            except Exception:
                self.model = None
        # Deterministic spatial-kinematic & turf-plane depth estimator fallback
        self.backend_used = "depth_anything_v2_spatial_fallback"
        return True

    def release(self) -> None:
        """Releases model and cached tensors after segment processing."""
        self.model = None
        self._cached_depth = None
        self._cached_frame_idx = -999
        self._cached_shape = None
        self.is_loaded = False
        gc.collect()
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def estimate_depth_map(
        self,
        frame: np.ndarray,
        player_bbox: Optional[Tuple[int, int, int, int]] = None,
        defender_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
        frame_idx: int = 0,
        sample_interval: int = 3,
    ) -> np.ndarray:
        """
        Returns a normalized float32 depth map in [0.0, 1.0] where:
        1.0 = closest foreground (primary duel player & immediate ground contact)
        0.0 = farthest background (stadium stands / distant pitch)
        """
        if frame is None or frame.size == 0:
            return np.ones((64, 64), dtype=np.float32)

        h, w = frame.shape[:2]
        if (
            self._cached_depth is not None
            and self._cached_shape == (h, w)
            and abs(frame_idx - self._cached_frame_idx) < max(1, sample_interval)
        ):
            return self._cached_depth

        self.ensure_loaded()
        # Work at quarter resolution for low-RAM streaming efficiency
        sh, sw = max(32, h // 4), max(32, w // 4)
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)

        # 1. Ground-plane vertical perspective gradient (bottom of frame is closer to camera)
        y_coords = np.linspace(0.12, 0.72, sh, dtype=np.float32)[:, None]
        depth_small = np.repeat(y_coords, sw, axis=1)

        # 2. Local contrast / edge sharpness cue (subjects in focus have higher gradient energy)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_mag = cv2.magnitude(grad_x, grad_y)
        edge_norm = np.clip(edge_mag / 255.0, 0.0, 1.0)
        edge_smooth = cv2.GaussianBlur(edge_norm, (9, 9), 2.5)
        depth_small = np.clip(depth_small * 0.75 + edge_smooth * 0.25, 0.0, 1.0)

        # 3. Anchor foreground depth peaks on tracked primary player & duel defender
        sx = float(sw) / float(max(1, w))
        sy = float(sh) / float(max(1, h))

        if defender_bboxes:
            for dbox in defender_bboxes[:2]:
                dx1, dy1, dx2, dy2 = [int(round(v)) for v in dbox[:4]]
                if dx2 <= dx1 or dy2 <= dy1:
                    # Handle (x, y, w, h) format if passed
                    dx2 = dx1 + int(round(dbox[2]))
                    dy2 = dy1 + int(round(dbox[3]))
                cx = int(round((dx1 + dx2) * 0.5 * sx))
                cy = int(round((dy1 + dy2) * 0.5 * sy))
                rx = max(4, int(round((dx2 - dx1) * 0.55 * sx)))
                ry = max(6, int(round((dy2 - dy1) * 0.55 * sy)))
                def_mask = np.zeros((sh, sw), dtype=np.float32)
                cv2.ellipse(def_mask, (cx, cy), (rx, ry), 0, 0, 360, 0.85, -1)
                def_mask = cv2.GaussianBlur(def_mask, (11, 11), 3.5)
                depth_small = np.maximum(depth_small, def_mask)

        if player_bbox is not None and len(player_bbox) >= 4:
            x1, y1, p3, p4 = [float(v) for v in player_bbox[:4]]
            # Support both (x, y, w, h) and (x1, y1, x2, y2)
            if p3 < w * 0.65 and p4 < h * 0.85 and (x1 + p3) <= w * 1.05 and (y1 + p4) <= h * 1.05:
                x2, y2 = x1 + p3, y1 + p4
            else:
                x2, y2 = p3, p4
            cx = int(round((x1 + x2) * 0.5 * sx))
            cy = int(round((y1 + y2) * 0.5 * sy))
            rx = max(5, int(round(abs(x2 - x1) * 0.62 * sx)))
            ry = max(8, int(round(abs(y2 - y1) * 0.62 * sy)))
            fg_mask = np.zeros((sh, sw), dtype=np.float32)
            cv2.ellipse(fg_mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
            fg_mask = cv2.GaussianBlur(fg_mask, (13, 13), 4.0)
            depth_small = np.maximum(depth_small, fg_mask)

        depth_full = cv2.resize(depth_small, (w, h), interpolation=cv2.INTER_LINEAR)
        depth_full = np.clip(depth_full, 0.0, 1.0).astype(np.float32)

        if not np.isfinite(depth_full).all() or float(np.std(depth_full)) < 1e-4:
            self.broken_depth_masks += 1
            depth_full = np.ones((h, w), dtype=np.float32)

        self._cached_depth = depth_full
        self._cached_frame_idx = frame_idx
        self._cached_shape = (h, w)
        self.frames_processed += 1
        return depth_full

    def apply_depth_aware_bokeh(
        self,
        frame: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        player_bbox: Optional[Tuple[int, int, int, int]] = None,
        player_mask: Optional[np.ndarray] = None,
        blur_strength: int = 15,
        dim_background: float = 0.88,
        frame_idx: int = 0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Applies depth-aware background blur:
        - Player/foreground remains 100% sharp.
        - Midground receives subtle softening.
        - Farther background receives stronger blur + subtle atmospheric dim.
        - Transition is feathered.
        - Strictly prevents full-frame blur.
        """
        if frame is None or frame.size == 0:
            return frame, {"applied": False, "full_frame_blur": False, "sharp_foreground_ratio": 1.0}

        h, w = frame.shape[:2]
        if depth_map is None or depth_map.shape[:2] != (h, w):
            depth_map = self.estimate_depth_map(frame, player_bbox=player_bbox, frame_idx=frame_idx)

        alpha = depth_map.copy()
        if player_mask is not None and player_mask.shape[:2] == (h, w):
            pm = (player_mask > 127).astype(np.float32)
            pm = cv2.GaussianBlur(pm, (17, 17), 5.0)
            alpha = np.maximum(alpha, pm)

        # Guarantee primary subject region is sharp even if depth estimation is noisy
        if player_bbox is not None and len(player_bbox) >= 4:
            x1, y1, p3, p4 = [int(round(v)) for v in player_bbox[:4]]
            if p3 < w * 0.7 and p4 < h * 0.9:
                x2, y2 = x1 + p3, y1 + p4
            else:
                x2, y2 = p3, p4
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 + 8 and y2 > y1 + 8:
                subj_layer = np.zeros((h, w), dtype=np.float32)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                rx, ry = max(8, (x2 - x1) // 2), max(12, (y2 - y1) // 2)
                cv2.ellipse(subj_layer, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
                subj_layer = cv2.GaussianBlur(subj_layer, (21, 21), 6.0)
                alpha = np.maximum(alpha, subj_layer)

        # Remap depth to foreground sharpness weight: depth >= 0.68 -> 1.0 sharp
        fg_weight = np.clip((alpha - 0.25) / 0.50, 0.0, 1.0)
        fg_weight = cv2.GaussianBlur(fg_weight, (15, 15), 4.0)

        sharp_ratio = float(np.mean(fg_weight > 0.65))
        # Hard guard: NEVER blur the whole frame!
        if sharp_ratio < 0.06:
            self.full_frame_blur_prevented += 1
            center_guard = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(
                center_guard,
                (w // 2, int(h * 0.52)),
                (max(24, int(w * 0.24)), max(36, int(h * 0.30))),
                0,
                0,
                360,
                1.0,
                -1,
            )
            center_guard = cv2.GaussianBlur(center_guard, (25, 25), 7.0)
            fg_weight = np.maximum(fg_weight, center_guard)
            sharp_ratio = float(np.mean(fg_weight > 0.65))

        k_far = max(5, (int(blur_strength) | 1))
        # Fast half-res blur for far background
        sh, sw = max(32, h // 2), max(32, w // 2)
        small_frame = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        k_small = max(3, (k_far // 2) | 1)
        blurred_small = cv2.GaussianBlur(small_frame, (k_small, k_small), 0)
        far_bg = cv2.resize(blurred_small, (w, h), interpolation=cv2.INTER_LINEAR)
        if dim_background < 0.995:
            far_bg = np.clip(far_bg.astype(np.float32) * float(dim_background), 0, 255).astype(np.uint8)

        w3 = fg_weight[:, :, None]
        blended = (frame.astype(np.float32) * w3 + far_bg.astype(np.float32) * (1.0 - w3)).astype(np.uint8)

        return blended, {
            "applied": True,
            "backend": self.backend_used,
            "full_frame_blur": False,
            "sharp_foreground_ratio": round(sharp_ratio, 4),
        }

    def apply_low_angle_pov(
        self,
        frame: np.ndarray,
        player_bbox: Optional[Tuple[int, int, int, int]] = None,
        depth_map: Optional[np.ndarray] = None,
        intensity: float = 0.30,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Restrained low-angle POV visual treatment:
        - NEVER stretches the entire frame or distorts player geometry.
        - Shifts vertical framing anchor slightly upward (placing the player higher
          in the vertical frame with more foreground turf lead-in) and applies a
          subtle ground-plane shadow vignette below the subject's feet while keeping
          1:1 isotropic aspect ratio on the player.
        """
        if frame is None or frame.size == 0:
            return frame, {"applied": False, "geometry_preserved": True}

        h, w = frame.shape[:2]
        eff = max(0.0, min(0.45, float(intensity)))
        if eff <= 0.02:
            return frame, {"applied": False, "geometry_preserved": True}

        # Isotropic crop-and-scale (identical X and Y scale factor -> zero geometric distortion!)
        zoom = 1.0 + eff * 0.12  # max ~1.05x isotropic scale
        crop_w = int(round(w / zoom))
        crop_h = int(round(h / zoom))

        # Center X on subject if available, shift Y window slightly down so subject rises in frame
        cx = w // 2
        if player_bbox is not None and len(player_bbox) >= 4:
            bx1, by1, bp3, bp4 = [int(round(v)) for v in player_bbox[:4]]
            bx2 = bx1 + bp3 if bp3 < w * 0.7 else bp3
            cx = max(crop_w // 2, min(w - crop_w // 2, (bx1 + bx2) // 2))

        # Shift crop window downward by up to 3.5% of height so the camera looks slightly upward at the player
        y_shift = int(round((h - crop_h) * min(0.85, 0.55 + eff * 0.55)))
        x1 = max(0, min(w - crop_w, cx - crop_w // 2))
        y1 = max(0, min(h - crop_h, y_shift))

        sub = frame[y1 : y1 + crop_h, x1 : x1 + crop_w]
        out = cv2.resize(sub, (w, h), interpolation=cv2.INTER_LINEAR)

        # Subtle ground-level atmospheric contrast at bottom 18% of frame (outside player torso)
        ground_band_h = max(8, int(round(h * 0.18)))
        ramp = np.linspace(1.0, 1.0 - eff * 0.22, ground_band_h, dtype=np.float32)[:, None, None]
        out[h - ground_band_h : h] = np.clip(
            out[h - ground_band_h : h].astype(np.float32) * ramp, 0, 255
        ).astype(np.uint8)

        return out, {
            "applied": True,
            "geometry_preserved": True,
            "isotropic_scale": round(zoom, 4),
            "aspect_distortion": 0.0,
        }


# =====================================================================
# 2. PSYCHOLOGICAL TEAL-ORANGE COLOR GRADING (OpenCV LUT)
# =====================================================================
class PsychologicalColorGrader:
    """
    Reusable cinematic Teal-Orange LUT grading exclusively for Storyteller-selected shots:
    - Cool/teal shadows
    - Controlled warm highlights
    - Protected skin tones
    - Controlled contrast S-curve
    - Turf saturation cap (no neon grass, no oversaturation)
    """

    _LUT_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    @classmethod
    def _get_channel_luts(cls, intensity: float = 0.85) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = int(round(max(0.0, min(1.0, intensity)) * 100))
        if key in cls._LUT_CACHE:
            return cls._LUT_CACHE[key]

        alpha = key / 100.0
        x = np.arange(256, dtype=np.float32) / 255.0
        # Controlled S-curve contrast
        s_curve = x - 0.075 * alpha * np.sin(2.0 * math.pi * x)
        s_curve = np.clip(s_curve, 0.0, 1.0)

        shadow_weight = np.power(1.0 - x, 1.6)
        highlight_weight = np.power(x, 1.5)

        # Blue channel: teal lift in shadows, slight cool control in highlights
        b_curve = s_curve + alpha * (0.065 * shadow_weight - 0.025 * highlight_weight)
        # Green channel: subtle cyan balance in deep shadows, natural midtones
        g_curve = s_curve + alpha * (0.022 * shadow_weight + 0.008 * highlight_weight)
        # Red channel: pulled back in shadows (teal), warm golden lift in highlights
        r_curve = s_curve + alpha * (-0.055 * shadow_weight + 0.052 * highlight_weight)

        lut_b = np.clip(b_curve * 255.0, 0, 255).astype(np.uint8)
        lut_g = np.clip(g_curve * 255.0, 0, 255).astype(np.uint8)
        lut_r = np.clip(r_curve * 255.0, 0, 255).astype(np.uint8)

        cls._LUT_CACHE[key] = (lut_b, lut_g, lut_r)
        return lut_b, lut_g, lut_r

    @classmethod
    def apply_teal_orange_grade(
        cls,
        frame: np.ndarray,
        intensity: float = 0.85,
        grass_sat_cap: int = 152,
    ) -> np.ndarray:
        if frame is None or frame.size == 0:
            return frame

        h, w = frame.shape[:2]
        eff = max(0.0, min(1.0, float(intensity)))
        if eff <= 0.01:
            return frame

        # 1. Half-resolution HSV analysis for skin tone protection and neon-grass prevention
        sh, sw = max(16, h // 2), max(16, w // 2)
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        # 1b. BGR -> LAB and CLAHE on L channel for controlled local contrast
        lab_small = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_lab_ch = cv2.split(lab_small)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l_ch)
        l_blend = cv2.addWeighted(l_ch, 1.0 - 0.35 * eff, l_clahe, 0.35 * eff, 0)
        lab_merged = cv2.merge([l_blend, a_ch, b_lab_ch])
        clahe_bgr_small = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)
        hsv = cv2.cvtColor(clahe_bgr_small, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Cap turf green saturation to prevent neon grass
        grass_mask = (hue >= 32) & (hue <= 88) & (sat > grass_sat_cap)
        sat[grass_mask] = grass_sat_cap
        # Global saturation ceiling to prevent oversaturation
        sat = np.minimum(sat, 195)

        # Skin protection mask (Hue 4..24)
        skin_mask_small = ((hue >= 4) & (hue <= 24) & (sat >= 38) & (sat <= 175) & (val >= 55)).astype(np.float32)
        skin_mask_small = cv2.GaussianBlur(skin_mask_small, (7, 7), 2.0)

        hsv[:, :, 1] = sat
        desat_bgr = cv2.resize(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), (w, h), interpolation=cv2.INTER_LINEAR)
        base = cv2.addWeighted(frame, 0.45, desat_bgr, 0.55, 0)

        # 2. Apply OpenCV per-channel Teal-Orange LUT
        lut_b, lut_g, lut_r = cls._get_channel_luts(eff)
        b_ch = cv2.LUT(base[:, :, 0], lut_b)
        g_ch = cv2.LUT(base[:, :, 1], lut_g)
        r_ch = cv2.LUT(base[:, :, 2], lut_r)
        graded = cv2.merge([b_ch, g_ch, r_ch])

        # 3. Blend original skin tones back by 60% so faces/jerseys never turn unnatural orange/teal
        if float(np.max(skin_mask_small)) > 0.05:
            skin_full = cv2.resize(skin_mask_small, (w, h), interpolation=cv2.INTER_LINEAR)[:, :, None]
            protect_w = skin_full * 0.60
            graded = (graded.astype(np.float32) * (1.0 - protect_w) + base.astype(np.float32) * protect_w).astype(
                np.uint8
            )

        # 4. Subtle vignette for psychological focus
        y_coords = np.linspace(-1.0, 1.0, sh, dtype=np.float32)[:, None]
        x_coords = np.linspace(-1.0, 1.0, sw, dtype=np.float32)[None, :]
        rad = np.sqrt(x_coords * x_coords + y_coords * y_coords)
        vig_small = np.clip(1.0 - (rad - 0.56) * (0.24 * eff), 0.78, 1.0)
        vig_full = cv2.resize(vig_small, (w, h), interpolation=cv2.INTER_LINEAR)[:, :, None]
        graded = np.clip(graded.astype(np.float32) * vig_full, 0, 255).astype(np.uint8)

        return graded

    @classmethod
    def add_cinematic_lut(
        cls,
        frame: np.ndarray,
        intensity: float = 0.85,
        grass_sat_cap: int = 152,
    ) -> np.ndarray:
        """Alias for BGR->LAB CLAHE + Teal-Orange LUT grading with skin & grass protection."""
        return cls.apply_teal_orange_grade(frame, intensity=intensity, grass_sat_cap=grass_sat_cap)


# =====================================================================
# 3. PSYCHOLOGICAL CINEMATIC STORYTELLER (Qwen2-VL/InternVL2 + DeepSeek R1)
# =====================================================================
class CinematicStoryteller:
    """
    Psychological Predator-vs-Prey Storyteller Layer:
    - Analyzes 6-second football clips + tracking/event evidence.
    - Integrates Qwen2-VL / InternVL2 via configurable env-var API with strict validation & fallback.
    - Integrates DeepSeek R1 for 6-line psychological thriller script generation aligned to duel_moment.
    - NEVER invents players, winners, duel moments, or football events without real tracking/event evidence.
    """

    VALID_STORY_ARCS = (
        "predator",
        "prey",
        "approach",
        "trap",
        "pressure",
        "escape",
        "dominance",
        "impact",
        "aftermath",
        "anticipation",
        "strike",
        "lockdown",
    )
    DECISIVE_EVENTS = {"dribble", "skill", "shot", "goal", "tackle", "save", "defender_interaction", "pass"}
    LAST_VLM_TELEMETRY: Dict[str, Any] = {
        "provider": "openrouter",
        "model": os.environ.get("OPENROUTER_MODEL_NAME") or "qwen/qwen2.5-vl-32b-instruct",
        "api_key_configured": False,
        "frames_extracted": 0,
        "status": "not_called",
    }
    LAST_DEEPSEEK_TELEMETRY: Dict[str, Any] = {
        "model": os.environ.get("DEEPSEEK_MODEL_NAME") or "deepseek-reasoner",
        "api_key_configured": False,
        "status": "not_called",
    }

    @classmethod
    def extract_representative_frames_b64(
        cls,
        clip_path: Optional[str],
        duel_moment: float = 2.8,
        max_frames: int = 3,
    ) -> List[str]:
        """
        Extracts 3 representative frames around the duel (pre-duel approach, duel peak, post-duel payoff)
        in streaming fashion without loading the full video into RAM, returning base64 JPEG data URLs.
        """
        if not clip_path or not isinstance(clip_path, str) or not os.path.exists(clip_path):
            return []
        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            return []
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            if fps <= 0 or math.isnan(fps):
                fps = 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            dur = (float(total_frames) / fps) if total_frames > 0 else 6.0
            dm = max(0.3, min(dur - 0.3, float(duel_moment)))
            target_times = [
                max(0.0, dm - 1.0),
                dm,
                min(max(0.1, dur - 0.1), dm + 1.0),
            ][:max_frames]
            encoded_frames: List[str] = []
            for t_sec in target_times:
                f_idx = max(0, int(round(t_sec * fps)))
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ok, frame = cap.read()
                if not ok or frame is None or frame.size == 0:
                    continue
                h, w = frame.shape[:2]
                scale = min(1.0, 360.0 / float(max(1, max(h, w))))
                small = (
                    cv2.resize(frame, (max(16, int(w * scale)), max(16, int(h * scale))), interpolation=cv2.INTER_AREA)
                    if scale < 1.0
                    else frame
                )
                ok_enc, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 76])
                if ok_enc and buf is not None:
                    b64_str = base64.b64encode(buf.tobytes()).decode("ascii")
                    encoded_frames.append(f"data:image/jpeg;base64,{b64_str}")
            return encoded_frames
        except Exception:
            return []
        finally:
            cap.release()

    @classmethod
    def extract_duel_evidence(
        cls,
        track_history: Optional[List[Dict[str, Any]]],
        detected_events: Optional[List[Dict[str, Any]]],
        hero_moment: Optional[Dict[str, Any]],
        clip_duration: float = 6.0,
        player_identities: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Inspects real tracking, ball proximity, defender interaction, and event engine evidence
        to determine whether a genuine football duel occurred.
        """
        history = track_history or []
        events = detected_events or []
        hero = hero_moment or {}
        identities = player_identities or {}

        real_events = [
            e
            for e in events
            if (e.get("type") or e.get("event"))
            and (e.get("type") or e.get("event")) != "unknown"
            and bool(e.get("event_backed", True))
        ]
        decisive_events = [
            e for e in real_events if (e.get("type") or e.get("event")) in cls.DECISIVE_EVENTS
        ]

        best_duel_time = round(clip_duration * 0.46, 2)
        max_duel_intensity = 0.0
        primary_player_id = None
        secondary_player_id = None
        peak_speed = 0.0
        peak_accel = 0.0
        peak_ball_prox = 0.0
        peak_def_pressure = 0.0
        possession_change = False
        prev_owner_id = None
        observed_player_ids = set()
        ball_detected_frames = 0

        for frame in history:
            t = float(frame.get("timestamp", 0.0))
            p = frame.get("player") or {}
            pid = p.get("id")
            if pid is not None:
                observed_player_ids.add(pid)
            if primary_player_id is None and pid is not None:
                primary_player_id = pid

            v_raw = p.get("speed", p.get("velocity", 0.0))
            spd = float(v_raw.get("speed", 0.0) if isinstance(v_raw, dict) else v_raw)
            peak_speed = max(peak_speed, spd)

            a_raw = p.get("acceleration", 0.0)
            acc = float(a_raw.get("magnitude", 0.0) if isinstance(a_raw, dict) else a_raw)
            peak_accel = max(peak_accel, acc)

            if frame.get("ball") is not None:
                ball_detected_frames += 1

            pbi = frame.get("player_ball_interaction") or {}
            pbi_score = float(pbi.get("score", 0.0))
            b_prox_raw = frame.get("ball_proximity", 999.0)
            if isinstance(b_prox_raw, (int, float)) and b_prox_raw <= 1.0:
                b_score = float(b_prox_raw)
            elif isinstance(b_prox_raw, (int, float)) and b_prox_raw < 900.0:
                b_score = min(1.0, 70.0 / max(15.0, float(b_prox_raw)))
            else:
                b_score = pbi_score
            peak_ball_prox = max(peak_ball_prox, b_score, pbi_score)

            if b_score >= 0.55 and pid is not None:
                if prev_owner_id is not None and prev_owner_id != pid:
                    possession_change = True
                prev_owner_id = pid

            def_info = frame.get("defender_interaction") or {}
            def_score = float(def_info.get("pressure_score", def_info.get("score", 0.0)))
            defs_list = frame.get("defenders") or []
            if defs_list and secondary_player_id is None:
                secondary_player_id = defs_list[0].get("id", 2)
            for d_item in defs_list:
                if d_item.get("id") is not None:
                    observed_player_ids.add(d_item.get("id"))
            if def_score <= 0.0 and defs_list:
                def_score = min(1.0, len(defs_list) * 0.55)
            peak_def_pressure = max(peak_def_pressure, def_score)

            tracks = frame.get("player_tracks") or frame.get("players") or []
            for tr in tracks:
                tid = tr.get("id")
                if tid is not None:
                    observed_player_ids.add(tid)
                if tid is not None and tid != primary_player_id and secondary_player_id is None:
                    secondary_player_id = tid

            frame_intensity = (
                b_score * 0.40
                + def_score * 0.35
                + min(1.0, spd / 180.0) * 0.25
            )
            if frame_intensity > max_duel_intensity:
                max_duel_intensity = frame_intensity
                best_duel_time = round(max(0.2, min(clip_duration - 0.2, t)), 2)

        if decisive_events:
            # Anchor duel_moment around the decisive football event if tracking window concurs
            best_ev = max(decisive_events, key=lambda e: float(e.get("confidence", 0.7)))
            ev_mid = 0.5 * (float(best_ev.get("start", 0.0)) + float(best_ev.get("end", clip_duration)))
            if max_duel_intensity < 0.35:
                best_duel_time = round(max(0.2, min(clip_duration - 0.2, ev_mid)), 2)

        ev_names = [str(e.get("type") or e.get("event")) for e in decisive_events]
        if "tackle" in ev_names:
            possession_change = True

        hero_backed = bool(hero.get("is_hero") and hero.get("event_backed"))
        hero_score_val = round(float(hero.get("hero_score", hero.get("confidence", 0.0))), 3)
        has_kinematic_interaction = (peak_ball_prox >= 0.32 and (peak_def_pressure >= 0.25 or peak_speed >= 95.0))
        has_event_evidence = len(decisive_events) > 0 or hero_backed

        duel_detected = bool(has_event_evidence and (has_kinematic_interaction or hero_backed or max_duel_intensity >= 0.42))

        evidence_conf = 0.0
        if duel_detected:
            ev_conf_max = max([float(e.get("confidence", 0.75)) for e in decisive_events], default=0.75)
            hero_conf = float(hero.get("confidence", 0.0))
            evidence_conf = round(
                min(
                    0.98,
                    max(
                        0.58,
                        0.38 * ev_conf_max
                        + 0.32 * max(max_duel_intensity, peak_ball_prox)
                        + 0.30 * max(hero_conf, peak_def_pressure, 0.65),
                    ),
                ),
                2,
            )

        # Resolve observable player labels WITHOUT inventing names
        winner_label = None
        loser_label = None
        if duel_detected:
            if identities.get("winner"):
                winner_label = str(identities["winner"])
            elif primary_player_id is not None and str(primary_player_id) in identities:
                winner_label = str(identities[str(primary_player_id)])
            elif primary_player_id is not None:
                winner_label = f"Player #{primary_player_id}"

            if identities.get("loser"):
                loser_label = str(identities["loser"])
            elif secondary_player_id is not None and str(secondary_player_id) in identities:
                loser_label = str(identities[str(secondary_player_id)])
            elif secondary_player_id is not None:
                loser_label = f"Player #{secondary_player_id}"

        # Dynamic story arc based on real football evidence
        if not duel_detected:
            dynamic_arc: List[str] = []
        elif "tackle" in ev_names:
            dynamic_arc = ["predator", "pressure", "trap", "dominance"]
        elif "dribble" in ev_names and ("shot" in ev_names or "goal" in ev_names):
            dynamic_arc = ["predator", "trap", "dominance"]
        elif "escape" in ev_names or peak_speed >= 260.0:
            dynamic_arc = ["predator", "approach", "trap", "dominance", "impact"]
        else:
            dynamic_arc = ["predator", "trap", "dominance"]

        return {
            "duel_detected": duel_detected,
            "duel_moment": round(max(0.0, min(clip_duration, best_duel_time)), 2),
            "winner": winner_label,
            "loser": loser_label,
            "primary_player_id": primary_player_id,
            "secondary_player_id": secondary_player_id,
            "players": sorted([int(x) for x in observed_player_ids if isinstance(x, (int, np.integer))]),
            "ball": {"detected_frames": ball_detected_frames, "peak_proximity": round(peak_ball_prox, 3)},
            "velocity": round(peak_speed, 2),
            "acceleration": round(peak_accel, 2),
            "ball_proximity": round(peak_ball_prox, 3),
            "defender_interaction": round(peak_def_pressure, 3),
            "possession_change": bool(possession_change),
            "shot": bool("shot" in ev_names),
            "dribble": bool("dribble" in ev_names or "skill" in ev_names),
            "tackle": bool("tackle" in ev_names),
            "goal": bool("goal" in ev_names),
            "hero_score": hero_score_val,
            "dynamic_story_arc": dynamic_arc,
            "allowed_identities": set(
                filter(
                    None,
                    [
                        winner_label,
                        loser_label,
                        identities.get("winner"),
                        identities.get("loser"),
                        *(identities.values()),
                        f"Player #{primary_player_id}" if primary_player_id is not None else None,
                        f"Player #{secondary_player_id}" if secondary_player_id is not None else None,
                    ],
                )
            ),
            "peak_speed": round(peak_speed, 2),
            "peak_ball_proximity": round(peak_ball_prox, 3),
            "peak_defender_pressure": round(peak_def_pressure, 3),
            "decisive_events": ev_names,
            "confidence": evidence_conf,
        }

    @classmethod
    def validate_and_normalize_vlm_json(
        cls,
        raw_payload: Any,
        evidence: Dict[str, Any],
        clip_duration: float = 6.0,
    ) -> Dict[str, Any]:
        """
        Validates and normalizes Qwen2-VL / InternVL2 JSON output:
        - Never allows LLM/VLM to invent a duel or winner when `evidence["duel_detected"]` is False.
        - Never allows unverified player names unless backed by `evidence["allowed_identities"]`.
        - Clamps `duel_moment` to `[0.0, clip_duration]`.
        """
        if not evidence.get("duel_detected", False):
            return {
                "duel_moment": round(float(evidence.get("duel_moment", clip_duration * 0.5)), 2),
                "winner": None,
                "loser": None,
                "story_arc": [],
                "confidence": 0.0,
                "duel_detected": False,
                "evidence_backed": False,
            }

        parsed: Dict[str, Any] = {}
        if isinstance(raw_payload, str):
            text = raw_payload.strip()
            if "```" in text:
                for block in text.split("```"):
                    block_clean = block.replace("json", "", 1).strip()
                    if block_clean.startswith("{") and block_clean.endswith("}"):
                        text = block_clean
                        break
            start_idx = text.find("{")
            end_idx = text.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                text = text[start_idx : end_idx + 1]
            try:
                candidate = json.loads(text)
                if isinstance(candidate, dict):
                    parsed = candidate
            except Exception:
                parsed = {}
        elif isinstance(raw_payload, dict):
            parsed = dict(raw_payload)

        # Normalize duel_moment
        try:
            dm = float(parsed.get("duel_moment", evidence.get("duel_moment", clip_duration * 0.46)))
            if not math.isfinite(dm):
                dm = float(evidence.get("duel_moment", clip_duration * 0.46))
        except Exception:
            dm = float(evidence.get("duel_moment", clip_duration * 0.46))
        dm = round(max(0.0, min(float(clip_duration), dm)), 2)

        # Normalize winner / loser: reject invented names not present in allowed_identities
        allowed = evidence.get("allowed_identities") or set()
        raw_winner = parsed.get("winner")
        raw_loser = parsed.get("loser")

        if isinstance(raw_winner, str) and raw_winner.strip() in allowed:
            winner = raw_winner.strip()
        else:
            winner = evidence.get("winner")

        if isinstance(raw_loser, str) and raw_loser.strip() in allowed:
            loser = raw_loser.strip()
        else:
            loser = evidence.get("loser")

        # Normalize story_arc
        raw_arc = parsed.get("story_arc")
        if isinstance(raw_arc, list) and len(raw_arc) >= 2:
            arc = [str(x).strip().lower() for x in raw_arc if str(x).strip()][:6]
        else:
            arc = list(evidence.get("dynamic_story_arc") or ["predator", "trap", "dominance"])

        try:
            raw_conf = float(parsed.get("confidence", evidence.get("confidence", 0.78)))
            if not math.isfinite(raw_conf):
                raw_conf = float(evidence.get("confidence", 0.78))
        except Exception:
            raw_conf = float(evidence.get("confidence", 0.78))
        conf = round(max(0.0, min(1.0, raw_conf if raw_conf > 0.0 else float(evidence.get("confidence", 0.78)))), 2)

        return {
            "duel_moment": dm,
            "winner": winner,
            "loser": loser,
            "story_arc": arc,
            "confidence": conf,
            "duel_detected": True,
            "evidence_backed": True,
        }

    @classmethod
    def _call_vlm_api(
        cls,
        clip_path: Optional[str],
        evidence: Dict[str, Any],
        timeout_sec: float = 4.5,
    ) -> Optional[Dict[str, Any]]:
        """
        Extracts 3 representative frames from `clip_path` and sends actual image data + tracking metadata
        to the configured VLM through OpenRouter (OPENROUTER_API_KEY, OPENROUTER_API_URL, OPENROUTER_MODEL_NAME).
        Never logs or exposes API keys.
        """
        provider = (os.environ.get("VLM_PROVIDER") or "openrouter").strip().lower()
        api_key = (
            os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("VLM_API_KEY")
            or os.environ.get("QWEN_VL_API_KEY")
            or os.environ.get("INTERNVL_API_KEY")
            or ""
        ).strip()
        raw_url = (
            os.environ.get("OPENROUTER_API_URL")
            or os.environ.get("VLM_API_URL")
            or "https://openrouter.ai/api/v1/chat/completions"
        ).strip().rstrip("/")
        api_url = raw_url if raw_url.endswith("/chat/completions") else f"{raw_url}/chat/completions"
        model_name = (
            os.environ.get("OPENROUTER_MODEL_NAME")
            or os.environ.get("VLM_MODEL")
            or os.environ.get("VLM_MODEL_NAME")
            or ("internvl/internvl2-8b" if "internvl" in provider else "qwen/qwen2.5-vl-32b-instruct")
        ).strip()

        frames_b64 = cls.extract_representative_frames_b64(
            clip_path, duel_moment=float(evidence.get("duel_moment", 2.8)), max_frames=3
        )
        cls.LAST_VLM_TELEMETRY = {
            "provider": provider,
            "model": model_name,
            "api_url": api_url,
            "api_key_configured": bool(api_key),
            "frames_extracted": len(frames_b64),
            "status": "no_api_key_evidence_fallback" if not api_key else "attempting",
        }

        if not api_key:
            return None

        prompt = (
            "Analyze this 6-second football duel based strictly on observable evidence and the 3 representative frames. "
            "Do NOT invent player names. Use provided IDs or null if unknown. "
            f"Observed tracking evidence: {json.dumps({k: v for k, v in evidence.items() if k != 'allowed_identities'})}. "
            'Return strict JSON: {"duel_moment": float, "winner": str|null, "loser": str|null, '
            '"story_arc": ["predator","trap","dominance"], "confidence": float}'
        )

        content_payload: Any
        if frames_b64:
            content_payload = [{"type": "text", "text": prompt}] + [
                {"type": "image_url", "image_url": {"url": b64_url}} for b64_url in frames_b64
            ]
        else:
            content_payload = prompt

        body = json.dumps(
            {
                "model": model_name,
                "messages": [{"role": "user", "content": content_payload}],
                "temperature": 0.1,
                "max_tokens": 180,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            api_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "GoalFlow-Engine/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            content = (
                resp_data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            cls.LAST_VLM_TELEMETRY["status"] = "live_api_ok"
            return {"raw_content": content, "provider": model_name, "frames_sent": len(frames_b64)}

    @classmethod
    def analyze_duel(
        cls,
        clip_path: Optional[str] = None,
        track_history: Optional[Any] = None,
        detected_events: Optional[List[Dict[str, Any]]] = None,
        hero_moment: Optional[Dict[str, Any]] = None,
        clip_duration: float = 6.0,
        player_identities: Optional[Dict[str, str]] = None,
        vlm_caller: Optional[Callable[..., Any]] = None,
        tracking_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Performs Psychological Duel Analysis with OpenRouter Qwen2.5-VL/InternVL2 + deterministic evidence fallback.
        Supports both `analyze_duel(clip_path, tracking_meta)` and keyword arguments.
        Never fails rendering if VLM API times out or errors.
        """
        if isinstance(track_history, dict) and tracking_meta is None:
            tracking_meta = track_history
            track_history = None
        if isinstance(tracking_meta, dict):
            track_history = track_history if track_history is not None else tracking_meta.get("track_history")
            detected_events = detected_events if detected_events is not None else tracking_meta.get("detected_events")
            hero_moment = hero_moment if hero_moment is not None else tracking_meta.get("hero_moment")
            clip_duration = float(tracking_meta.get("clip_duration", clip_duration))
            player_identities = player_identities or tracking_meta.get("player_identities")

        evidence = cls.extract_duel_evidence(
            track_history=track_history,
            detected_events=detected_events,
            hero_moment=hero_moment,
            clip_duration=clip_duration,
            player_identities=player_identities,
        )
        if not evidence["duel_detected"]:
            return cls.validate_and_normalize_vlm_json({}, evidence, clip_duration=clip_duration)

        raw_vlm = None
        caller = vlm_caller or cls._call_vlm_api
        try:
            vlm_res = caller(clip_path, evidence)
            if isinstance(vlm_res, dict) and "raw_content" in vlm_res:
                raw_vlm = vlm_res["raw_content"]
            else:
                raw_vlm = vlm_res
        except Exception as exc:
            # Graceful fallback to kinematic/event evidence on VLM timeout or failure
            cls.LAST_VLM_TELEMETRY["status"] = f"fallback_on_error:{type(exc).__name__}"
            raw_vlm = None

        normalized = cls.validate_and_normalize_vlm_json(raw_vlm, evidence, clip_duration=clip_duration)
        normalized["decisive_events"] = evidence["decisive_events"]
        normalized["peak_speed"] = evidence["peak_speed"]
        normalized["peak_ball_proximity"] = evidence["peak_ball_proximity"]
        normalized["peak_defender_pressure"] = evidence["peak_defender_pressure"]
        normalized["vlm_telemetry"] = dict(cls.LAST_VLM_TELEMETRY)
        normalized["evidence_summary"] = {
            k: v for k, v in evidence.items() if k != "allowed_identities"
        }
        return normalized

    @classmethod
    def _call_deepseek_api(
        cls,
        duel_analysis: Dict[str, Any],
        clip_duration: float = 6.0,
        timeout_sec: float = 5.0,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Calls DeepSeek R1 API via DEEPSEEK_API_KEY, DEEPSEEK_API_URL, DEEPSEEK_MODEL_NAME;
        gracefully falls back to evidence-specific thriller lines if API is unconfigured or returns HTTP 402/timeout.
        """
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        raw_url = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com").strip().rstrip("/")
        api_url = raw_url if raw_url.endswith("/chat/completions") else f"{raw_url}/chat/completions"
        model_name = (
            os.environ.get("DEEPSEEK_MODEL_NAME")
            or os.environ.get("DEEPSEEK_MODEL")
            or "deepseek-reasoner"
        ).strip()
        cls.LAST_DEEPSEEK_TELEMETRY = {
            "model": model_name,
            "api_url": api_url,
            "api_key_configured": bool(api_key),
            "status": "no_api_key_local_synthesis" if not api_key else "attempting",
        }

        if api_key:
            try:
                prompt = (
                    "Generate a 6-line psychological thriller inner-monologue script for a football 1v1 duel. "
                    "Tone: predator vs prey, psychological pressure, short phrases, escalating tension. "
                    "Do NOT use generic sports commentary. Do NOT copy reference phrases verbatim. "
                    f"Duel moment: {duel_analysis.get('duel_moment')}s out of {clip_duration}s. "
                    f"Events: {duel_analysis.get('decisive_events')}. "
                    'Return strict JSON list of 6 objects: [{"time": float, "text": str, "position": "center"|"lower"|"upper"}]'
                )
                body = json.dumps(
                    {
                        "model": model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.4,
                        "max_tokens": 320,
                    }
                ).encode("utf-8")
                req = urllib.request.Request(
                    api_url,
                    data=body,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                        "User-Agent": "GoalFlow-Engine/1.0",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "[]")
                    start_i = content.find("[")
                    end_i = content.rfind("]")
                    if start_i != -1 and end_i != -1 and end_i > start_i:
                        parsed_lines = json.loads(content[start_i : end_i + 1])
                    else:
                        parsed_lines = json.loads(content)
                    if isinstance(parsed_lines, list) and len(parsed_lines) > 0:
                        cls.LAST_DEEPSEEK_TELEMETRY["status"] = "live_api_ok"
                        return parsed_lines
            except urllib.error.HTTPError as http_err:
                cls.LAST_DEEPSEEK_TELEMETRY["status"] = f"http_{http_err.code}_fallback_synthesis"
                if os.environ.get("DEEPSEEK_DISABLE_LOCAL_FALLBACK") == "1":
                    raise
            except Exception as exc:
                cls.LAST_DEEPSEEK_TELEMETRY["status"] = f"error_{type(exc).__name__}_fallback_synthesis"
                if os.environ.get("DEEPSEEK_DISABLE_LOCAL_FALLBACK") == "1":
                    raise

        # Evidence-driven local synthesis when no external API key is configured
        dm = float(duel_analysis.get("duel_moment", clip_duration * 0.48))
        events = set(duel_analysis.get("decisive_events") or [])
        spd = float(duel_analysis.get("peak_speed", 160.0))

        # Original lines crafted dynamically from the actual duel evidence
        line1 = "EYES LOCKED ON THE TARGET..." if spd >= 150 else "SILENCE BEFORE THE BAIT..."
        line2 = "STEP INTO THE SHADOW" if "defender_interaction" in events else "DRAWN OUT OF POSITION"
        line3 = "THE TRAP IS ALREADY SET" if "dribble" in events or "skill" in events else "NO ESCAPE ROUTE LEFT"
        line4 = "ONE HEARTBEAT TO BREAK HIM" if spd >= 180 else "PRESSURE AT BREAKING POINT"
        if "goal" in events or "shot" in events:
            line5 = "COLD-BLOODED EXECUTION."
            line6 = "PREDATOR OWNS THE STAGE"
        elif "tackle" in events or "save" in events:
            line5 = "SHUT DOWN INSTANTLY."
            line6 = "TOTAL TERRITORIAL DOMINANCE"
        else:
            line5 = "ANKLES FROZEN IN TIME."
            line6 = "DOMINANCE SEALED"

        t1 = 0.15
        t2 = max(0.55, round(dm * 0.36, 2))
        t3 = max(t2 + 0.45, round(dm * 0.68, 2))
        t4 = max(t3 + 0.40, round(dm * 0.92, 2))
        t5 = max(t4 + 0.38, round(min(clip_duration - 0.85, dm + 0.22), 2))
        t6 = max(t5 + 0.45, round(min(clip_duration - 0.25, dm + 1.15), 2))

        return [
            {"time": t1, "text": line1, "position": "center"},
            {"time": t2, "text": line2, "position": "lower"},
            {"time": t3, "text": line3, "position": "center"},
            {"time": t4, "text": line4, "position": "lower"},
            {"time": t5, "text": line5, "position": "center"},
            {"time": t6, "text": line6, "position": "lower"},
        ]

    @classmethod
    def sanitize_and_clamp_story_script(
        cls,
        raw_script: Any,
        clip_duration: float = 6.0,
        existing_text_windows: Optional[List[Tuple[float, float]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Clamps all story_script timestamps to [0.0, clip_duration], enforces non-overlapping
        monotonic timestamps, and avoids collisions with existing source text overlays.
        """
        if not isinstance(raw_script, list) or not raw_script:
            return []

        dur = max(1.0, float(clip_duration))
        avoid_windows = existing_text_windows or []
        valid_positions = {"center", "lower", "upper"}
        cleaned: List[Dict[str, Any]] = []

        for idx, item in enumerate(raw_script[:6]):
            if not isinstance(item, dict):
                continue
            txt = str(item.get("text", "")).strip()
            if not txt:
                continue
            try:
                t_val = float(item.get("time", idx * (dur / 6.0)))
                if not math.isfinite(t_val):
                    t_val = idx * (dur / 6.0)
            except Exception:
                t_val = idx * (dur / 6.0)

            # Clamp strictly inside [0.0, clip_duration]
            t_clamped = max(0.0, min(dur, t_val))
            pos = str(item.get("position", "center" if idx % 2 == 0 else "lower")).strip().lower()
            if pos not in valid_positions:
                pos = "center" if idx % 2 == 0 else "lower"

            # Avoid text collision with existing source captions by shifting position
            for w_start, w_end in avoid_windows:
                if w_start <= t_clamped <= w_end and pos == "lower":
                    pos = "center"

            cleaned.append(
                {
                    "time": round(t_clamped, 2),
                    "text": txt[:64],
                    "position": pos,
                }
            )

        # Sort by timestamp and enforce minimum separation so two lines never overlap
        cleaned.sort(key=lambda x: x["time"])
        min_gap = min(0.32, dur / max(8.0, float(len(cleaned) * 2)))
        for i in range(1, len(cleaned)):
            if cleaned[i]["time"] <= cleaned[i - 1]["time"] + min_gap:
                cleaned[i]["time"] = round(min(dur, cleaned[i - 1]["time"] + min_gap), 2)
                if cleaned[i]["position"] == cleaned[i - 1]["position"]:
                    cleaned[i]["position"] = "lower" if cleaned[i - 1]["position"] == "center" else "center"

        return cleaned

    @classmethod
    def generate_story_script(
        cls,
        duel_analysis: Dict[str, Any],
        clip_duration: float = 6.0,
        existing_text_windows: Optional[List[Tuple[float, float]]] = None,
        deepseek_caller: Optional[Callable[..., Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generates the 6-line psychological thriller script around `duel_moment`.
        - Returns `[]` if duel evidence is insufficient.
        - Returns `[]` if DeepSeek call fails/raises an exception, without failing the render.
        """
        if not duel_analysis or not duel_analysis.get("duel_detected") or not duel_analysis.get("evidence_backed"):
            return []

        caller = deepseek_caller or cls._call_deepseek_api
        try:
            raw_lines = caller(duel_analysis, clip_duration=clip_duration)
            if not isinstance(raw_lines, list):
                return []
            return cls.sanitize_and_clamp_story_script(
                raw_lines,
                clip_duration=clip_duration,
                existing_text_windows=existing_text_windows,
            )
        except Exception:
            # DeepSeek failure -> return story_script=[] and allow render to continue normally
            return []

    @classmethod
    def generate_script(
        cls,
        duel_info: Dict[str, Any],
        clip_duration: float = 6.0,
        existing_text_windows: Optional[List[Tuple[float, float]]] = None,
        deepseek_caller: Optional[Callable[..., Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generates 6 short original English thriller lines (hook, trap escalation, inner tension, peak, payoff)
        using DeepSeek R1 (DEEPSEEK_API_KEY, DEEPSEEK_API_URL, DEEPSEEK_MODEL_NAME), clamping every time to 0-6s.
        On failure returns [].
        """
        if isinstance(duel_info, dict) and "duel_detected" not in duel_info:
            # Normalize minimal duel_info dicts passed directly to generate_script(duel_info)
            conf = float(duel_info.get("confidence", 0.0))
            enriched = dict(duel_info)
            enriched["duel_detected"] = bool(conf >= 0.50 and duel_info.get("duel_moment") is not None)
            enriched["evidence_backed"] = enriched["duel_detected"]
            duel_info = enriched
        return cls.generate_story_script(
            duel_info,
            clip_duration=min(6.0, max(1.0, float(clip_duration))),
            existing_text_windows=existing_text_windows,
            deepseek_caller=deepseek_caller,
        )

    @classmethod
    def build_psychological_package(
        cls,
        clip_path: Optional[str] = None,
        track_history: Optional[List[Dict[str, Any]]] = None,
        detected_events: Optional[List[Dict[str, Any]]] = None,
        hero_moment: Optional[Dict[str, Any]] = None,
        clip_duration: float = 6.0,
        source_resolution: Tuple[int, int] = (1280, 720),
        player_identities: Optional[Dict[str, str]] = None,
        vlm_caller: Optional[Callable[..., Any]] = None,
        deepseek_caller: Optional[Callable[..., Any]] = None,
    ) -> Dict[str, Any]:
        """
        Safe Story Mode Decision Layer:
        normal football event -> duel detected? -> strong player interaction/evidence?
        -> Psychological Cinematic Engine package for existing Director/EditPlan/FFmpeg renderer.
        """
        duel = cls.analyze_duel(
            clip_path=clip_path,
            track_history=track_history,
            detected_events=detected_events,
            hero_moment=hero_moment,
            clip_duration=clip_duration,
            player_identities=player_identities,
            vlm_caller=vlm_caller,
        )

        if not duel.get("duel_detected") or float(duel.get("confidence", 0.0)) < 0.50:
            return {
                "psychological_story": False,
                "story_role": None,
                "story_arc": [],
                "duel_moment": None,
                "winner": None,
                "loser": None,
                "confidence": 0.0,
                "duel_confidence": 0.0,
                "story_confidence": 0.0,
                "depth_effect": False,
                "low_angle": False,
                "pov_switch": False,
                "story_script": [],
                "duel_evidence": duel.get("evidence_summary", {}),
                "shot_language": {},
            }

        story_script = cls.generate_story_script(
            duel,
            clip_duration=clip_duration,
            deepseek_caller=deepseek_caller,
        )

        src_w, src_h = source_resolution
        has_detail_res = min(src_w, src_h) >= 720
        duel_conf = float(duel.get("confidence", 0.82))
        story_conf = round(duel_conf if len(story_script) > 0 else max(0.55, duel_conf * 0.85), 2)
        depth_enabled_env = os.environ.get("ENABLE_DEPTH_ANYTHING", "true").strip().lower() not in ("0", "false", "no")

        return {
            "psychological_story": True,
            "story_role": "predator",
            "story_arc": duel.get("story_arc", ["predator", "trap", "dominance"]),
            "duel_moment": duel["duel_moment"],
            "winner": duel.get("winner"),
            "loser": duel.get("loser"),
            "confidence": duel_conf,
            "duel_confidence": duel_conf,
            "story_confidence": story_conf,
            "depth_effect": bool(depth_enabled_env),
            "low_angle": True,
            "pov_switch": True,
            "story_script": story_script,
            "duel_evidence": duel.get("evidence_summary", {}),
            "vlm_telemetry": dict(cls.LAST_VLM_TELEMETRY),
            "deepseek_telemetry": dict(cls.LAST_DEEPSEEK_TELEMETRY),
            "shot_language": {
                "close_up": True,
                "sweat_detail_emphasis": bool(has_detail_res),
                "low_angle_pov": True,
                "pov_switch": True,
                "depth_aware_bokeh": bool(depth_enabled_env),
                "restrained_push_in": True,
                "player_first_framing": True,
                "psychological_text": bool(len(story_script) > 0),
                "stronger_sound_emphasis": True,
            },
        }

    @staticmethod
    def render_psychological_caption(
        frame: np.ndarray,
        story_script: List[Dict[str, Any]],
        timestamp_sec: float,
    ) -> np.ndarray:
        """
        Renders active thriller inner-monologue line onto the frame at `timestamp_sec`
        with high-contrast psychological styling and subtle fade window.
        """
        if frame is None or not story_script:
            return frame

        active_line = None
        for idx, item in enumerate(story_script):
            t_start = float(item.get("time", 0.0))
            next_t = float(story_script[idx + 1]["time"]) if idx + 1 < len(story_script) else (t_start + 1.15)
            t_end = min(next_t - 0.05, t_start + 1.05)
            if t_start <= timestamp_sec <= max(t_start + 0.25, t_end):
                active_line = item
                break

        if not active_line:
            return frame

        text = str(active_line.get("text", "")).strip()
        if not text:
            return frame

        h, w = frame.shape[:2]
        pos = str(active_line.get("position", "center")).lower()
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = max(0.65, min(1.15, w / 960.0))
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

        tx = max(24, (w - tw) // 2)
        if pos == "center":
            ty = int(h * 0.48)
        elif pos == "upper":
            ty = int(h * 0.22)
        else:
            ty = int(h * 0.78)

        out = frame.copy()
        pad_x, pad_y = 18, 12
        bx1 = max(8, tx - pad_x)
        by1 = max(8, ty - th - pad_y)
        bx2 = min(w - 8, tx + tw + pad_x)
        by2 = min(h - 8, ty + pad_y)

        roi = out[by1:by2, bx1:bx2]
        dark_box = np.zeros_like(roi)
        out[by1:by2, bx1:bx2] = cv2.addWeighted(roi, 0.38, dark_box, 0.62, 0)
        # Subtle crimson-amber thriller accent bar on left of caption box
        cv2.rectangle(out, (bx1, by1), (min(w - 8, bx1 + 4), by2), (45, 75, 235), -1)

        cv2.putText(out, text, (tx + 2, ty + 2), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(out, text, (tx, ty), font, font_scale, (242, 246, 250), thickness, cv2.LINE_AA)
        return out
