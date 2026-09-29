import os
import urllib.request
import cv2
import numpy as np

try:
    import torch
    try:
        torch.backends.nnpack.enabled = False
    except Exception:
        pass
    from mobile_sam import SamPredictor, sam_model_registry
    HAS_MOBILESAM = True
except Exception:
    torch = None
    SamPredictor = None
    sam_model_registry = None
    HAS_MOBILESAM = False

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except Exception:
    YOLO = None
    HAS_ULTRALYTICS = False

MOBILESAM_WEIGHTS_URLS = {
    "mobile_sam.pt": "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
    "sam_vit_b_01ec64.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}

# Global caches so heavy models are loaded at most ONCE per process
_GLOBAL_SAM_PREDICTOR_CACHE = {}
_GLOBAL_YOLO_DETECTOR_CACHE = {}


def get_cached_yolo_model(model_name: str = "yolov8n.pt"):
    """Loads and caches the YOLO model once in memory for low-RAM reuse."""
    if not HAS_ULTRALYTICS or YOLO is None:
        return None
    if model_name in _GLOBAL_YOLO_DETECTOR_CACHE:
        return _GLOBAL_YOLO_DETECTOR_CACHE[model_name]
    try:
        model = YOLO(model_name)
        _GLOBAL_YOLO_DETECTOR_CACHE[model_name] = model
        return model
    except Exception:
        _GLOBAL_YOLO_DETECTOR_CACHE[model_name] = None
        return None


class PlayerTracker:
    """
    Production-grade AI Football Player Segmentation & Subject Isolation.
    Strict 4-Tier Fallback Hierarchy:
      1. MobileSAM (vit_t / mobile_sam.pt or vit_b) with box + point prompts
      2. YOLO player bounding-box / anatomical articulated silhouette mask
      3. Motion contour (MOG2 foreground segmentation with hull refinement)
      4. No isolation ("no_isolation" — never full-frame blur; segmentation failure never fails render)
    """

    FALLBACK_CHAIN = ("mobilesam", "yolo", "motion", "no_isolation")

    def __init__(self, checkpoint_path: str = "mobile_sam.pt", model_type: str = "vit_t", lazy_load: bool = True):
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.predictor = None
        self.is_loaded = False
        self.last_valid_bbox = None
        self.last_isolation_method = "no_isolation"
        self.mask_error_count = 0
        self.frames_isolated = 0
        self.method_counts = {
            "mobilesam": 0,
            "yolo": 0,
            "motion": 0,
            "no_isolation": 0,
        }
        self._cached_mask = None
        self._cached_mask_shape = None
        self._cached_bbox = None
        self._mask_call_counter = 0
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=150, varThreshold=24, detectShadows=False
        )
        if not lazy_load and HAS_MOBILESAM:
            self._init_mobile_sam()

    def ensure_loaded(self) -> None:
        if self.is_loaded:
            return
        self.is_loaded = True
        if HAS_MOBILESAM:
            self._init_mobile_sam()

    def _init_mobile_sam(self) -> None:
        cache_key = f"{self.checkpoint_path}:{self.model_type}"
        if cache_key in _GLOBAL_SAM_PREDICTOR_CACHE:
            self.predictor = _GLOBAL_SAM_PREDICTOR_CACHE[cache_key]
            return

        resolved_ckpt = self.checkpoint_path
        if not os.path.exists(resolved_ckpt):
            alt_ckpt = "sam_vit_b_01ec64.pth"
            if os.path.exists(alt_ckpt):
                resolved_ckpt = alt_ckpt
                self.model_type = "vit_b"
            else:
                url = MOBILESAM_WEIGHTS_URLS.get("mobile_sam.pt")
                try:
                    urllib.request.urlretrieve(url, "mobile_sam.pt")
                    resolved_ckpt = "mobile_sam.pt"
                    self.model_type = "vit_t"
                except Exception:
                    _GLOBAL_SAM_PREDICTOR_CACHE[cache_key] = None
                    return
        try:
            device = "cpu"
            sam = sam_model_registry[self.model_type](checkpoint=resolved_ckpt)
            sam.to(device=device)
            sam.eval()
            self.predictor = SamPredictor(sam)
            _GLOBAL_SAM_PREDICTOR_CACHE[cache_key] = self.predictor
        except Exception:
            self.predictor = None
            _GLOBAL_SAM_PREDICTOR_CACHE[cache_key] = None

    def _detect_yolo_player_bbox(self, frame: np.ndarray):
        """Uses cached YOLO model to locate primary player if bbox was not supplied."""
        h, w = frame.shape[:2]
        yolo = get_cached_yolo_model("yolov8n.pt")
        if yolo is None:
            return None
        try:
            scale = min(1.0, 480.0 / float(max(h, w)))
            small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame
            results = yolo.predict(small, classes=[0], conf=0.25, verbose=False)
            if not results or len(results[0].boxes) == 0:
                return None
            boxes = results[0].boxes.xyxy.cpu().numpy() / scale
            ref_cx = (self.last_valid_bbox[0] + self.last_valid_bbox[2]) / 2.0 if self.last_valid_bbox is not None else w * 0.5
            ref_cy = (self.last_valid_bbox[1] + self.last_valid_bbox[3]) / 2.0 if self.last_valid_bbox is not None else h * 0.5
            best_box = min(
                boxes,
                key=lambda b: ((b[0] + b[2]) * 0.5 - ref_cx) ** 2 + ((b[1] + b[3]) * 0.5 - ref_cy) ** 2
            )
            return np.array(best_box, dtype=np.float32)
        except Exception:
            return None

    def _detect_motion_contour(self, frame: np.ndarray):
        """Tier 3: MOG2 foreground motion segmentation when YOLO bbox is unavailable."""
        h, w = frame.shape[:2]
        fg_mask = self.bg_subtractor.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = (w * h) * 0.002
        max_area = (w * h) * 0.38
        valid = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]
        if not valid:
            return None, None

        if self.last_valid_bbox is not None:
            prev_cx = (self.last_valid_bbox[0] + self.last_valid_bbox[2]) / 2.0
            prev_cy = (self.last_valid_bbox[1] + self.last_valid_bbox[3]) / 2.0

            def dist_to_prev(c):
                m = cv2.moments(c)
                if m["m00"] == 0:
                    return float("inf")
                cx = m["m10"] / m["m00"]
                cy = m["m01"] / m["m00"]
                return (cx - prev_cx) ** 2 + (cy - prev_cy) ** 2

            best = min(valid, key=dist_to_prev)
        else:
            best = max(valid, key=cv2.contourArea)

        x, y, bw, bh = cv2.boundingRect(best)
        pad_x = int(bw * 0.12)
        pad_y = int(bh * 0.12)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w - 1, x + bw + pad_x)
        y2 = min(h - 1, y + bh + pad_y)
        bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
        self.last_valid_bbox = bbox
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [best], -1, 255, thickness=cv2.FILLED)
        hull = cv2.convexHull(best)
        cv2.drawContours(mask, [hull], -1, 255, thickness=cv2.FILLED)
        return bbox, mask

    def auto_detect_player_bbox(self, frame: np.ndarray):
        """
        Locates player when bbox is not supplied.
        Hierarchy: YOLO -> Motion contour -> None (no_isolation).
        Never invents a fake center box on empty/static frames.
        """
        h, w = frame.shape[:2]
        yolo_box = self._detect_yolo_player_bbox(frame)
        if yolo_box is not None:
            self.last_valid_bbox = yolo_box
            mask = self._build_articulated_player_mask(h, w, yolo_box)
            return yolo_box, mask, "yolo"

        motion_box, motion_mask = self._detect_motion_contour(frame)
        if motion_box is not None and motion_mask is not None:
            return motion_box, motion_mask, "motion"

        return None, None, "no_isolation"

    @staticmethod
    def _build_articulated_player_mask(h: int, w: int, box_arr: np.ndarray, ball_xy: tuple = None) -> np.ndarray:
        """Creates an anatomically proportioned head+shoulders+torso+legs silhouette mask from a YOLO/tracked bbox."""
        mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = [int(round(v)) for v in box_arr[:4]]
        x1 = max(0, min(w - 2, x1))
        y1 = max(0, min(h - 2, y1))
        x2 = max(x1 + 4, min(w, x2))
        y2 = max(y1 + 8, min(h, y2))
        bw = max(12, x2 - x1)
        bh = max(24, y2 - y1)
        cx = (x1 + x2) // 2

        # Torso & lower body primary ellipse
        cv2.ellipse(mask, (cx, int(y1 + bh * 0.56)), (int(bw * 0.45), int(bh * 0.43)), 0, 0, 360, 255, -1)
        # Chest & shoulders ellipse
        cv2.ellipse(mask, (cx, int(y1 + bh * 0.34)), (int(bw * 0.48), int(bh * 0.22)), 0, 0, 360, 255, -1)
        # Head ellipse
        cv2.ellipse(mask, (cx, int(y1 + bh * 0.14)), (int(bw * 0.24), int(bh * 0.13)), 0, 0, 360, 255, -1)

        if ball_xy is not None and len(ball_xy) >= 2:
            bx, by = int(round(ball_xy[0])), int(round(ball_xy[1]))
            br = max(10, int(min(bw, bh) * 0.18))
            if 0 <= bx < w and 0 <= by < h:
                cv2.circle(mask, (bx, by), br, 255, -1)
        return mask

    @staticmethod
    def _is_valid_mask(mask: np.ndarray) -> bool:
        if mask is None or mask.size == 0:
            return False
        fg_ratio = float(np.count_nonzero(mask > 127)) / float(mask.size)
        # Reject empty masks (<0.2%) or full-frame masks (>82%) to prevent full-frame blur
        return 0.002 <= fg_ratio <= 0.82

    def get_player_mask(self, frame: np.ndarray, bbox=None, ball_xy: tuple = None) -> np.ndarray:
        """
        Generates binary mask of the player following the strict hierarchy:
          1. MobileSAM predictor (with box + foreground/background point prompts)
          2. YOLO articulated player silhouette mask
          3. Motion contour mask (MOG2)
          4. No isolation (empty mask -> caller skips blur)
        """
        h, w = frame.shape[:2]
        self._mask_call_counter += 1

        if bbox is not None:
            box_arr = np.array(bbox[:4], dtype=np.float32)
            # Propagate cached MobileSAM/YOLO mask across intermediate frames via affine shift
            if (
                self._cached_mask is not None
                and self._cached_mask_shape == (h, w)
                and self._cached_bbox is not None
                and (self._mask_call_counter % 24 != 1)
            ):
                old_cx = (self._cached_bbox[0] + self._cached_bbox[2]) * 0.5
                old_cy = (self._cached_bbox[1] + self._cached_bbox[3]) * 0.5
                new_cx = (box_arr[0] + box_arr[2]) * 0.5
                new_cy = (box_arr[1] + box_arr[3]) * 0.5
                dx = float(new_cx - old_cx)
                dy = float(new_cy - old_cy)
                if abs(dx) < 2.0 and abs(dy) < 2.0:
                    return self._cached_mask
                M = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
                shifted = cv2.warpAffine(self._cached_mask, M, (w, h), flags=cv2.INTER_NEAREST)
                if self._is_valid_mask(shifted):
                    return shifted
            self.last_valid_bbox = box_arr
            detected_source = "yolo"
            contour_mask = None
        else:
            box_arr, contour_mask, detected_source = self.auto_detect_player_bbox(frame)

        # Tier 1: MobileSAM (box + positive torso point) — lazy-loaded on demand
        if not self.is_loaded:
            self.ensure_loaded()
        if self.predictor is not None and torch is not None and box_arr is not None:
            try:
                max_dim = 320
                scale = min(1.0, float(max_dim) / float(max(h, w)))
                if scale < 1.0:
                    small_w, small_h = int(round(w * scale)), int(round(h * scale))
                    rgb_small = cv2.cvtColor(
                        cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2RGB,
                    )
                    scaled_box = box_arr * scale
                else:
                    rgb_small = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    scaled_box = box_arr

                bx1, by1, bx2, by2 = scaled_box
                tcx = (bx1 + bx2) * 0.5
                tcy = by1 + (by2 - by1) * 0.45
                point_coords = np.array([[tcx, tcy]], dtype=np.float32)
                point_labels = np.array([1], dtype=np.int32)

                with torch.no_grad():
                    self.predictor.set_image(rgb_small)
                    masks, scores, _ = self.predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        box=scaled_box[None, :],
                        multimask_output=False,
                    )
                    best_mask = (masks[0] > 0).astype(np.uint8) * 255
                    self.predictor.reset_image()
                if scale < 1.0:
                    best_mask = cv2.resize(best_mask, (w, h), interpolation=cv2.INTER_LINEAR)
                    _, best_mask = cv2.threshold(best_mask, 127, 255, cv2.THRESH_BINARY)
                del rgb_small, masks, scores
                if self._is_valid_mask(best_mask):
                    self.last_isolation_method = "mobilesam"
                    self.method_counts["mobilesam"] = self.method_counts.get("mobilesam", 0) + 1
                    self._cached_mask = best_mask
                    self._cached_mask_shape = (h, w)
                    self._cached_bbox = box_arr.copy()
                    return best_mask
            except Exception:
                pass

        # Tier 2: YOLO / Tracked BBox Articulated Silhouette Mask
        if box_arr is not None and detected_source == "yolo":
            yolo_mask = self._build_articulated_player_mask(h, w, box_arr, ball_xy=ball_xy)
            if self._is_valid_mask(yolo_mask):
                self.last_isolation_method = "yolo"
                self.method_counts["yolo"] = self.method_counts.get("yolo", 0) + 1
                self._cached_mask = yolo_mask
                self._cached_mask_shape = (h, w)
                self._cached_bbox = box_arr.copy()
                return yolo_mask

        # Tier 3: Motion Contour (MOG2)
        if contour_mask is None and box_arr is None:
            _, contour_mask = self._detect_motion_contour(frame)
        if contour_mask is not None and self._is_valid_mask(contour_mask):
            self.last_isolation_method = "motion"
            self.method_counts["motion"] = self.method_counts.get("motion", 0) + 1
            self._cached_mask = contour_mask
            self._cached_mask_shape = (h, w)
            return contour_mask

        # Tier 4: No Isolation (never full-frame blur)
        self.last_isolation_method = "no_isolation"
        self.method_counts["no_isolation"] = self.method_counts.get("no_isolation", 0) + 1
        return np.zeros((h, w), dtype=np.uint8)

    def apply_cinematic_blur(
        self,
        frame: np.ndarray,
        bbox=None,
        blur_kernel: tuple = (21, 21),
        dim_ratio: float = 0.88,
        feather_radius: int = 17,
        ball_xy: tuple = None,
    ) -> np.ndarray:
        """
        Subject Isolation:
        - Fallback hierarchy: MobileSAM -> YOLO -> motion -> no_isolation
        - Sharp foreground subject + feathered subtle background blur/dim
        - Never applies full-frame blur and never lets segmentation failure fail the render
        """
        try:
            h, w = frame.shape[:2]
            mask = self.get_player_mask(frame, bbox=bbox, ball_xy=ball_xy)

            # Tier 4: If mask is invalid or empty, fall back to NO isolation
            if not self._is_valid_mask(mask):
                self.last_isolation_method = "no_isolation"
                return frame

            f_rad = max(3, int(feather_radius) | 1)
            soft_mask = cv2.GaussianBlur(mask, (f_rad, f_rad), 0).astype(np.float32) * (1.0 / 255.0)
            alpha = soft_mask[:, :, None]

            k_w = max(3, (int(blur_kernel[0]) // 2) | 1)
            k_h = max(3, (int(blur_kernel[1]) // 2) | 1)
            small_bg = cv2.resize(frame, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
            small_bg = cv2.GaussianBlur(small_bg, (k_w, k_h), 0)
            if dim_ratio < 1.0:
                small_bg = cv2.convertScaleAbs(small_bg, alpha=float(dim_ratio), beta=0)
            blurred_bg = cv2.resize(small_bg, (w, h), interpolation=cv2.INTER_LINEAR)

            composite = (
                frame.astype(np.float32) * alpha + blurred_bg.astype(np.float32) * (1.0 - alpha)
            ).astype(np.uint8)

            self.frames_isolated += 1
            del mask, soft_mask, alpha, small_bg, blurred_bg
            return composite
        except Exception:
            self.mask_error_count += 1
            self.last_isolation_method = "no_isolation"
            self.method_counts["no_isolation"] = self.method_counts.get("no_isolation", 0) + 1
            return frame

    def apply_portrait_bokeh(
        self,
        frame: np.ndarray,
        player_bbox=None,
        is_hero: bool = True,
        ball_center: tuple = None,
        isolation_config: dict = None,
    ) -> np.ndarray:
        cfg = isolation_config or {}
        if not cfg.get("enabled", is_hero):
            self.last_isolation_method = "no_isolation"
            return frame
        b_str = int(cfg.get("blur_strength", 21))
        k_val = max(3, (b_str * 2 + 1) if b_str < 15 else (b_str | 1))
        dim_f = float(cfg.get("dim_factor", 0.88 if is_hero else 0.93))
        feather_px = int(cfg.get("feather_px", 17 if is_hero else 13))
        return self.apply_cinematic_blur(
            frame,
            bbox=player_bbox,
            blur_kernel=(k_val, k_val),
            dim_ratio=dim_f,
            feather_radius=feather_px,
            ball_xy=ball_center,
        )

    def get_isolation_telemetry(self) -> dict:
        return {
            "method_used": self.last_isolation_method,
            "fallback_chain": list(self.FALLBACK_CHAIN),
            "frames_isolated": self.frames_isolated,
            "mask_errors": self.mask_error_count,
            "method_counts": dict(self.method_counts),
        }


MobileSAMTracker = PlayerTracker

