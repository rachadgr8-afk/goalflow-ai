# GoalFlow AI — Professional Football Cinematic Engine

GoalFlow AI is a low-memory, production-grade Football Cinematic AI engine that transforms raw match clips into **1080x1920 (30fps, H.264, yuv420p, AAC)** vertical reels (adaptive up to 64s).

## 14-Stage Pipeline Architecture

`Video → Analysis → Football Events → Player/Ball Detection (YOLO) → Multi-ID Tracking → Hero Detection → Cinematic Director → Smart Reframing (Anchor Track) → Speed Ramps → Subject Isolation (MobileSAM) → Color Grade → Sound Design → FFmpeg → Technical & Style QC → Final MP4`

1. **Player & Ball Detection + Multi-ID Tracking (`video_engine.py`)**
   - Primary **YOLOv8n** detector (`person` & `sports ball`) cached once in memory.
   - Multi-object Kalman-style tracker assigning `id`, `confidence`, `bbox`, `centerX`, `centerY`, `scale`, `frameStart`, `frameEnd`, velocity, acceleration, and ball-proximity linking.
   - Strict fallback chain: `YOLO → MOG2 motion contour → inertial prediction → center crop`.
2. **Evidence-Based Football Events (`FootballEventEngine`)**
   - Classifies `shot`, `pass`, `dribble`, `skill`, `tackle`, `save`, `acceleration`, `defender_interaction`, `goal`, `celebration`, `reaction`, `high-motion`, or `unknown` when evidence is insufficient.
3. **Hero Moment Scoring (`HeroMomentDetector`)**
   - Composite score from `motion + acceleration + ball_proximity + shot + dribble + skill + defender_interaction + goal/save/celebration`.
4. **Cinematic Director & Multi-Keyframe Smart Reframing (`cinematic_modules.py`)**
   - Generates multi-keyframe `anchor_track` supporting `followX`, `followY`, `scale`, `push-in`, `pull-out`, `drift`, and cubic/cosine easing without crop jumps.
5. **Event-Linked Speed Ramps & Selective Quarter-Res Optical Flow**
   - Nonlinear speed curves (`Hero: 1 → 0.85 → 0.60 → 0.30 → 0.50 → 1`, `Skill`, `Shot`, `Celebration`) with selective Farneback optical flow.
6. **Subject Isolation (`sam_tracker.py`)**
   - Cached **MobileSAM** on Hero/isolated scenes with feathered alpha masks and subtle background blur/dim (`MobileSAM → YOLO → motion → no isolation`).
7. **Reference Style Analyzer & Technical/Style QC (`cinematic_modules.py`)**
   - Extracts 12 style metrics as Director constraints (zero copied frames/timestamps) and validates 11 technical checks + 11 style axes post-render.
