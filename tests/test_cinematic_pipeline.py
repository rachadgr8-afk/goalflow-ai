import json
import os
import tempfile
import unittest
import cv2
import numpy as np

from video_engine import (
    CinematicColorGrader,
    CinematicDirector,
    CinematicEngine,
    FootballDetectorTracker,
    FootballEventEngine,
    HeroMomentDetector,
    QualityControlEngine,
    ReferenceStyleAnalyzer,
    SmartReframer,
)
from sam_tracker import MobileSAMTracker
from cinematic_storyteller import (
    CinematicStoryteller,
    DepthAnythingV2Engine,
    PsychologicalColorGrader,
    get_process_rss_mb,
)


class TestFootballCinematicPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp(prefix="goalflow_test_")
        cls.input_video = os.path.join(cls.tmp_dir, "test_input.mp4")
        cls.output_video = os.path.join(cls.tmp_dir, "test_output_9_16.mp4")
        cls.output_psych_video = os.path.join(cls.tmp_dir, "test_output_psych_9_16.mp4")
        CinematicEngine.generate_synthetic_football_video(cls.input_video, duration=6.0, fps=30)

    def test_1_yolo_tracker_and_events(self):
        tracker = FootballDetectorTracker(1280, 720, 30.0, use_yolo=True)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:, :] = (35, 115, 45)
        cv2.rectangle(frame, (580, 280), (650, 460), (220, 50, 50), -1)
        cv2.rectangle(frame, (665, 285), (725, 455), (50, 80, 220), -1)
        cv2.circle(frame, (660, 440), 12, (250, 250, 250), -1)

        res1 = tracker.detect_and_track_frame(frame, 0.0)
        res2 = tracker.detect_and_track_frame(frame, 0.1)
        self.assertIn("player", res2)
        self.assertIn("ball_proximity", res2)
        self.assertIn("player_ball_interaction", res2)
        self.assertIn("defender_interaction", res2)
        self.assertIn("direction_deg", res2["player"])
        self.assertIn("velocity", res2["player"])
        self.assertIn("acceleration", res2["player"])
        self.assertIsNotNone(res2["player"]["bbox"])

        empty_events = FootballEventEngine.detect_events([], 4.0)
        self.assertEqual(empty_events[0]["event"], "unknown")
        self.assertFalse(empty_events[0]["event_backed"])

    def test_2_hero_moment_and_director_script(self):
        motion_only_history = [
            {
                "timestamp": i * 0.2,
                "motion_energy": 0.88,
                "ball_proximity": 0.12,
                "player_ball_interaction": {"interacting": False, "score": 0.10},
                "defender_interaction": {"interacting": False, "score": 0.0, "defenders_nearby": 0},
                "player": {"id": 1, "velocity": 40.0, "acceleration": 20.0, "centerX": 500, "centerY": 360},
                "ball": None,
                "players": [],
            }
            for i in range(20)
        ]
        unknown_events = [{"event": "unknown", "start": 0.0, "end": 4.0, "confidence": 0.4, "event_backed": False}]
        no_hero = HeroMomentDetector.identify_hero_moment(
            motion_only_history, 4.0, (1.0, 2.5), unknown_events
        )
        self.assertFalse(no_hero["is_hero"])
        self.assertFalse(no_hero["event_backed"])

        real_history = [
            {
                "timestamp": i * 0.2,
                "motion_energy": 0.62 if 6 <= i <= 12 else 0.18,
                "ball_proximity": 0.88 if 6 <= i <= 12 else 0.25,
                "player_ball_interaction": {
                    "interacting": 6 <= i <= 12,
                    "score": 0.90 if 6 <= i <= 12 else 0.2,
                    "type": "strike_release" if i == 9 else "dribble_control",
                },
                "defender_interaction": {
                    "interacting": 6 <= i <= 10,
                    "score": 0.78 if 6 <= i <= 10 else 0.1,
                    "defenders_nearby": 1 if 6 <= i <= 10 else 0,
                },
                "player": {
                    "id": 1,
                    "velocity": 320.0 if 6 <= i <= 12 else 45.0,
                    "acceleration": 740.0 if 6 <= i <= 12 else 20.0,
                    "centerX": 640,
                    "centerY": 360,
                },
                "ball": {"velocity": 680.0 if i == 9 else 120.0, "centerX": 660, "centerY": 390},
                "players": [{"id": 1}, {"id": 2}],
            }
            for i in range(25)
        ]
        real_events = [
            {"event": "build_up", "start": 0.0, "end": 1.2, "confidence": 0.78, "event_backed": True},
            {"event": "dribble", "start": 1.2, "end": 1.8, "confidence": 0.86, "event_backed": True},
            {"event": "shot", "start": 1.8, "end": 2.6, "confidence": 0.93, "event_backed": True},
            {"event": "goal", "start": 2.6, "end": 3.4, "confidence": 0.95, "event_backed": True},
            {"event": "celebration", "start": 3.4, "end": 5.0, "confidence": 0.88, "event_backed": True},
        ]
        hero = HeroMomentDetector.identify_hero_moment(real_history, 5.0, (1.5, 3.0), real_events)
        self.assertTrue(hero["is_hero"])
        self.assertTrue(hero["event_backed"])

        script = CinematicDirector.build_director_script(
            real_events, hero, 5.0, "9:16", "CINEMATIC", "hero"
        )
        self.assertGreaterEqual(len(script), 3)

    def test_3_smart_reframer_and_speed_ramps(self):
        track_history = [
            {
                "timestamp": i * 0.1,
                "ball_proximity": 0.82,
                "player_ball_interaction": {"interacting": True, "score": 0.85},
                "defender_interaction": {"interacting": True, "score": 0.72, "defenders_nearby": 1},
                "player": {
                    "id": 1,
                    "centerX": 380 + i * 14,
                    "centerY": 360 + int(12 * np.sin(i * 0.3)),
                    "scale": 0.14,
                    "bbox": (350 + i * 14, 280, 60, 160),
                    "velocity": 220.0,
                    "vx": 180.0,
                    "vy": 20.0,
                    "direction_rad": 0.11,
                    "direction_deg": 6.3,
                },
                "ball": {"centerX": 410 + i * 15, "centerY": 420, "velocity": 340.0, "vx": 260.0, "vy": -15.0},
                "defenders": [{"id": 2, "x": 460 + i * 12, "y": 355, "dist": 75.0}],
                "players": [
                    {"id": 1, "centerX": 380 + i * 14, "centerY": 360, "bbox": (350 + i * 14, 280, 60, 160)},
                    {"id": 2, "centerX": 460 + i * 12, "centerY": 355, "bbox": (430 + i * 12, 280, 60, 155)},
                ],
            }
            for i in range(40)
        ]
        director_script = [
            {
                "scene_type": "build_up",
                "shot_type": "medium_wide",
                "start": 0.0,
                "end": 1.3,
                "zoom": 1.12,
                "pov_switch": True,
                "duel_moment": 2.6,
                "camera_trajectory": {"framing_mode": "follow", "lead_factor": 0.14},
            },
            {
                "scene_type": "shot",
                "shot_type": "medium_Action",
                "start": 1.3,
                "end": 2.6,
                "zoom": 1.24,
                "pov_switch": True,
                "duel_moment": 2.6,
                "camera_trajectory": {"framing_mode": "anticipation", "lead_factor": 0.22},
            },
            {
                "scene_type": "hero",
                "shot_type": "close_hero",
                "start": 2.6,
                "end": 4.0,
                "zoom": 1.32,
                "pov_switch": True,
                "duel_moment": 2.6,
                "camera_trajectory": {"framing_mode": "impact", "lead_factor": 0.16},
            },
        ]
        anchor = SmartReframer.build_anchor_track(
            track_history, director_script, 1280, 720, 1080, 1920, 30.0, 4.0
        )
        self.assertTrue(anchor["smooth_passed"])
        self.assertLess(anchor["max_jump_ratio"], 0.035)
        self.assertEqual(anchor["lost_player_frames"], 0)
        self.assertEqual(anchor["player_retention_rate"], 1.0)

        s_unbacked = CinematicDirector.evaluate_speed_curve(0.5, "hero", 0.30, event_backed=False)
        self.assertEqual(s_unbacked, 1.0)

    def test_4_subject_isolation_color_and_reference_style(self):
        frame = np.zeros((480, 270, 3), dtype=np.uint8)
        frame[:, :] = (30, 230, 40)
        graded = CinematicColorGrader.apply_grade(
            frame, intensity=0.85, grade_config={"profile": "hero_cinema_prime", "intensity": 0.92}
        )
        self.assertEqual(graded.shape, frame.shape)

        psych_graded = PsychologicalColorGrader.apply_teal_orange_grade(frame, intensity=0.88)
        self.assertEqual(psych_graded.shape, frame.shape)

        sam = MobileSAMTracker()
        isolated = sam.apply_portrait_bokeh(
            graded,
            player_bbox=(80, 120, 110, 240),
            is_hero=True,
            ball_center=(140, 340),
            isolation_config={"enabled": True, "blur_strength": 9, "dim_factor": 0.86, "feather_px": 15},
        )
        self.assertEqual(isolated.shape, graded.shape)

        ref_profile = ReferenceStyleAnalyzer.analyze_reference_video(
            self.input_video, "ucl_broadcast_reel"
        )
        self.assertIn("shot_duration", ref_profile)
        self.assertTrue(ref_profile["ip_safety"]["copied_timestamps"] is False)

    def test_5_psychological_storyteller_vlm_deepseek_and_fallbacks(self):
        # 5a. Empty story when no duel evidence exists (LLM cannot invent winner/duel)
        no_evidence_history = [
            {
                "timestamp": i * 0.2,
                "ball_proximity": 999.0,
                "player_ball_interaction": {"score": 0.0},
                "defender_interaction": {"pressure_score": 0.0},
                "player": {"id": 1, "speed": 10.0},
                "defenders": [],
            }
            for i in range(15)
        ]
        unknown_events = [{"type": "unknown", "start": 0.0, "end": 3.0, "confidence": 0.4, "event_backed": False}]
        empty_pkg = CinematicStoryteller.build_psychological_package(
            track_history=no_evidence_history,
            detected_events=unknown_events,
            hero_moment={"is_hero": False, "event_backed": False},
            clip_duration=6.0,
            # Even if a rogue VLM tries to claim Raphinha won, lack of evidence must block it!
            vlm_caller=lambda *_: '{"duel_moment": 2.8, "winner": "Raphinha", "loser": "Sangante", "story_arc": ["predator", "trap", "dominance"], "confidence": 0.99}',
        )
        self.assertFalse(empty_pkg["psychological_story"])
        self.assertEqual(empty_pkg["story_script"], [])
        self.assertIsNone(empty_pkg["winner"])

        # 5b. Valid & Invalid Qwen2-VL/InternVL2 JSON with real duel evidence
        duel_history = [
            {
                "timestamp": i * 0.2,
                "ball_proximity": 0.88 if 10 <= i <= 18 else 0.35,
                "player_ball_interaction": {"score": 0.91 if 10 <= i <= 18 else 0.30},
                "defender_interaction": {"pressure_score": 0.84 if 11 <= i <= 16 else 0.20},
                "player": {"id": 11, "speed": 235.0 if 10 <= i <= 18 else 65.0},
                "defenders": [{"id": 5, "x": 660, "y": 350, "dist": 48.0}],
            }
            for i in range(30)
        ]
        duel_events = [
            {"type": "dribble", "start": 1.8, "end": 3.1, "confidence": 0.89, "event_backed": True},
            {"type": "shot", "start": 3.1, "end": 4.1, "confidence": 0.94, "event_backed": True},
        ]
        duel_hero = {"is_hero": True, "event_backed": True, "confidence": 0.91, "hero_score": 0.87}

        # Valid JSON with verified identities
        valid_duel = CinematicStoryteller.analyze_duel(
            track_history=duel_history,
            detected_events=duel_events,
            hero_moment=duel_hero,
            clip_duration=6.0,
            player_identities={"winner": "Raphinha", "loser": "Sangante"},
            vlm_caller=lambda *_: '```json\n{"duel_moment": 2.8, "winner": "Raphinha", "loser": "Sangante", "story_arc": ["predator", "trap", "dominance"], "confidence": 0.92}\n```',
        )
        self.assertTrue(valid_duel["duel_detected"])
        self.assertEqual(valid_duel["duel_moment"], 2.8)
        self.assertEqual(valid_duel["winner"], "Raphinha")
        self.assertEqual(valid_duel["loser"], "Sangante")
        self.assertEqual(valid_duel["story_arc"], ["predator", "trap", "dominance"])

        # Reject invented player names when identities are NOT provided (fallback to Player #ID)
        unverified_name_duel = CinematicStoryteller.analyze_duel(
            track_history=duel_history,
            detected_events=duel_events,
            hero_moment=duel_hero,
            clip_duration=6.0,
            player_identities=None,
            vlm_caller=lambda *_: '{"duel_moment": 99.5, "winner": "InventedStar", "loser": "FakeDefender", "story_arc": ["predator", "trap"], "confidence": 0.88}',
        )
        self.assertEqual(unverified_name_duel["duel_moment"], 6.0)  # clamped to clip_duration
        self.assertEqual(unverified_name_duel["winner"], "Player #11")
        self.assertEqual(unverified_name_duel["loser"], "Player #5")

        # 5c. Fallback when VLM API raises timeout/network exception
        def failing_vlm(*_args, **_kwargs):
            raise TimeoutError("Qwen2-VL API timed out")

        fallback_duel = CinematicStoryteller.analyze_duel(
            track_history=duel_history,
            detected_events=duel_events,
            hero_moment=duel_hero,
            clip_duration=6.0,
            vlm_caller=failing_vlm,
        )
        self.assertTrue(fallback_duel["duel_detected"])
        self.assertEqual(fallback_duel["winner"], "Player #11")
        self.assertGreater(fallback_duel["confidence"], 0.5)

        # 5d. Fallback when DeepSeek fails -> returns story_script=[] and does not raise
        def failing_deepseek(*_args, **_kwargs):
            raise RuntimeError("DeepSeek R1 503 Unavailable")

        failed_script = CinematicStoryteller.generate_story_script(
            fallback_duel, clip_duration=6.0, deepseek_caller=failing_deepseek
        )
        self.assertEqual(failed_script, [])

        # 5e. Timestamp clamping & collision avoidance in story_script
        out_of_bounds_lines = [
            {"time": -3.5, "text": "COME CLOSER...", "position": "center"},
            {"time": 1.2, "text": "DEAD END", "position": "lower"},
            {"time": 1.21, "text": "OVERLAPPING LINE", "position": "lower"},
            {"time": 2.8, "text": "LOCKED IN", "position": "invalid_pos"},
            {"time": 14.0, "text": "PAST END OF CLIP", "position": "lower"},
            {"time": 25.0, "text": "FINAL STRIKE", "position": "center"},
        ]
        clamped_script = CinematicStoryteller.sanitize_and_clamp_story_script(
            out_of_bounds_lines, clip_duration=6.0, existing_text_windows=[(1.0, 2.0)]
        )
        self.assertEqual(len(clamped_script), 6)
        for idx, line in enumerate(clamped_script):
            self.assertGreaterEqual(line["time"], 0.0)
            self.assertLessEqual(line["time"], 6.0)
            self.assertIn(line["position"], ("center", "lower", "upper"))
            if idx > 0:
                self.assertGreaterEqual(line["time"], clamped_script[idx - 1]["time"])

    def test_6_depth_anything_v2_bokeh_low_angle_and_edit_plan(self):
        depth_eng = DepthAnythingV2Engine(lazy_load=True)
        self.assertFalse(depth_eng.is_loaded)

        frame = np.zeros((640, 360, 3), dtype=np.uint8)
        frame[:, :] = (40, 120, 48)
        cv2.rectangle(frame, (130, 200), (230, 480), (45, 70, 230), -1)

        d_map = depth_eng.estimate_depth_map(frame, player_bbox=(130, 200, 230, 480), frame_idx=0)
        self.assertTrue(depth_eng.is_loaded)
        self.assertEqual(d_map.shape, (640, 360))

        # Test no full-frame blur even if degenerate all-zero depth map is passed
        zero_depth = np.zeros((640, 360), dtype=np.float32)
        bokeh_frame, bokeh_meta = depth_eng.apply_depth_aware_bokeh(
            frame, depth_map=zero_depth, player_bbox=None, blur_strength=17
        )
        self.assertEqual(bokeh_frame.shape, frame.shape)
        self.assertFalse(bokeh_meta["full_frame_blur"])
        self.assertGreaterEqual(bokeh_meta["sharp_foreground_ratio"], 0.06)
        self.assertGreaterEqual(depth_eng.full_frame_blur_prevented, 1)

        # Test restrained low-angle POV preserves player geometry (0.0 aspect distortion)
        la_frame, la_meta = depth_eng.apply_low_angle_pov(
            bokeh_frame, player_bbox=(130, 200, 230, 480), depth_map=d_map, intensity=0.30
        )
        self.assertEqual(la_frame.shape, frame.shape)
        self.assertTrue(la_meta["geometry_preserved"])
        self.assertEqual(la_meta["aspect_distortion"], 0.0)

        depth_eng.release()
        self.assertFalse(depth_eng.is_loaded)

        # Test CinematicEngine helper methods (add_depth_bokeh, add_low_angle_pov, add_cinematic_lut, slow_motion_rife)
        b_frame = CinematicEngine.add_depth_bokeh(frame, depth_enabled=True, player_bbox=(130, 200, 230, 480))
        self.assertEqual(b_frame.shape, frame.shape)
        pov_frame = CinematicEngine.add_low_angle_pov(b_frame, player_bbox=(130, 200, 230, 480), duel_verified=True)
        self.assertEqual(pov_frame.shape, frame.shape)
        lut_frame = CinematicEngine.add_cinematic_lut(pov_frame, intensity=0.85)
        self.assertEqual(lut_frame.shape, frame.shape)

        # Test 3-frame extraction & generate_script
        rep_frames = CinematicStoryteller.extract_representative_frames_b64(self.input_video, duel_moment=2.8)
        self.assertEqual(len(rep_frames), 3)
        script_6 = CinematicStoryteller.generate_script(
            {"duel_moment": 2.8, "confidence": 0.88, "decisive_events": ["dribble", "shot"]}
        )
        self.assertEqual(len(script_6), 6)
        rife_out = os.path.join(self.tmp_dir, "test_rife_fallback.mp4")
        rife_res = CinematicEngine(use_player_tracker=False).slow_motion_rife(
            self.input_video, rife_out, slow_factor=0.5, target_segment=(2.0, 2.5)
        )
        self.assertEqual(rife_res["status"], "completed")
        self.assertGreater(rife_res["interpolated_frames"], 0)

    def test_7_full_pipeline_render_and_psychological_story_mode(self):
        engine = CinematicEngine(use_player_tracker=True)
        # 7a. Normal CINEMATIC render remains unchanged (psychological_story=False, story_script=[])
        normal_res = engine.execute_pipeline(
            input_path=self.input_video,
            output_path=self.output_video,
            slow_factor=0.30,
            target_segment=(1.8, 3.8),
            options={
                "mode": "CINEMATIC",
                "aspect_ratio": "9:16",
                "speed_ramp_type": "hero",
                "reference_style_preset": "ucl_broadcast_reel",
            },
        )
        self.assertEqual(normal_res["status"], "completed")
        self.assertFalse(normal_res["psychological_story"])
        self.assertEqual(normal_res["story_script"], [])
        self.assertFalse(normal_res["edit_plan"]["psychological_story"])
        self.assertTrue(normal_res["qc_report"]["passed"])

        # 7b. PSYCHOLOGICAL_STORY render on 6-second duel clip
        psych_res = engine.execute_pipeline(
            input_path=self.input_video,
            output_path=self.output_psych_video,
            slow_factor=0.30,
            target_segment=(1.8, 3.8),
            options={
                "mode": "PSYCHOLOGICAL_STORY",
                "aspect_ratio": "9:16",
                "speed_ramp_type": "hero",
                "player_identities": {"winner": "Raphinha", "loser": "Sangante"},
            },
        )
        self.assertEqual(psych_res["status"], "completed")
        self.assertTrue(os.path.exists(self.output_psych_video))
        self.assertTrue(psych_res["psychological_story"])
        self.assertEqual(len(psych_res["story_script"]), 6)
        self.assertTrue(psych_res["edit_plan"]["psychological_story"])
        self.assertEqual(psych_res["edit_plan"]["story_role"], "predator")
        self.assertIn("predator", psych_res["edit_plan"]["story_arc"])
        self.assertTrue(psych_res["edit_plan"]["depth_effect"])
        self.assertTrue(psych_res["edit_plan"]["low_angle"])
        self.assertTrue(psych_res["edit_plan"]["pov_switch"])
        self.assertTrue(psych_res["qc_report"]["passed"])
        self.assertTrue(psych_res["qc_report"]["psychological_qc"]["no_full_frame_blur"])
        self.assertTrue(psych_res["qc_report"]["psychological_qc"]["timestamps_inside_segment"])
        self.assertGreater(psych_res["qc_report"]["duel_confidence"], 0.5)
        self.assertGreater(psych_res["qc_report"]["story_confidence"], 0.5)
        self.assertTrue(psych_res["qc_report"]["depth_effect_used"])
        self.assertTrue(psych_res["qc_report"]["low_angle_used"])
        self.assertTrue(psych_res["qc_report"]["pov_switch_used"])
        self.assertEqual(psych_res["qc_report"]["story_script_count"], 6)
        self.assertIn("peak_rss_mb", psych_res["memory_telemetry"])

        # 7c. Verify server.ts routes remain intact
        with open("server.ts", "r", encoding="utf-8") as sf:
            server_code = sf.read()
        for route in (
            "/api/render-full-cinematic",
            "/api/job/:job_id",
            "/api/render-progress",
            "/api/render-result",
            "/api/download/:job_id",
            "story_script",
            "video_url",
            "poster_url",
        ):
            self.assertIn(route, server_code)


if __name__ == "__main__":
    unittest.main()
