"""
Sign2Kannada Real-Time Gesture Inference Pipeline.

Features:
- Robust error handling for files, webcams, and models.
- Temporal smoothing using Exponential Moving Average (EMA) of probability vectors.
- Symmetrical, non-preferential multi-hand tracking.
- Proper resource allocation and cleanup.
- PEP8 compliant styling.
"""

from collections import deque
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Any

import cv2
import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config
from extractor import MediaPipeExtractor
from translator import DIGIT_TO_KANNADA

# Video Recognition Pipeline Imports
from video_model.preprocessing.video_extractor import MediaPipeVideoExtractor
from video_model.inference.inference_pipeline import VideoInferencePipeline

WORD_TO_KANNADA: Dict[str, str] = {
    "book": "ಪುಸ್ತಕ",
    "drink": "ಕುಡಿ",
    "computer": "ಗಣಕಯಂತ್ರ",
    "before": "ಮುಂಚೆ",
    "chair": "ಖುರ್ಚಿ",
    "go": "ಹೋಗು",
    "clothes": "ಬಟ್ಟೆಗಳು",
    "who": "ಯಾರು",
    "candy": "ಮಿಠಾಯಿ",
    "cousin": "ಸೋದರಸಂಬಂಧಿ",
    "deaf": "ಕಿವುಡು",
    "fine": "ಚೆನ್ನಾಗಿದೆ",
    "help": "ಸಹಾಯ",
    "no": "ಇಲ್ಲ",
    "thin": "ತೆಳು",
    "walk": "ನಡೆ",
    "yes": "ಹೌದು"
}


def draw_kannada_text(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font: Optional[ImageFont.FreeTypeFont],
    color: Tuple[int, int, int] = config.COL_BLACK
) -> np.ndarray:
    """Render Kannada Unicode text using PIL to support Kannada font outlines."""
    if not text or font is None:
        return frame

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    x, y = position

    # Render white outline for readability
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        draw.text((x + dx, y + dy), text, font=font, fill=(255, 255, 255))
    
    # PIL uses RGB, BGR input needs conversion
    draw.text(position, text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_confidence_bar(
    frame: np.ndarray,
    confidence: float,
    label: str = "Confidence",
    y_start: int = 150
) -> None:
    """Draw a visual confidence bar indicator."""
    x, y, w, h = 20, y_start, 320, 24
    filled = int(max(0.0, min(1.0, confidence)) * w)
    
    # Track background
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), config.COL_WHITE, 1)

    # Dynamic color scheme
    if confidence >= 0.65:
        fill_col = config.COL_GREEN
    elif confidence >= 0.45:
        fill_col = config.COL_AMBER
    else:
        fill_col = config.COL_RED

    cv2.rectangle(frame, (x, y), (x + filled, y + h), fill_col, -1)
    cv2.putText(
        frame,
        f"{label}: {confidence * 100:.1f}%",
        (x, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        config.COL_WHITE,
        1,
        cv2.LINE_AA,
    )


def draw_top3(
    frame: np.ndarray,
    classes: np.ndarray,
    probs: np.ndarray,
    y_start: int = 200
) -> None:
    """Draw top 3 probability scores panel."""
    top3_idx = np.argsort(probs)[::-1][:3]
    cv2.putText(
        frame,
        "Top predictions:",
        (20, y_start - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
        cv2.LINE_AA
    )
    for rank, idx in enumerate(top3_idx):
        label = str(classes[idx])
        prob = probs[idx]
        bar_w = int(prob * 160)
        row_y = y_start + rank * 22
        cv2.rectangle(frame, (20, row_y), (20 + bar_w, row_y + 16), (60, 100, 60), -1)
        cv2.putText(
            frame,
            f"  {label}: {prob*100:.1f}%",
            (20, row_y + 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            config.COL_WHITE,
            1,
            cv2.LINE_AA
        )


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """Display FPS metric on top right."""
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (frame.shape[1] - 110, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        config.COL_CYAN,
        2,
        cv2.LINE_AA
    )


def resolve_font() -> Optional[ImageFont.FreeTypeFont]:
    """Gracefully load Kannada TTF Font. Fall back to None instead of crashing."""
    font_path: Optional[Path] = None
    for candidate in config.FONT_CANDIDATES:
        if candidate.exists():
            font_path = candidate
            break

    if font_path is None:
        # Scan for NotoSansKannada patterns dynamically
        candidates = sorted(config.PROJECT_ROOT.glob("**/NotoSansKannada*.ttf"))
        if candidates:
            font_path = candidates[0]

    if font_path is not None:
        try:
            return ImageFont.truetype(str(font_path), size=52)
        except Exception as exc:
            print(f"[WARNING] Failed to load font {font_path}: {exc}")
    
    print("[WARNING] Kannada font not found. Kannada text outline rendering will be disabled.")
    return None


def main() -> None:
    print("Loading classification model (model.pkl)...")
    if not config.MODEL_PATH.exists():
        print(f"[CRITICAL ERROR] Model file not found: {config.MODEL_PATH}")
        print("Please run train.py to train the model first.")
        return

    try:
        model = joblib.load(config.MODEL_PATH)
    except Exception as exc:
        print(f"[CRITICAL ERROR] Failed to load model file: {exc}")
        return

    if not hasattr(model, "classes_") or len(model.classes_) < 2:
        print("[CRITICAL ERROR] Loaded model contains insufficient classes. Retrain model.")
        return

    print("Resolving font path...")
    font_large = resolve_font()

    print("Initializing MediaPipe Hand Extractor...")
    try:
        extractor = MediaPipeExtractor(mode=False)
    except Exception as exc:
        print(f"[CRITICAL ERROR] MediaPipe Extractor initialization failed: {exc}")
        return

    print("Initializing Video Pipeline & Video Extractor...")
    try:
        video_extractor = MediaPipeVideoExtractor(config.PROJECT_ROOT)
        video_pipeline = VideoInferencePipeline(config.PROJECT_ROOT)
        video_pipeline.load_model()
    except Exception as exc:
        print(f"[WARNING] Video Pipeline initialization failed: {exc}")
        video_extractor = None
        video_pipeline = None

    print("Opening webcam index 0...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[CRITICAL ERROR] Could not open webcam device. Check connection and permissions.")
        extractor.close()
        return

    # Attempt to set frame size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Webcam opened. Resolution: {actual_width}x{actual_height}")
    print("Starting real-time recognition loop...")
    print("Press 'Q' inside the window to exit.")

    window_name = "Sign2Kannada"
    
    # State Trackers
    # Hand Key -> Probability Vector (EMA smoothed)
    hand_prob_emas: Dict[str, np.ndarray] = {}
    hand_missing_frames: Dict[str, int] = {}
    
    no_hand_frames = 0
    number_buffer = ""
    last_commit_time = 0.0
    last_committed_digit = None
    last_probs = np.zeros(len(model.classes_), dtype=np.float32)

    # FPS Calculations
    fps_counter = deque(maxlen=30)
    prev_time = time.perf_counter()

    # Mode Tracker: 'image' (for digits) or 'video' (for words)
    mode = "image"
    show_pose = True
    show_hands = True
    show_face = True

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARNING] Failed to capture frame from webcam. Exiting loop...")
                break

            now = time.perf_counter()
            fps_counter.append(1.0 / max(now - prev_time, 1e-6))
            prev_time = now
            fps = float(np.mean(fps_counter))

            # Landmarking & Symmetrical Drawing
            hand_states = {}
            active_digit = "-"
            active_confidence = 0.0
            active_is_stable = False

            if mode == "image":
                hands = extractor.extract_all(frame)
                extractor.draw_all_hands(frame)

                hands_present = set()
                active_probs = None

                if hands:
                    no_hand_frames = 0
                    for idx, hand in enumerate(hands):
                        label = hand.get("label", "Unknown")
                        hand_key = label.title() if label.lower() in {"left", "right"} else f"Hand{idx + 1}"
                        hands_present.add(hand_key)
                        hand_missing_frames[hand_key] = 0

                        # Run ML prediction
                        landmarks = hand["landmarks"]
                        raw_probs = model.predict_proba(landmarks.reshape(1, -1))[0]

                        # Temporal smoothing via EMA
                        if hand_key in hand_prob_emas:
                            hand_prob_emas[hand_key] = (
                                config.EMA_ALPHA * raw_probs + 
                                (1 - config.EMA_ALPHA) * hand_prob_emas[hand_key]
                            )
                        else:
                            hand_prob_emas[hand_key] = raw_probs.copy()

                        smoothed_probs = hand_prob_emas[hand_key]
                        best_idx = int(np.argmax(smoothed_probs))
                        best_conf = float(smoothed_probs[best_idx])
                        
                        predicted_class = str(model.classes_[best_idx])
                        current_threshold = 0.15 if predicted_class == "5" else config.CONF_THRESHOLD
                        
                        digit = "-"
                        confidence = 0.0
                        is_stable = False

                        if best_conf >= current_threshold:
                            digit = predicted_class
                            confidence = best_conf
                            is_stable = True

                        hand_states[hand_key] = {
                            "digit": digit,
                            "confidence": confidence,
                            "is_stable": is_stable,
                            "probs": smoothed_probs,
                        }

                        if confidence > active_confidence and digit != "-":
                            active_digit = digit
                            active_confidence = confidence
                            active_probs = smoothed_probs
                            active_is_stable = is_stable
                else:
                    no_hand_frames += 1

                # Cleanup missing hand histories
                for hand_key in list(hand_prob_emas.keys()):
                    if hand_key not in hands_present:
                        hand_missing_frames[hand_key] = hand_missing_frames.get(hand_key, 0) + 1
                        if hand_missing_frames[hand_key] > config.HAND_TIMEOUT_FRAMES:
                            hand_prob_emas.pop(hand_key, None)
                            hand_missing_frames.pop(hand_key, None)

                # Debounced stable buffering
                if active_is_stable and active_digit != "-":
                    if last_committed_digit != active_digit and (now - last_commit_time) >= config.COMMIT_COOLDOWN_SEC:
                        if number_buffer and (now - last_commit_time) <= config.SEQ_MAX_GAP_SEC:
                            number_buffer += active_digit
                        else:
                            number_buffer = active_digit
                        last_commit_time = now
                        last_committed_digit = active_digit

                if no_hand_frames > config.NO_HAND_TIMEOUT:
                    last_committed_digit = None
                    last_probs = np.zeros(len(model.classes_), dtype=np.float32)
                    if no_hand_frames > config.NUMBER_RESET_NO_HAND_FRAMES:
                        number_buffer = ""

                if active_probs is not None:
                    last_probs = active_probs

                # Overlay UI HUD
                kannada_word = DIGIT_TO_KANNADA.get(active_digit, "")
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (360, frame.shape[0]), config.COL_DARK_BG, -1)
                cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

                cv2.putText(frame, "Mode: IMAGE DIGITS", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, config.COL_CYAN, 1, cv2.LINE_AA)
                cv2.putText(frame, f"Active: {active_digit}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.8, config.COL_WHITE, 4, cv2.LINE_AA)

                if font_large is not None:
                    frame = draw_kannada_text(frame, kannada_word, (20, 70), font_large)
                else:
                    cv2.putText(frame, f"Word: {kannada_word}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COL_AMBER, 2, cv2.LINE_AA)

                cv2.putText(frame, f"Number: {number_buffer or '-'}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, config.COL_WHITE, 2, cv2.LINE_AA)

                # Symmetrical per-hand stats
                y_offset = 180
                for hand_key, state in sorted(hand_states.items()):
                    cv2.putText(frame, f"{hand_key}: {state['digit']} ({state['confidence']*100:.0f}%)", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COL_WHITE, 1, cv2.LINE_AA)
                    y_offset += 22

                draw_confidence_bar(frame, active_confidence, y_start=230)
                if not np.allclose(last_probs, 0.0):
                    draw_top3(frame, model.classes_, last_probs, y_start=275)

            elif mode == "video":
                # Video Recognition Mode
                if video_extractor is not None and video_pipeline is not None:
                    landmarks = video_extractor.extract_frame(frame)
                    pred_res = video_pipeline.process_frame_landmarks(landmarks)
                    
                    # Draw skeletal overlay keys on the live camera feed
                    video_extractor.draw_overlays(frame, show_pose=show_pose, show_hands=show_hands, show_face=show_face)
                    
                    word = pred_res["word"]
                    active_confidence = pred_res["confidence"]
                    stability = pred_res["stability"]
                    seq_len = pred_res["sequence_len"]
                    status = pred_res["status"]
                    
                    kannada_word = WORD_TO_KANNADA.get(word, "")
                    
                    # Draw video HUD
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (360, frame.shape[0]), config.COL_DARK_BG, -1)
                    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
                    
                    cv2.putText(frame, "Mode: VIDEO WORDS", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, config.COL_CYAN, 1, cv2.LINE_AA)
                    cv2.putText(frame, f"Word: {word}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.3, config.COL_WHITE, 3, cv2.LINE_AA)
                    
                    if font_large is not None:
                        frame = draw_kannada_text(frame, kannada_word, (20, 80), font_large)
                    else:
                        cv2.putText(frame, f"Kannada: {kannada_word}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COL_AMBER, 2, cv2.LINE_AA)
                        
                    cv2.putText(frame, f"Model Status: {status}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.45, config.COL_WHITE, 1, cv2.LINE_AA)
                    cv2.putText(frame, f"Sequence: {seq_len} frames", (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.45, config.COL_WHITE, 1, cv2.LINE_AA)
                    cv2.putText(frame, f"Stability: {stability:.0f}%", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.45, config.COL_WHITE, 1, cv2.LINE_AA)
                    
                    # Draw overlay toggle states indicator
                    pose_lbl = "ON" if show_pose else "OFF"
                    hands_lbl = "ON" if show_hands else "OFF"
                    face_lbl = "ON" if show_face else "OFF"
                    cv2.putText(frame, f"Pose[P]: {pose_lbl} | Hands[H]: {hands_lbl} | Face[F]: {face_lbl}", (20, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.4, config.COL_WHITE, 1, cv2.LINE_AA)
                    
                    draw_confidence_bar(frame, active_confidence, y_start=230)
                else:
                    cv2.putText(frame, "Video model not loaded.", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COL_RED, 2, cv2.LINE_AA)

            if mode == "image" and no_hand_frames > 0:
                cv2.putText(
                    frame,
                    "No hand detected",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    config.COL_RED,
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                "Keys: Q=quit | M=toggle mode | P=toggle pose | H=toggle hands | F=toggle face",
                (20, frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (140, 140, 140),
                1,
                cv2.LINE_AA
            )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                mode = "video" if mode == "image" else "image"
                if video_pipeline is not None:
                    video_pipeline.reset()
                number_buffer = ""
                no_hand_frames = 0
            elif key == ord("p"):
                show_pose = not show_pose
            elif key == ord("h"):
                show_hands = not show_hands
            elif key == ord("f"):
                show_face = not show_face
    finally:
        # Proper resource cleanup
        cap.release()
        extractor.close()
        if video_extractor is not None:
            video_extractor.close()
        cv2.destroyAllWindows()
        print("System shutdown and resources cleared successfully.")


if __name__ == "__main__":
    main()
