from pathlib import Path
from collections import deque, Counter
import time

import cv2
import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from extractor import MediaPipeExtractor
from translator import DIGIT_TO_KANNADA


MODEL_PATH = Path("model.pkl")
FONT_CANDIDATES = (
    Path("assets/fonts/NotoSansKannada-Regular.ttf"),
    Path("assets/fonts/NotoSansKannada-VariableFont_wdth,wght.ttf"),
    Path("NotoSansKannada-Regular.ttf"),
)

# ── Inference hyper-parameters ────────────────────────────────────────────────
HISTORY_LEN      = 9     # Frames kept for majority-vote smoothing
MIN_VOTES        = 3     # Minimum frames a class must win to be displayed
CONF_THRESHOLD   = 0.28  # Minimum model confidence to accept a prediction
                         # (kept low because domain gap between dataset & webcam
                         #  lowers raw probabilities; majority voting provides
                         #  the real stability guarantee)
NO_HAND_TIMEOUT  = 15    # Frames without detection before resetting display
HAND_TIMEOUT_FRAMES = 10 # Frames before dropping a hand's history when it disappears
SEQ_MAX_GAP_SEC   = 1.2  # Max gap between digits to keep same number
COMMIT_COOLDOWN_SEC = 0.35  # Debounce to prevent duplicate commits while holding
NUMBER_RESET_NO_HAND_FRAMES = 45  # Clear number after extended no-hand period
# ─────────────────────────────────────────────────────────────────────────────

# Colour palette
COL_WHITE   = (255, 255, 255)
COL_BLACK   = (20,  20,  20)
COL_GREEN   = (0,   200, 0)
COL_AMBER   = (0,   165, 255)   # BGR
COL_RED     = (0,   0,   220)
COL_CYAN    = (220, 220, 0)
COL_DARK_BG = (30,  30,  30)


def draw_kannada_text(frame, text, position, font, color=COL_BLACK):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    x, y = position
    outline = COL_WHITE[::-1]  # PIL uses RGB
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        draw.text((x + dx, y + dy), text, font=font, fill=(255, 255, 255))
    draw.text(position, text, font=font, fill=color[::-1])  # BGR → RGB for PIL
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_confidence_bar(frame, confidence, label="Confidence", y_start=150):
    x, y, w, h = 20, y_start, 320, 24
    filled = int(max(0.0, min(1.0, confidence)) * w)
    # Background track
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), COL_WHITE, 1)
    # Fill colour: green ≥65%, amber 45-65%, red <45%
    if confidence >= 0.65:
        fill_col = COL_GREEN
    elif confidence >= 0.45:
        fill_col = COL_AMBER
    else:
        fill_col = COL_RED
    cv2.rectangle(frame, (x, y), (x + filled, y + h), fill_col, -1)
    cv2.putText(
        frame,
        f"{label}: {confidence * 100:.1f}%",
        (x, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        COL_WHITE,
        1,
        cv2.LINE_AA,
    )


def draw_top3(frame, model, probs, y_start=200):
    """Draw a compact top-3 predictions panel for debugging."""
    top3_idx = np.argsort(probs)[::-1][:3]
    cv2.putText(frame, "Top predictions:", (20, y_start - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    for rank, idx in enumerate(top3_idx):
        label = str(model.classes_[idx])
        prob  = probs[idx]
        bar_w = int(prob * 160)
        row_y = y_start + rank * 22
        cv2.rectangle(frame, (20, row_y), (20 + bar_w, row_y + 16), (60, 100, 60), -1)
        cv2.putText(frame, f"  {label}: {prob*100:.1f}%", (20, row_y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, COL_WHITE, 1, cv2.LINE_AA)


def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 110, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_CYAN, 2, cv2.LINE_AA)


def resolve_font_path() -> Path:
    for candidate in FONT_CANDIDATES:
        if candidate.exists():
            return candidate
    dynamic_candidates = sorted(Path("assets/fonts").glob("NotoSansKannada*.ttf"))
    if dynamic_candidates:
        return dynamic_candidates[0]
    raise FileNotFoundError(
        "Kannada font file not found. "
        "Expected one of: "
        + ", ".join(str(p) for p in FONT_CANDIDATES)
    )


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    if not hasattr(model, "classes_") or len(model.classes_) < 2:
        raise ValueError(
            "model.pkl has fewer than 2 classes. Retrain using multiple gesture labels."
        )

    font_path = resolve_font_path()
    font_large = ImageFont.truetype(str(font_path), size=52)
    extractor  = MediaPipeExtractor(mode=False)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")
    # Request a higher resolution for better hand detection
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "Sign2Kannada"

    last_probs        = np.zeros(len(model.classes_), dtype=np.float32)
    no_hand_frames    = 0
    hand_histories: dict[str, deque[str]] = {}
    hand_missing: dict[str, int] = {}
    number_buffer     = ""
    last_commit_time  = 0.0
    last_committed_digit = None

    # FPS tracking
    fps_counter = deque(maxlen=30)
    prev_time   = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # ── FPS calculation ──────────────────────────────────────────────
            now = time.perf_counter()
            fps_counter.append(1.0 / max(now - prev_time, 1e-6))
            prev_time = now
            fps = np.mean(fps_counter)

            # ── Hand extraction + prediction ─────────────────────────────────
            hands = extractor.extract_all(frame)
            extractor.draw_all_hands(frame)

            hand_states = {}
            hands_present = set()
            active_digit = "-"
            active_confidence = 0.0
            active_probs = None
            active_is_stable = False
            active_history_len = 0

            if hands:
                no_hand_frames = 0
                for idx, hand in enumerate(hands):
                    label = hand.get("label", "Unknown")
                    if label.lower() in {"left", "right"}:
                        hand_key = label.title()
                    else:
                        hand_key = f"Hand{idx + 1}"

                    hands_present.add(hand_key)
                    hand_missing[hand_key] = 0
                    history = hand_histories.setdefault(hand_key, deque(maxlen=HISTORY_LEN))

                    landmarks = hand["landmarks"]
                    probs = model.predict_proba(landmarks.reshape(1, -1))[0]
                    best_idx = int(np.argmax(probs))
                    best_conf = float(probs[best_idx])

                    if best_conf >= CONF_THRESHOLD:
                        history.append(str(model.classes_[best_idx]))

                    if len(history) >= MIN_VOTES:
                        vote_counter = Counter(history)
                        winner, votes = vote_counter.most_common(1)[0]
                        if votes >= MIN_VOTES:
                            digit = winner
                            win_idx = int(np.where(model.classes_ == int(digit))[0][0])
                            confidence = float(probs[win_idx])
                            is_stable = True
                        else:
                            digit = str(model.classes_[best_idx])
                            confidence = best_conf
                            is_stable = False
                    elif len(history) > 0:
                        digit = str(model.classes_[best_idx])
                        confidence = best_conf
                        is_stable = False
                    else:
                        digit = "-"
                        confidence = 0.0
                        is_stable = False

                    hand_states[hand_key] = {
                        "digit": digit,
                        "confidence": confidence,
                        "is_stable": is_stable,
                        "probs": probs,
                        "history_len": len(history),
                    }

                    if confidence > active_confidence and digit != "-":
                        active_digit = digit
                        active_confidence = confidence
                        active_probs = probs
                        active_is_stable = is_stable
                        active_history_len = len(history)
            else:
                no_hand_frames += 1

            for hand_key in list(hand_missing.keys()):
                if hand_key not in hands_present:
                    hand_missing[hand_key] += 1
                    if hand_missing[hand_key] > HAND_TIMEOUT_FRAMES:
                        hand_missing.pop(hand_key, None)
                        hand_histories.pop(hand_key, None)

            if active_is_stable and active_digit != "-":
                now = time.perf_counter()
                if last_committed_digit != active_digit and (now - last_commit_time) >= COMMIT_COOLDOWN_SEC:
                    if number_buffer and (now - last_commit_time) <= SEQ_MAX_GAP_SEC:
                        number_buffer += active_digit
                    else:
                        number_buffer = active_digit
                    last_commit_time = now
                    last_committed_digit = active_digit

            if no_hand_frames > NO_HAND_TIMEOUT:
                last_committed_digit = None
                last_probs = np.zeros(len(model.classes_), dtype=np.float32)
                if no_hand_frames > NUMBER_RESET_NO_HAND_FRAMES:
                    number_buffer = ""

            if active_probs is not None:
                last_probs = active_probs

            # ── Overlay rendering ────────────────────────────────────────────
            kannada_word = DIGIT_TO_KANNADA.get(active_digit, "")

            # Dark semi-transparent HUD strip on the left
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (360, frame.shape[0]), COL_DARK_BG, -1)
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

            # Active digit label
            cv2.putText(
                frame,
                f"Active: {active_digit}",
                (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.8,
                COL_WHITE,
                4,
                cv2.LINE_AA,
            )

            # Kannada word (rendered with PIL for Unicode support)
            frame = draw_kannada_text(frame, kannada_word, (20, 70), font_large)

            # Number buffer (two digits in quick succession)
            cv2.putText(
                frame,
                f"Number: {number_buffer or '-'}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                COL_WHITE,
                2,
                cv2.LINE_AA,
            )

            # Per-hand status lines
            def hand_sort_key(name: str):
                lname = name.lower()
                if lname == "left":
                    return (0, name)
                if lname == "right":
                    return (1, name)
                return (2, name)

            y = 160
            for hand_key in sorted(hand_states.keys(), key=hand_sort_key):
                state = hand_states[hand_key]
                cv2.putText(
                    frame,
                    f"{hand_key}: {state['digit']} ({state['confidence']*100:.0f}%)",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    COL_WHITE,
                    1,
                    cv2.LINE_AA,
                )
                y += 22

            # Confidence bar (active hand)
            draw_confidence_bar(frame, active_confidence, y_start=215)

            # Top-3 prediction breakdown
            if not np.allclose(last_probs, 0.0):
                draw_top3(frame, model, last_probs, y_start=255)

            # FPS counter (top-right)
            draw_fps(frame, fps)

            # Hand not detected warning
            if no_hand_frames > 0:
                cv2.putText(
                    frame,
                    "No hand detected",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    COL_RED,
                    2,
                    cv2.LINE_AA,
                )

            # Stability indicator (shows how filled the vote buffer is)
            stability = active_history_len / HISTORY_LEN
            draw_confidence_bar(frame, stability, label="Stability", y_start=330)

            # Key hint
            cv2.putText(frame, "Press Q to quit", (20, frame.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1, cv2.LINE_AA)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
