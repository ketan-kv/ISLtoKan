from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)

DEFAULT_TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)

# MediaPipe landmark indices
_WRIST = 0
_MCP_INDEX = 5   # Index finger MCP — used for scale reference


class MediaPipeExtractor:
    def __init__(self, mode: bool = True, model_path: str | Path | None = None) -> None:
        self.static_mode = mode
        self.backend = None
        self.last_results = None
        self.last_right_hand_landmarks = None
        self.last_right_hand_proto = None
        self.last_detected_handedness = None
        self.last_hand_landmarks = []
        self.last_hand_protos = []
        self.last_detected_handedness_all = []
        self.mp_hands = None
        self.mp_drawing = None
        self.hands = None
        self.landmarker = None

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self._init_solutions_backend()
        else:
            self._init_tasks_backend(model_path)

    def _init_solutions_backend(self) -> None:
        self.backend = "solutions"
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_mode,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.7,   # Raised from 0.5 for cleaner detections
            min_tracking_confidence=0.5,
        )

    def _init_tasks_backend(self, model_path: str | Path | None) -> None:
        from mediapipe.tasks.python import vision

        self.backend = "tasks"
        resolved_model_path = self._resolve_task_model_path(model_path)
        base_options = mp.tasks.BaseOptions(model_asset_path=str(resolved_model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.7,   # Raised from 0.5
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def _resolve_task_model_path(self, model_path: str | Path | None) -> Path:
        candidates = []
        if model_path is not None:
            candidates.append(Path(model_path))
        candidates.extend(
            [
                Path("assets/models/hand_landmarker.task"),
                Path("hand_landmarker.task"),
            ]
        )

        for candidate in candidates:
            if candidate.exists():
                return candidate

        download_path = Path("assets/models/hand_landmarker.task")
        download_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(DEFAULT_TASK_MODEL_URL, str(download_path))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "MediaPipe tasks backend requires hand_landmarker.task. "
                f"Auto-download failed: {exc}. "
                "Download the model manually and place it at assets/models/hand_landmarker.task."
            ) from exc
        return download_path

    @staticmethod
    def _normalize_landmarks(landmarks_flat: np.ndarray) -> np.ndarray:
        """
        Make landmark features position- and scale-invariant.

        Steps:
          1. Reshape to (21, 3) — x, y, z per landmark.
          2. Subtract wrist (landmark 0) so the wrist is the origin.
          3. Compute scale = Euclidean distance from wrist to index-finger MCP (landmark 5).
          4. Divide all coords by scale (or 1e-6 if scale is near zero to avoid div-by-zero).
          5. Flatten back to (63,).

        This makes the same gesture invariant to where on screen the hand is
        and how far from the camera it is.
        """
        coords = landmarks_flat.reshape(21, 3).copy()

        # 1. Translate so wrist is at origin
        wrist = coords[_WRIST].copy()
        coords -= wrist

        # 2. Compute scale: distance from wrist (now at 0) to index MCP
        scale = float(np.linalg.norm(coords[_MCP_INDEX]))
        if scale < 1e-6:
            scale = 1e-6

        # 3. Scale-normalise
        coords /= scale

        return coords.flatten().astype(np.float32)

    @staticmethod
    def _flatten_landmarks(landmarks) -> np.ndarray:
        coords = []
        for lm in landmarks:
            coords.extend([lm.x, lm.y, lm.z])
        return np.array(coords, dtype=np.float32)

    def _pick_best_hand_solutions(self):
        """
        Pick the best hand from the solutions backend result.
        Accepts ANY detected hand — prefer the one with higher wrist confidence
        (MediaPipe Solutions doesn't expose per-hand confidence directly, so we
        just take the first detected hand which is usually the most confident).
        Returns (hand_landmarks, handedness_label) or (None, None).
        """
        results = self.last_results
        if not results.multi_hand_landmarks or not results.multi_handedness:
            return None, None

        # Take the first detected hand (MediaPipe returns highest-confidence first)
        hand_landmarks = results.multi_hand_landmarks[0]
        handedness = results.multi_handedness[0]
        label = getattr(handedness.classification[0], "label", "Unknown")
        return hand_landmarks, label

    @staticmethod
    def _select_primary_index(labels: list[str]) -> int:
        for idx, label in enumerate(labels):
            if label.lower() == "right":
                return idx
        return 0

    def extract_all(self, frame: np.ndarray) -> list[dict]:
        """Return normalized landmarks for all detected hands.

        Each entry has keys: label, landmarks.
        """
        if frame is None:
            self.last_results = None
            self.last_right_hand_landmarks = None
            self.last_right_hand_proto = None
            self.last_detected_handedness = None
            self.last_hand_landmarks = []
            self.last_hand_protos = []
            self.last_detected_handedness_all = []
            return []

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.last_right_hand_landmarks = None
        self.last_right_hand_proto = None
        self.last_detected_handedness = None
        self.last_hand_landmarks = []
        self.last_hand_protos = []
        self.last_detected_handedness_all = []

        output = []

        if self.backend == "solutions":
            self.last_results = self.hands.process(rgb_frame)
            if not self.last_results.multi_hand_landmarks:
                return []

            for idx, hand_landmarks in enumerate(self.last_results.multi_hand_landmarks):
                label = "Unknown"
                if self.last_results.multi_handedness and idx < len(self.last_results.multi_handedness):
                    handedness = self.last_results.multi_handedness[idx]
                    label = getattr(handedness.classification[0], "label", "Unknown")

                self.last_hand_protos.append(hand_landmarks)
                self.last_hand_landmarks.append(hand_landmarks.landmark)
                self.last_detected_handedness_all.append(label)

                raw = self._flatten_landmarks(hand_landmarks.landmark)
                output.append({"label": label, "landmarks": self._normalize_landmarks(raw)})

            primary_idx = self._select_primary_index(self.last_detected_handedness_all)
            if self.last_hand_landmarks:
                self.last_right_hand_landmarks = self.last_hand_landmarks[primary_idx]
                self.last_right_hand_proto = self.last_hand_protos[primary_idx]
                self.last_detected_handedness = self.last_detected_handedness_all[primary_idx]
            return output

        # Tasks backend
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.last_results = self.landmarker.detect(mp_image)
        if not self.last_results.hand_landmarks:
            return []

        for idx, hand_landmarks in enumerate(self.last_results.hand_landmarks):
            label = "Unknown"
            if self.last_results.handedness and idx < len(self.last_results.handedness):
                cat = self.last_results.handedness[idx]
                label = getattr(cat[0], "category_name", "Unknown") if cat else "Unknown"

            self.last_hand_landmarks.append(hand_landmarks)
            self.last_detected_handedness_all.append(label)

            raw = self._flatten_landmarks(hand_landmarks)
            output.append({"label": label, "landmarks": self._normalize_landmarks(raw)})

        primary_idx = self._select_primary_index(self.last_detected_handedness_all)
        if self.last_hand_landmarks:
            self.last_right_hand_landmarks = self.last_hand_landmarks[primary_idx]
            self.last_detected_handedness = self.last_detected_handedness_all[primary_idx]
        return output

    def extract(self, frame: np.ndarray) -> np.ndarray:
        zero_vector = np.zeros(63, dtype=np.float32)
        hands = self.extract_all(frame)
        if not hands:
            return zero_vector

        primary_idx = self._select_primary_index(self.last_detected_handedness_all)
        if primary_idx >= len(hands):
            return hands[0]["landmarks"]
        return hands[primary_idx]["landmarks"]

    def draw_right_hand(self, frame: np.ndarray) -> None:
        if frame is None or self.last_right_hand_landmarks is None:
            return

        if self.backend == "solutions":
            self.mp_drawing.draw_landmarks(
                frame,
                self.last_right_hand_proto,
                self.mp_hands.HAND_CONNECTIONS,
            )
            return

        h, w = frame.shape[:2]
        points = []
        for lm in self.last_right_hand_landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append((x, y))
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, points[a], points[b], (0, 255, 0), 2)

    def draw_all_hands(self, frame: np.ndarray) -> None:
        if frame is None or not self.last_hand_landmarks:
            return

        if self.backend == "solutions":
            for proto in self.last_hand_protos:
                self.mp_drawing.draw_landmarks(
                    frame,
                    proto,
                    self.mp_hands.HAND_CONNECTIONS,
                )
            return

        h, w = frame.shape[:2]
        for hand_landmarks in self.last_hand_landmarks:
            points = []
            for lm in hand_landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                points.append((x, y))
                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, points[a], points[b], (0, 255, 0), 2)

    def get_right_hand_landmarks(self):
        return self.last_right_hand_landmarks

    def close(self) -> None:
        if self.hands is not None:
            self.hands.close()
        if self.landmarker is not None:
            self.landmarker.close()
