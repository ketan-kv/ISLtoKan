import urllib.request
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp

HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# MediaPipe Connection indices
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28)
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# Standard 40 facial landmark indices (lips contour, eyes contour, brows, nose)
FACE_KEY_INDICES = [
    33, 133, 362, 263, 61, 291, 0, 17, 78, 308, 14, 324, 13, 312, 27, 257, 50, 280, 102, 331,
    223, 443, 70, 107, 336, 285, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 327
]

class MediaPipeVideoExtractor:
    def __init__(self, project_root: Path = None):
        if project_root is None:
            project_root = Path(__file__).resolve().parents[2]
            
        self.assets_dir = project_root / "assets" / "models"
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        
        hand_model_path = self._resolve_model("hand_landmarker.task", HAND_MODEL_URL)
        pose_model_path = self._resolve_model("pose_landmarker_full.task", POSE_MODEL_URL)
        face_model_path = self._resolve_model("face_landmarker.task", FACE_MODEL_URL)
        
        from mediapipe.tasks.python import vision
        
        # Initialize Hand Landmarker
        hand_options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(hand_model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
        
        # Initialize Pose Landmarker
        pose_options = vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(pose_model_path)),
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

        # Initialize Face Landmarker
        face_options = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(face_model_path)),
            running_mode=vision.RunningMode.IMAGE,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
        
        # Cache for drawing overlays
        self.last_pose_landmarks = None
        self.last_hand_landmarks = []
        self.last_face_landmarks = None

    def _resolve_model(self, name: str, url: str) -> Path:
        model_path = self.assets_dir / name
        if not model_path.exists():
            print(f"Downloading model {name} from {url}...")
            try:
                urllib.request.urlretrieve(url, str(model_path))
            except Exception as e:
                fallback_path = Path(__file__).resolve().parent / name
                if fallback_path.exists():
                    return fallback_path
                raise RuntimeError(f"Failed to download {name}: {e}")
        return model_path

    def extract_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract pose, hands, and face landmarks from a single video frame.
        Normalizes coordinates to make features position invariant.
        
        Returns a flat float32 array of shape (345,):
        [Left Hand (63), Right Hand (63), Pose (99), Face (120)]
        """
        self.last_pose_landmarks = None
        self.last_hand_landmarks = []
        self.last_face_landmarks = None
        
        if frame is None:
            return np.zeros(345, dtype=np.float32)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Initialize output vectors
        lh_coords = np.zeros(63, dtype=np.float32)
        rh_coords = np.zeros(63, dtype=np.float32)
        pose_coords = np.zeros(99, dtype=np.float32)
        face_coords = np.zeros(120, dtype=np.float32)
        
        # 1. Pose landmarks
        pose_res = self.pose_landmarker.detect(mp_image)
        if pose_res.pose_landmarks:
            self.last_pose_landmarks = pose_res.pose_landmarks[0]
            landmarks = self.last_pose_landmarks
            
            # Calculate mid-shoulder point for normalization
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            mid_x = (left_shoulder.x + right_shoulder.x) / 2.0
            mid_y = (left_shoulder.y + right_shoulder.y) / 2.0
            mid_z = (left_shoulder.z + right_shoulder.z) / 2.0
            
            pose_list = []
            for lm in landmarks:
                # Store coordinates normalized relative to mid-shoulder
                pose_list.extend([lm.x - mid_x, lm.y - mid_y, lm.z - mid_z])
            pose_coords = np.array(pose_list, dtype=np.float32)
            
        # 2. Hand landmarks
        hand_res = self.hand_landmarker.detect(mp_image)
        if hand_res.hand_landmarks and hand_res.handedness:
            self.last_hand_landmarks = hand_res.hand_landmarks
            for hand_landmarks, handedness in zip(hand_res.hand_landmarks, hand_res.handedness):
                label = handedness[0].category_name  # 'Left' or 'Right'
                
                # Extract coordinates
                coords_list = []
                for lm in hand_landmarks:
                    coords_list.append([lm.x, lm.y, lm.z])
                coords = np.array(coords_list, dtype=np.float32)
                
                # Normalize relative to wrist (index 0)
                wrist = coords[0].copy()
                coords -= wrist
                
                flattened = coords.flatten()
                
                if label == 'Left':
                    lh_coords = flattened
                elif label == 'Right':
                    rh_coords = flattened
                    
        # 3. Face landmarks
        face_res = self.face_landmarker.detect(mp_image)
        if face_res.face_landmarks:
            self.last_face_landmarks = face_res.face_landmarks[0]
            landmarks = self.last_face_landmarks
            
            # Nose bridge (index 8) coordinate for normalization
            nose = landmarks[8]
            
            face_list = []
            for idx in FACE_KEY_INDICES:
                lm = landmarks[idx]
                face_list.extend([lm.x - nose.x, lm.y - nose.y, lm.z - nose.z])
            face_coords = np.array(face_list, dtype=np.float32)
            
        # Concatenate features into a single 345-element vector
        return np.concatenate([lh_coords, rh_coords, pose_coords, face_coords])

    def draw_overlays(self, frame: np.ndarray, show_pose: bool = True, show_hands: bool = True, show_face: bool = True):
        """Draw pose, hand, and face skeletons on frame."""
        if frame is None:
            return
            
        height, width = frame.shape[:2]
        
        # 1. Draw Pose skeleton
        if show_pose and self.last_pose_landmarks is not None:
            points = {}
            for idx, lm in enumerate(self.last_pose_landmarks):
                if 11 <= idx <= 28:
                    x_pos = int(lm.x * width)
                    y_pos = int(lm.y * height)
                    points[idx] = (x_pos, y_pos)
                    cv2.circle(frame, (x_pos, y_pos), 4, (0, 255, 255), -1)
            
            for a, b in POSE_CONNECTIONS:
                if a in points and b in points:
                    cv2.line(frame, points[a], points[b], (255, 0, 0), 2)
                    
        # 2. Draw Hand skeletons
        if show_hands and self.last_hand_landmarks:
            for hand_landmarks in self.last_hand_landmarks:
                points = []
                for lm in hand_landmarks:
                    x_pos = int(lm.x * width)
                    y_pos = int(lm.y * height)
                    points.append((x_pos, y_pos))
                    cv2.circle(frame, (x_pos, y_pos), 3, (0, 0, 255), -1)
                    
                for a, b in HAND_CONNECTIONS:
                    if a < len(points) and b < len(points):
                        cv2.line(frame, points[a], points[b], (0, 255, 0), 2)

        # 3. Draw Face landmarks (key facial contours)
        if show_face and self.last_face_landmarks is not None:
            for idx in FACE_KEY_INDICES:
                lm = self.last_face_landmarks[idx]
                x_pos = int(lm.x * width)
                y_pos = int(lm.y * height)
                cv2.circle(frame, (x_pos, y_pos), 1, (255, 255, 0), -1)

    def close(self):
        """Release MediaPipe resources."""
        self.hand_landmarker.close()
        self.pose_landmarker.close()
        self.face_landmarker.close()
