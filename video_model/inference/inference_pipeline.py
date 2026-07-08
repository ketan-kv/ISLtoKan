import json
from pathlib import Path
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F

# Import modular sequence models
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from video_model.models.sequence_model import SequenceClassifier


class VideoInferencePipeline:
    def __init__(self, project_root: Path, window_size: int = 40, min_frames: int = 12):
        self.project_root = project_root
        self.window_size = window_size
        self.min_frames = min_frames

        # Load vocabulary
        vocab_path = project_root / "video_model" / "landmarks" / "vocabulary.json"
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}. Please run preprocess/training steps first.")

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        # Reverse vocabulary to map index -> gloss name
        self.idx_to_gloss = {idx: gloss for gloss, idx in self.vocab.items()}
        self.num_classes = len(self.vocab)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_status = "Not Loaded"

        # Sliding queues for sequence modeling
        self.frame_buffer = deque(maxlen=window_size)
        self.prediction_history = deque(maxlen=8)

    def load_model(self) -> bool:
        """Load the best trained sequence classifier model weights."""
        model_weight_path = self.project_root / "video_model" / "models" / "best_model.pth"
        if not model_weight_path.exists():
            self.model_status = "No trained model found"
            print(f"[WARNING] Model weights not found at {model_weight_path}")
            return False

        try:
            checkpoint = torch.load(model_weight_path, map_location=self.device)
            encoder_type = checkpoint.get('encoder_type', 'lstm')

            self.model = SequenceClassifier(
                encoder_type=encoder_type,
                num_classes=self.num_classes,
                input_size=345,
                hidden_size=128,
                num_layers=2,
                dropout=0.2,
                bidirectional=True
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            self.model_status = f"Loaded ({encoder_type.upper()}) on {self.device}"
            print(f"Successfully loaded sequence model: {self.model_status}")
            return True
        except Exception as e:
            self.model_status = f"Load Error: {e}"
            print(f"[ERROR] Failed to load sequence model: {e}")
            return False

    def reset(self):
        """Clear sequence and prediction histories."""
        self.frame_buffer.clear()
        self.prediction_history.clear()

    def process_frame_landmarks(self, landmarks_345: np.ndarray):
        """
        Append new frame landmarks, run sequence classification, and return prediction details.

        Returns:
            dict containing:
                - 'word': Predicted gloss/word (or '-' if insufficient frames / no model)
                - 'confidence': Prediction probability
                - 'stability': Floating percentage representing predictions consistency
                - 'sequence_len': Current frames in buffer
                - 'status': Model status details
        """
        self.frame_buffer.append(landmarks_345)

        # Default empty prediction state
        result = {
            "word": "-",
            "confidence": 0.0,
            "stability": 0.0,
            "sequence_len": len(self.frame_buffer),
            "status": self.model_status
        }

        if self.model is None:
            return result

        if len(self.frame_buffer) < self.min_frames:
            return result

        # Prepare inputs for PyTorch inference
        seq_array = np.array(self.frame_buffer, dtype=np.float32) # [seq_len, 345]
        input_tensor = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0).to(self.device) # [1, seq_len, 345]
        length_tensor = torch.tensor([len(seq_array)], dtype=torch.long)

        with torch.no_grad():
            logits = self.model(input_tensor, length_tensor)
            probs = F.softmax(logits, dim=1)[0]

            best_idx = int(torch.argmax(probs).item())
            confidence = float(probs[best_idx].item())

        predicted_word = self.idx_to_gloss[best_idx]
        self.prediction_history.append(predicted_word)

        # Calculate prediction stability (percentage of matching predictions in recent history)
        most_common = predicted_word
        if self.prediction_history:
            most_common = max(set(self.prediction_history), key=list(self.prediction_history).count)
            match_count = list(self.prediction_history).count(most_common)
            stability = (match_count / len(self.prediction_history)) * 100.0
        else:
            stability = 0.0

        result.update({
            "word": most_common,
            "confidence": confidence,
            "stability": stability
        })

        return result
