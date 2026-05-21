# Sign2Kannada - Program Design and Working

## 1. Purpose
Sign2Kannada is a hand-sign recognition pipeline that:
- extracts hand landmarks from images/webcam frames,
- trains a digit classifier,
- translates predicted digits into Kannada words,
- displays live prediction + confidence in a webcam window.

---

## 2. Project Files and Responsibilities

## `requirements.txt`
Declares required libraries:
- `mediapipe`, `opencv-python`, `numpy`, `scikit-learn`, `tqdm`, `Pillow`, `joblib`.

## `extractor.py`
Contains `MediaPipeExtractor`, the feature extraction core.

### What it does
1. Accepts `mode`:
   - `True`: static/offline image usage (`preprocess.py`)
   - `False`: webcam/live usage (`main.py`)
2. Uses MediaPipe hand detection backend:
   - tries classic `mp.solutions` if available,
   - otherwise uses MediaPipe Tasks backend and a `.task` model.
3. For each frame, returns a **63-length vector**:
   - 21 landmarks × (x, y, z).
4. Hand selection strategy:
   - prefers right hand,
   - falls back to first detected hand when right is unavailable.
5. Returns all-zero vector if no hand is detected.
6. Provides drawing support for hand skeleton overlay.

---

## `preprocess.py`
Builds training data (`data/landmarks.csv`) from dataset images.

### Expected input
Image folders under:
`data/dataset/<label>/...` where `<label>` is numeric.

### Flow
1. Scans numeric label folders dynamically.
2. Reads each image with OpenCV.
3. Extracts 63 features using `MediaPipeExtractor(mode=True)`.
4. Skips samples with zero-vector features (no hand detected).
5. Writes CSV with columns:
   - `label, x0,y0,z0, ... , x20,y20,z20`
6. Shows tqdm progress and final summary (`processed`, `skipped`, `total`).

---

## `train.py`
Trains and evaluates the classifier.

### Flow
1. Loads `data/landmarks.csv`.
2. Validates:
   - file exists,
   - has valid numeric shape (64 columns),
   - no NaN values,
   - enough samples,
   - at least 2 distinct labels,
   - enough per-class samples for stratified split.
3. Splits data 80/20 (`train_test_split(..., stratify=y)`).
4. Trains `KNeighborsClassifier(n_neighbors=5)`.
5. Prints classification report + accuracy.
6. Saves trained model to `model.pkl` using `joblib`.

---

## `translator.py`
Stores the Kannada mapping dictionary:
- `"0"` -> `"ಸೊನ್ನೆ"`, `"1"` -> `"ಒಂದು"`, ..., `"9"` -> `"ಒಂಬತ್ತು"`.

---

## `main.py`
Runs real-time webcam inference.

### Runtime behavior
1. Loads:
   - trained `model.pkl`
   - Kannada font via PIL (`Noto Sans Kannada` path resolution)
2. Opens webcam (`cv2.VideoCapture(0)`).
3. For each frame:
   - extracts landmarks via `MediaPipeExtractor(mode=False)`,
   - draws hand skeleton,
   - predicts class probabilities with the trained KNN,
   - updates stable digit/confidence,
   - converts digit to Kannada word via `translator.py`,
   - overlays:
     - digit (OpenCV text),
     - Kannada word (PIL text, dark with outline),
     - confidence bar.
4. If no hand is detected for a while, display state resets.
5. Press `q` to quit.

---

## 3. End-to-End Working Sequence

1. **Preprocess**
   - `python preprocess.py`
   - Output: `data/landmarks.csv`
2. **Train**
   - `python train.py`
   - Output: `model.pkl`
3. **Run Demo**
   - `python main.py`
   - Output: live webcam prediction UI.

---

## 4. Intended Design Decisions

- Landmark-based approach (63 numeric features) keeps model lightweight.
- KNN chosen for simple baseline classification.
- Training guards fail early with clear errors for bad/incomplete data.
- PIL rendering ensures Kannada text support where OpenCV fonts fail.
- Backend fallback in extractor improves compatibility across MediaPipe versions.

---

## 5. Important Practical Note

If training data contains only one label (example: only `9`), model predictions will appear "stuck" to that label.  
For true real-time gesture change behavior, dataset must include multiple gesture classes with enough samples per class, then re-run preprocessing and training.
