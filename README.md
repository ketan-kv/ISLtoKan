# Sign2Kannada

Sign2Kannada is a sign-recognition project that has two independent pipelines:
1. **Image Recognition Mode**: Extracts hand landmarks from images or webcam frames, trains a lightweight Random Forest classifier on those landmarks, and shows real-time predictions for digits (0-9) with Kannada text output.
2. **Video Recognition Mode**: Preprocesses video datasets (WLASL) using MediaPipe Holistic Pose, Hands, and Face tracking to extract 345-dimensional coordinate sequences. Trains a PyTorch sequence LSTM/Transformer classifier for word predictions, and runs real-time camera inference with confidence smoothing, prediction stability tracking, and skeletal overlays.

---

## 🏗️ Project Layout

### 1. Image Digit recognition (Image Mode)
- `extractor.py` - hand landmark extraction and drawing utilities.
- `preprocess.py` - extracts landmarks from the image dataset into `data/landmarks.csv`.
- `combine_landmarks.py` - combines dataset variants into a single CSV.
- `mirror_dataset.py` - creates mirrored samples for left/right hand balance.
- `preprocess_mirrored.py` - preprocesses the mirrored dataset.
- `train.py` - trains and saves `model.pkl`.
- `main.py` - webcam demo and live prediction UI.
- `translator.py` - digit-to-Kannada mapping.
- `config.py` - shared paths, thresholds, and colors.

### 2. Word Sequence recognition (Video Mode)
Located under `video_model/`:
- `video_model/dataset/organizer.py` - parses metadata and organizes videos by class gloss.
- `video_model/preprocessing/video_extractor.py` - MediaPipe Holistic Pose + Hands + Face landmarker and overlay drawing utilities.
- `video_model/preprocessing/generator.py` - converts variable length videos into `.npy` landmark sequences of shape `(seq_len, 345)`.
- `video_model/preprocessing/label_generator.py` - indexes sequences into `labels.csv` and `vocabulary.json`.
- `video_model/models/sequence_model.py` - modular sequence model architectures (LSTM & Transformer Encoder).
- `video_model/training/train.py` - unbuffered, RAM-cached training script with sequence augmentations (frame drop, noise, translations, scaling, Z-axis roll rotations).
- `video_model/inference/inference_pipeline.py` - sliding window prediction, smoothing, and stability tracking.
- `video_model/utils/visualizer.py` - debug visualization video exporter.

---

## ⚙️ Setup and Requirements

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Large video assets (`video_data/`, `video_dataset/`, `video_model/`, `Video_word_context/`) are local-only and ignored by git to keep checkout times under 30 seconds. Point your raw video source to `C:\Users\Dayananda Shetty\Downloads\videos` or create a junction link to let the organizer find it.

---

## 🛠️ Video Pipeline Workflows

### 1. Organize Dataset
```bash
python video_model/dataset/organizer.py
```
Organizes raw videos into class directories and generates `integrity_report.md`.

### 2. Extract Skeletal Landmark Sequences
```bash
python video_model/preprocessing/generator.py
```
Extracts Pose, Hands, and Face coordinates. Normalized sequences are saved under `video_model/landmarks/`.

### 3. Generate Label Indexing
```bash
python video_model/preprocessing/label_generator.py
```
Generates dataset mapping files `labels.csv` and `vocabulary.json`.

### 4. Train Sequence Classifier
```bash
python video_model/training/train.py --encoder lstm --epochs 30 --batch_size 32
```
Trains the PyTorch model with RAM-cached dataset optimization and sequence data augmentations. Saved weights are placed in `video_model/models/best_model.pth`.

### 5. Run Live Inference
```bash
python main.py
```
* **'M' key**: Toggle between Image Digits mode and Video Continuous Word mode.
* **'P' key**: Toggle Pose skeletal overlay (ON/OFF).
* **'H' key**: Toggle Hands skeletal overlay (ON/OFF).
* **'F' key**: Toggle Face mesh overlay (ON/OFF).
* **'Q' key**: Quit.

---

## 🌐 Notes

- Digit predictions are mapped to Kannada inside `translator.py`.
- Word translations are loaded dynamically using the lookup map in `main.py`.
- Visual inspections of tracking can be done using the sample exporter script `python video_model/utils/visualizer.py`.

