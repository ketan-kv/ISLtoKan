# Sign2Kannada

Sign2Kannada is a hand-sign recognition project that extracts hand landmarks from images or webcam frames, trains a lightweight classifier on those landmarks, and shows real-time predictions with Kannada text output.

## What it does

- Extracts 21 hand landmarks per frame with MediaPipe.
- Builds a CSV dataset from images under `data/dataset/<label>/`.
- Trains a `KNeighborsClassifier` on landmark features.
- Runs live webcam inference with confidence tracking and smoothing.
- Renders Kannada output using a bundled Kannada font when available.

## Project layout

- `extractor.py` - hand landmark extraction and drawing utilities.
- `preprocess.py` - extracts landmarks from the image dataset into `data/landmarks.csv`.
- `combine_landmarks.py` - combines dataset variants into a single CSV.
- `mirror_dataset.py` - creates mirrored samples for left/right hand balance.
- `preprocess_mirrored.py` - preprocesses the mirrored dataset.
- `train.py` - trains and saves `model.pkl`.
- `main.py` - webcam demo and live prediction UI.
- `translator.py` - digit-to-Kannada mapping.
- `config.py` - shared paths, thresholds, and colors.

## Requirements

Install the Python dependencies listed in `requirements.txt`.

## Setup

1. Create and activate your virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure your dataset is present under `data/dataset/` using numeric label folders such as `0`, `1`, `2`, and so on.

## Training workflow

### 1. Extract landmarks

```bash
python preprocess.py
```

This creates `data/landmarks.csv` from the source images.

### 2. Optionally combine or mirror data

If you are using the mirrored or combined dataset pipeline, run the matching helper scripts before training.

### 3. Train the model

```bash
python train.py
```

This trains the classifier and saves the model to `model.pkl`.

### 4. Run live inference

```bash
python main.py
```

Press `Q` in the webcam window to quit.

## Data format

The landmark CSV uses one label column followed by 63 landmark values:

- `label`
- `x0, y0, z0`
- `x1, y1, z1`
- ...
- `x20, y20, z20`

## Kannada output

Digit predictions are translated using `translator.py`:

- `0` -> `ಸೊನ್ನೆ`
- `1` -> `ಒಂದು`
- `2` -> `ಎರಡು`
- `3` -> `ಮೂರು`
- `4` -> `ನಾಲ್ಕು`
- `5` -> `ಐದು`
- `6` -> `ಆರು`
- `7` -> `ಏಳು`
- `8` -> `ಎಂಟು`
- `9` -> `ಒಂಬತ್ತು`

## Notes

- A valid Kannada TTF font is expected under `assets/fonts/` for best text rendering.
- The model needs at least two gesture classes to train properly.
- If you only train on a single label, predictions will appear stuck to that class.
