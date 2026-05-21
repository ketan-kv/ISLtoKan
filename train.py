from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score


LANDMARKS_CSV = Path("data/landmarks.csv")
MODEL_PATH = Path("model.pkl")


def main():
    if not LANDMARKS_CSV.exists():
        raise FileNotFoundError(f"Missing input file: {LANDMARKS_CSV}")

    data = np.loadtxt(str(LANDMARKS_CSV), delimiter=",", skiprows=1)
    if data.size == 0:
        raise ValueError(
            "data/landmarks.csv has no samples. Run preprocess.py and ensure some hands are detected."
        )
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 64:
        raise ValueError(
            f"Invalid landmarks shape: expected 64 columns (label + 63 landmarks), got {data.shape[1]}."
        )
    if np.isnan(data).any():
        raise ValueError("data/landmarks.csv contains invalid numeric values (NaN).")

    y = data[:, 0].astype(int)
    X = data[:, 1:]

    if len(y) < 2:
        raise ValueError("Need at least 2 samples in data/landmarks.csv to train classifier.")
    classes, class_counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError(
            "Need at least 2 distinct gesture labels in data/landmarks.csv. "
            f"Found labels: {classes.tolist()}"
        )
    if np.any(class_counts < 2):
        raise ValueError(
            "Each class must have at least 2 samples for stratified 80/20 split. "
            f"Current class counts: {dict(zip(classes.tolist(), class_counts.tolist()))}"
        )

    print(f"Dataset: {len(y)} samples, {len(classes)} classes: {classes.tolist()}")
    print(f"Samples per class: {dict(zip(classes.tolist(), class_counts.tolist()))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining on {len(X_train)} samples, testing on {len(X_test)} samples...")

    # Random Forest — much more powerful than KNN for this task:
    #   • Scale-invariant: no distance metric sensitivity
    #   • Fast inference: O(depth) per tree, not O(n) like KNN
    #   • Robust to slight hand orientation variance
    #   • Returns well-calibrated probabilities for confidence thresholding
    model = RandomForestClassifier(
        n_estimators=200,    # 200 trees for strong ensemble
        max_depth=None,      # Grow fully — data is clean
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt", # Standard for classification
        class_weight="balanced",  # Handles any slight class imbalance
        random_state=42,
        n_jobs=-1,           # Use all CPU cores
    )
    model.fit(X_train, y_train)

    # 5-fold cross-validation on full training data for a reliable accuracy estimate
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"\n5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\nClassification report (held-out test set):")
    print(classification_report(y_test, y_pred))
    print(f"Test Accuracy: {accuracy:.4f}")

    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model: {MODEL_PATH}")
    print("Done! Run main.py to start real-time recognition.")


if __name__ == "__main__":
    main()
