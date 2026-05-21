import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from extractor import MediaPipeExtractor


DATASET_ROOT = Path("data/dataset")
OUTPUT_CSV = Path("data/landmarks.csv")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_header():
    header = ["label"]
    for idx in range(21):
        header.extend([f"x{idx}", f"y{idx}", f"z{idx}"])
    return header


def collect_images():
    items = []
    if not DATASET_ROOT.exists():
        return items

    label_dirs = sorted(
        (path for path in DATASET_ROOT.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda p: int(p.name),
    )
    for label_dir in label_dirs:
        label = label_dir.name
        for image_path in sorted(label_dir.iterdir()):
            if image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                items.append((label, image_path))
    return items


def main():
    extractor = MediaPipeExtractor(mode=True)
    samples = collect_images()
    if not samples:
        raise FileNotFoundError(
            "No dataset images found under data/dataset/<label> folders. "
            "Expected numeric label directories containing image files."
        )
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(build_header())

        for label, image_path in tqdm(samples, desc="Extracting landmarks", unit="image"):
            frame = cv2.imread(str(image_path))
            if frame is None:
                skipped += 1
                continue

            landmarks = extractor.extract(frame)
            if np.allclose(landmarks, 0.0):
                skipped += 1
                continue

            writer.writerow([label, *landmarks.tolist()])
            processed += 1

    extractor.close()
    print(f"Done. processed={processed} skipped={skipped} total={len(samples)}")
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
