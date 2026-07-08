import sys
import json
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# Add project root to path to resolve absolute imports correctly
sys.path.append(str(Path(__file__).resolve().parents[2]))
from video_model.preprocessing.video_extractor import MediaPipeVideoExtractor

def generate_landmarks(project_root: Path, limit=None):
    metadata_json_path = project_root / "Video_word_context" / "WLASL_v0.3.json"
    dataset_dir = project_root / "video_dataset"
    landmarks_dir = project_root / "video_model" / "landmarks"

    # Create target directories
    for split_dir in ["train", "validation", "test"]:
        (landmarks_dir / split_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading metadata from {metadata_json_path}...")
    with open(metadata_json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Map video ID to its metadata details
    video_metadata = {}
    for entry in metadata:
        gloss = entry["gloss"]
        for inst in entry["instances"]:
            video_id = inst["video_id"]
            video_metadata[video_id] = {
                "gloss": gloss,
                "split": inst["split"]
            }

    # Find all videos in organized dataset_dir
    video_paths = list(dataset_dir.glob("**/*.mp4"))
    print(f"Found {len(video_paths)} organized videos.")

    # Support limit for testing/speed
    if limit is not None:
        video_paths = video_paths[:limit]
        print(f"Limiting execution to first {limit} videos.")

    extractor = MediaPipeVideoExtractor()
    processed_count = 0
    skipped_count = 0

    split_map = {
        "train": "train",
        "val": "validation",
        "test": "test"
    }

    print("Extracting landmarks from videos...")
    for video_path in tqdm(video_paths):
        video_id = video_path.stem

        # Look up split/metadata
        meta = video_metadata.get(video_id)
        if not meta:
            # Fallback to int matching
            meta = video_metadata.get(str(int(video_id)))

        if not meta:
            print(f"[WARNING] No metadata found for video ID {video_id}, skipping...")
            skipped_count += 1
            continue

        split_name = meta["split"]
        target_split_folder = split_map.get(split_name)
        if not target_split_folder:
            print(f"[WARNING] Invalid split {split_name} for video ID {video_id}, skipping...")
            skipped_count += 1
            continue

        dest_file = landmarks_dir / target_split_folder / f"{video_id}.npy"

        # Skip if already exists
        if dest_file.exists():
            processed_count += 1
            continue

        # Extract landmarks frame-by-frame
        cap = cv2.VideoCapture(str(video_path))
        frame_landmarks = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Extract landmarks for this frame
            lms = extractor.extract_frame(frame)
            frame_landmarks.append(lms)

        cap.release()

        if len(frame_landmarks) > 0:
            seq_array = np.array(frame_landmarks, dtype=np.float32)
            np.save(dest_file, seq_array)
            processed_count += 1
        else:
            print(f"[WARNING] No frames read from {video_path.name}")
            skipped_count += 1

    extractor.close()
    print(f"Landmark generation complete. Processed/verified: {processed_count}, Skipped: {skipped_count}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos processed")
    args = parser.parse_args()

    proj_dir = Path(__file__).resolve().parents[2]
    generate_landmarks(proj_dir, limit=args.limit)
