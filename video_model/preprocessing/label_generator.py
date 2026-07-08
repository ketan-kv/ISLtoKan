import csv
import json
from pathlib import Path

def generate_labels_and_vocab(project_root: Path):
    landmarks_dir = project_root / "video_model" / "landmarks"
    metadata_json_path = project_root / "Video_word_context" / "WLASL_v0.3.json"
    
    # Load metadata to map video ID to word gloss
    print(f"Loading metadata from {metadata_json_path}...")
    with open(metadata_json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    video_to_gloss = {}
    for entry in metadata:
        gloss = entry["gloss"]
        for inst in entry["instances"]:
            video_id = inst["video_id"]
            video_to_gloss[video_id] = gloss
            # Support integer mapping fallback
            video_to_gloss[str(int(video_id))] = gloss
            
    # Gather all .npy files in split folders
    rows = []
    unique_glosses = set()
    
    split_folders = {
        "train": "train",
        "validation": "validation",
        "test": "test"
    }
    
    for split_name, folder_name in split_folders.items():
        folder_path = landmarks_dir / folder_name
        if not folder_path.exists():
            continue
            
        npy_files = list(folder_path.glob("*.npy"))
        print(f"Split '{split_name}': found {len(npy_files)} files.")
        
        for npy_file in npy_files:
            video_id = npy_file.stem
            gloss = video_to_gloss.get(video_id)
            if not gloss:
                gloss = video_to_gloss.get(str(int(video_id)))
                
            if not gloss:
                print(f"[WARNING] No gloss mapping found for landmark ID {video_id}, skipping labels registration.")
                continue
                
            unique_glosses.add(gloss)
            
            # Use relative path from project root
            rel_path = npy_file.relative_to(project_root).as_posix()
            
            rows.append({
                "video_id": video_id,
                "gloss": gloss,
                "split": split_name,
                "filepath": rel_path
            })
            
    # Create sorted vocabulary mapping
    sorted_glosses = sorted(list(unique_glosses))
    vocab = {gloss: idx for idx, gloss in enumerate(sorted_glosses)}
    
    # Save vocabulary
    vocab_path = landmarks_dir / "vocabulary.json"
    with open(vocab_path, "w", encoding="utf-8") as vf:
        json.dump(vocab, vf, indent=4)
    print(f"Vocabulary saved to {vocab_path} (unique classes: {len(vocab)})")
    
    # Save labels.csv
    labels_csv_path = landmarks_dir / "labels.csv"
    with open(labels_csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=["video_id", "gloss", "split", "filepath"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Labels CSV saved to {labels_csv_path} (total rows: {len(rows)})")

if __name__ == "__main__":
    proj_dir = Path(__file__).resolve().parents[2]
    generate_labels_and_vocab(proj_dir)
