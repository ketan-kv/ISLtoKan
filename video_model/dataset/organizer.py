import os
import json
import shutil
from pathlib import Path
import cv2
from tqdm import tqdm

def organize_and_check(project_root: Path):
    video_data_dir = project_root / "video_data"
    metadata_json_path = project_root / "Video_word_context" / "WLASL_v0.3.json"
    dest_dataset_dir = project_root / "video_dataset"
    
    # Create output directory
    dest_dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading metadata from {metadata_json_path}...")
    with open(metadata_json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    # Get all video files in video_data
    video_files = list(video_data_dir.glob("*.mp4"))
    print(f"Found {len(video_files)} videos in {video_data_dir}")
    
    # Index available videos by integer and string forms
    available_videos = {}
    for vf in video_files:
        name = vf.stem
        # Store both absolute name and parsed integer form
        available_videos[name] = vf
        try:
            available_videos[str(int(name))] = vf
        except ValueError:
            pass

    copied_count = 0
    missing_videos = []
    corrupted_videos = []
    mapped_count = 0
    mismatches = []
    
    # Word gloss folders mapping
    gloss_instances = {}
    
    print("Organizing dataset and verifying integrity...")
    for entry in tqdm(metadata):
        gloss = entry["gloss"]
        instances = entry["instances"]
        
        for inst in instances:
            video_id = inst["video_id"]
            split = inst["split"]
            mapped_count += 1
            
            # Check if video is available
            src_file = None
            # Check direct match
            if video_id in available_videos:
                src_file = available_videos[video_id]
            # Check int match
            elif str(int(video_id)) in available_videos:
                src_file = available_videos[str(int(video_id))]
                
            if src_file is None:
                missing_videos.append((video_id, gloss, split))
                continue
                
            # Define destination path
            word_dir = dest_dataset_dir / gloss
            word_dir.mkdir(parents=True, exist_ok=True)
            dest_file = word_dir / f"{video_id}.mp4"
            
            # Copy file (safer than move, doesn't break original folder)
            try:
                if not dest_file.exists():
                    shutil.copy2(src_file, dest_file)
                copied_count += 1
            except Exception as e:
                mismatches.append(f"Failed to copy {src_file.name} to {dest_file}: {e}")
                continue
                
            # Verify if corrupted
            cap = cv2.VideoCapture(str(dest_file))
            if not cap.isOpened():
                corrupted_videos.append((video_id, gloss, split, "Could not open video file"))
            else:
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if frame_count <= 0 or width <= 0 or height <= 0:
                    corrupted_videos.append((video_id, gloss, split, f"Invalid dimension or frames (frames={frame_count}, {width}x{height})"))
                cap.release()

    # Find unmapped videos
    mapped_video_ids = set()
    for entry in metadata:
        for inst in entry["instances"]:
            mapped_video_ids.add(inst["video_id"])
            mapped_video_ids.add(str(int(inst["video_id"])))
            
    unmapped_videos = []
    for vf in video_files:
        name = vf.stem
        if name not in mapped_video_ids and str(int(name)) not in mapped_video_ids:
            unmapped_videos.append(vf.name)

    # Generate integrity report
    report_content = f"""# WLASL Dataset Integrity Report

## Summary
- **Total Source Videos**: {len(video_files)}
- **Total Metadata Entries (Instances)**: {mapped_count}
- **Successfully Organized Videos**: {copied_count}
- **Missing Videos (in metadata but not in video_data)**: {len(missing_videos)}
- **Corrupted/Invalid Videos**: {len(corrupted_videos)}
- **Unmapped Videos (in video_data but not in metadata)**: {len(unmapped_videos)}

## Detail: Corrupted Videos
{f"| Video ID | Gloss | Split | Error |" if corrupted_videos else "No corrupted videos found."}
{f"|---|---|---|---|" if corrupted_videos else ""}
"""
    for cv in corrupted_videos:
        report_content += f"| {cv[0]} | {cv[1]} | {cv[2]} | {cv[3]} |\n"
        
    report_content += f"\n## Detail: Unmapped Videos ({len(unmapped_videos)} total)\n"
    for uv in unmapped_videos[:50]: # limit to top 50
        report_content += f"- {uv}\n"
    if len(unmapped_videos) > 50:
        report_content += f"- ... and {len(unmapped_videos) - 50} more\n"
        
    report_path = project_root / "integrity_report.md"
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write(report_content)
        
    print(f"Dataset organization complete. Organized {copied_count} videos.")
    print(f"Integrity report written to: {report_path}")
    return report_path

if __name__ == "__main__":
    import sys
    proj_dir = Path(__file__).resolve().parents[2]
    organize_and_check(proj_dir)
