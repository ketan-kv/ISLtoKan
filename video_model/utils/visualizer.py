import sys
from pathlib import Path
import cv2
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from video_model.preprocessing.video_extractor import MediaPipeVideoExtractor

def create_sample_visualization(project_root: Path):
    dataset_dir = project_root / "video_dataset"
    runs_dir = project_root / "video_model" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    # Find the first available video file
    video_paths = list(dataset_dir.glob("**/*.mp4"))
    if not video_paths:
        print("[ERROR] No videos found in video_dataset/. Please organize dataset first.")
        return
        
    sample_video = video_paths[0]
    output_video = runs_dir / "sample_visualization.mp4"
    
    print(f"Loading sample video: {sample_video.name}")
    print("Initializing MediaPipe Holistic Extractor...")
    extractor = MediaPipeVideoExtractor(project_root)
    
    cap = cv2.VideoCapture(str(sample_video))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {sample_video}")
        extractor.close()
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    
    print(f"Processing and visualizing {frame_count} frames...")
    for _ in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Extract features (internally caches landmarks for drawing)
        _ = extractor.extract_frame(frame)
        
        # Overlay skeletal structures on frame
        extractor.draw_overlays(frame, show_pose=True, show_hands=True, show_face=True)
        
        # Write frame to output video
        out.write(frame)
        
    cap.release()
    out.release()
    extractor.close()
    
    print(f"Sample visualization video saved successfully at: {output_video}")

if __name__ == "__main__":
    proj_dir = Path(__file__).resolve().parents[2]
    create_sample_visualization(proj_dir)
