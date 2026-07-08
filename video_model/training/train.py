import os
import json
import csv
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Import modular sequence models
import sys
# Add parent directory of video_model to path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from video_model.models.sequence_model import SequenceClassifier


class LandmarkDataset(Dataset):
    def __init__(self, labels_csv: Path, vocab_json: Path, split: str, project_root: Path):
        self.df = pd.read_csv(labels_csv)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.split = split

        with open(vocab_json, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        self.project_root = project_root
        
        # Pre-load all landmark sequences into RAM to avoid slow disk I/O bottlenecks
        print(f"Pre-loading {len(self.df)} {split} sequences into RAM...")
        self.cached_landmarks = []
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            npy_path = self.project_root / row['filepath']
            self.cached_landmarks.append(np.load(npy_path))
        print(f"Pre-loaded {split} split successfully.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        landmarks = self.cached_landmarks[idx].copy()  # Use copy to avoid mutating cached source

        # Apply sequence-level data augmentation during training
        if self.split == "train":
            seq_len = len(landmarks)

            # 1. Random Frame Drop (Temporal scaling)
            if seq_len > 10 and np.random.rand() > 0.5:
                # Randomly drop 10% of frames
                keep_prob = 0.9
                keep_mask = np.random.rand(seq_len) < keep_prob
                if np.sum(keep_mask) > 5:
                    landmarks = landmarks[keep_mask]

            # 2. Random Scaling
            if np.random.rand() > 0.5:
                scale = np.random.uniform(0.9, 1.1)
                landmarks = landmarks * scale

            # 3. Random Translation
            if np.random.rand() > 0.5:
                translation = np.random.uniform(-0.03, 0.03, size=landmarks.shape)
                landmarks = landmarks + translation

            # 4. Add Gaussian Noise
            if np.random.rand() > 0.5:
                noise = np.random.normal(0, 0.005, size=landmarks.shape)
                landmarks = landmarks + noise

            # 5. Random Rotation around z-axis (simulates slight camera roll)
            if np.random.rand() > 0.5:
                theta = np.random.uniform(-0.1, 0.1)  # angle in radians (~5.7 degrees)
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                rot_matrix = np.array([
                    [cos_t, -sin_t, 0],
                    [sin_t, cos_t, 0],
                    [0, 0, 1]
                ], dtype=np.float32)

                # Reshape from [seq_len, 345] to [seq_len, 115, 3] to apply 3D rotation
                seq_len = len(landmarks)
                reshaped = landmarks.reshape(seq_len, 115, 3)
                rotated = np.dot(reshaped, rot_matrix)
                landmarks = rotated.reshape(seq_len, 345)

        label_name = row['gloss']
        label_idx = self.vocab[label_name]

        return torch.tensor(landmarks, dtype=torch.float32), torch.tensor(label_idx, dtype=torch.long)


def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    padded_seqs = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)
    labels = torch.stack(labels)
    return padded_seqs, lengths, labels


def train_model(
    project_root: Path,
    encoder_type: str = "lstm",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 10,
    resume_path: str = None
):
    # Paths
    landmarks_dir = project_root / "video_model" / "landmarks"
    labels_csv = landmarks_dir / "labels.csv"
    vocab_json = landmarks_dir / "vocabulary.json"
    models_dir = project_root / "video_model" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    log_dir = project_root / "video_model" / "runs" / f"{encoder_type}_run"
    writer = SummaryWriter(log_dir=str(log_dir))

    # Load vocabulary
    with open(vocab_json, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    num_classes = len(vocab)
    print(f"Dataset Vocabulary: {num_classes} unique classes.")

    # Create Datasets and Dataloaders
    train_dataset = LandmarkDataset(labels_csv, vocab_json, "train", project_root)
    val_dataset = LandmarkDataset(labels_csv, vocab_json, "validation", project_root)

    # Use num_workers=0 to avoid multiprocessing lock issues in Windows
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    print(f"Data loaders created. Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model
    model = SequenceClassifier(
        encoder_type=encoder_type,
        num_classes=num_classes,
        input_size=345,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        bidirectional=True
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    start_epoch = 0
    best_val_loss = float('inf')
    best_val_acc = 0.0
    early_stop_counter = 0

    # Resume support
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming training from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resuming at epoch {start_epoch} with best validation loss {best_val_loss:.4f}")

    for epoch in range(start_epoch, epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_seqs, batch_lengths, batch_labels in train_loader:
            batch_seqs, batch_labels = batch_seqs.to(device), batch_labels.to(device)
            # lengths stays on CPU for packed sequence

            optimizer.zero_grad()
            outputs = model(batch_seqs, batch_lengths)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_seqs.size(0)
            _, predicted = outputs.max(1)
            train_total += batch_labels.size(0)
            train_correct += predicted.eq(batch_labels).sum().item()

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_seqs, batch_lengths, batch_labels in val_loader:
                batch_seqs, batch_labels = batch_seqs.to(device), batch_labels.to(device)

                outputs = model(batch_seqs, batch_lengths)
                loss = criterion(outputs, batch_labels)

                val_loss += loss.item() * batch_seqs.size(0)
                _, predicted = outputs.max(1)
                val_total += batch_labels.size(0)
                val_correct += predicted.eq(batch_labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        # LR Scheduling
        scheduler.step(epoch_val_loss)

        # Log to TensorBoard
        writer.add_scalar("Loss/Train", epoch_train_loss, epoch)
        writer.add_scalar("Loss/Val", epoch_val_loss, epoch)
        writer.add_scalar("Accuracy/Train", epoch_train_acc, epoch)
        writer.add_scalar("Accuracy/Val", epoch_val_acc, epoch)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc*100:.2f}%")

        # Save Best Validation Checkpoint
        is_best = epoch_val_loss < best_val_loss
        if is_best:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            early_stop_counter = 0

            # Save best model weight
            best_model_path = models_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
                'encoder_type': encoder_type
            }, best_model_path)
            print(f"--> Saved best model checkpoint to {best_model_path}")
        else:
            early_stop_counter += 1

        # Periodic checkpoint for resume support
        periodic_checkpoint_path = models_dir / "latest_checkpoint.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
            'encoder_type': encoder_type
        }, periodic_checkpoint_path)

        # Early Stopping
        if early_stop_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1} due to no validation loss improvement.")
            break

    # Load best weights to generate final metrics
    best_model_path = models_dir / "best_model.pth"
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded best weights for final evaluation.")

    # Generate Confusion Matrix
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_seqs, batch_lengths, batch_labels in val_loader:
            batch_seqs = batch_seqs.to(device)
            outputs = model(batch_seqs, batch_lengths)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    np.save(models_dir / "confusion_matrix.npy", cm)
    print(f"Confusion matrix saved to {models_dir / 'confusion_matrix.npy'}")

    writer.close()
    print("Training pipeline finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train WLASL sequence model.")
    parser.add_argument("--encoder", type=str, default="lstm", choices=["lstm", "transformer"], help="Type of model encoder")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")
    args = parser.parse_args()

    proj_dir = Path(__file__).resolve().parents[2]
    train_model(
        project_root=proj_dir,
        encoder_type=args.encoder,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        resume_path=args.resume
    )
