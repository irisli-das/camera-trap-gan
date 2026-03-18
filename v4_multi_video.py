"""
v4 Multi-Video Autoencoder Pipeline — Option A
================================================
Trains one dual model (raw + diff) per video independently.
Evaluates each video separately then prints a comparison table.
GPU-ready: uses CUDA if available, falls back to MPS/CPU.

Usage:
  python v4_multi_video.py

Outputs per video:
  outputs/v4_multi/{VIDEO_NAME}/
    model_A_raw_final.pth
    model_B_diff_final.pth
    loss_curve.png
    results.png
    weight_sweep.png
    suspicious_frames/
    config.txt

Final output:
  outputs/v4_multi/comparison_table.csv
  outputs/v4_multi/comparison_plot.png
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (roc_auc_score, precision_recall_fscore_support,
                             confusion_matrix, average_precision_score)
from tqdm import tqdm
from datetime import datetime
import cv2

# ══════════════════════════════════════════════════════════════
# CONFIG — update paths for RenkuLab
# ══════════════════════════════════════════════════════════════
# Base folder containing one subfolder per video
# Each subfolder should be named exactly as the video appears in the CSV
# e.g. data/frames/FH102_02/frame_00000.jpg

FRAMES_BASE    = "/home/renku/work/camera-trap-gan/data/frames"
CSV_PATH       = "/home/renku/work/camera-trap-gan/data/Weinstein2018MEE_ground_truth.csv"
OUTPUT_BASE    = "/home/renku/work/camera-trap-gan/outputs/v4_multi"
TIMESTAMP      = datetime.now().strftime("%Y%m%d_%H%M%S")

# Model hyperparameters
IMG_SIZE       = 256
BATCH_SIZE     = 64      # increased for GPU
EPOCHS         = 30      # full training with GPU
LEARNING_RATE  = 0.001
LATENT_DIM     = 256
THRESHOLD_STD  = 2
SMOOTH_WINDOW  = 2
WEIGHT_RAW     = 0.6     # from weight sweep results — raw model better
WEIGHT_DIFF    = 0.4

# Videos to process — must match subfolder names AND CSV Video column
VIDEOS = [
    "FH102_02",
    "FH102_03",
    "FH102_04",
    "FH102_05",
    "FH102_06",
    "FH102_07",
    "FH102_08",
]
# ══════════════════════════════════════════════════════════════


# ── Dataset ───────────────────────────────────────────────────
class CameraTrapDataset(Dataset):
    def __init__(self, frames_folder, img_size=256, use_diff=False):
        self.use_diff    = use_diff
        self.frame_paths = sorted([
            os.path.join(frames_folder, f)
            for f in os.listdir(frames_folder)
            if f.endswith('.jpg')
        ])
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        mode = "diff" if use_diff else "raw"
        print(f"  Dataset: {len(self.frame_paths)} frames  [{mode} mode]")

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        if self.use_diff and idx > 0:
            curr = cv2.cvtColor(
                cv2.imread(self.frame_paths[idx]),   cv2.COLOR_BGR2RGB)
            prev = cv2.cvtColor(
                cv2.imread(self.frame_paths[idx-1]), cv2.COLOR_BGR2RGB)
            img  = cv2.absdiff(curr, prev)
        else:
            img = cv2.cvtColor(
                cv2.imread(self.frame_paths[idx]), cv2.COLOR_BGR2RGB)
        return self.transform(img), self.frame_paths[idx]


# ── Model ─────────────────────────────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64,  4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512 * 16 * 16, latent_dim)
        )
        self.decoder_fc = nn.Linear(latent_dim, 512 * 16 * 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64,  4, 2, 1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.ConvTranspose2d(64,  3,   4, 2, 1), nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(self.decoder_fc(z).view(-1, 512, 16, 16))

    def anomaly_score(self, x):
        return torch.mean((x - self.forward(x)) ** 2, dim=[1, 2, 3])


# ── Helpers ───────────────────────────────────────────────────
def train_model(dataset, device, name, output_folder):
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE,
                           shuffle=True, num_workers=4,
                           pin_memory=True)
    model     = Autoencoder(LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    losses    = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for frames, _ in tqdm(loader,
                               desc=f"    Epoch {epoch+1}/{EPOCHS}",
                               leave=False):
            frames = frames.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(frames), frames)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(loader)
        losses.append(epoch_loss)
        print(f"    Epoch [{epoch+1}/{EPOCHS}]  Loss: {epoch_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(),
                       os.path.join(output_folder,
                                    f"{name}_epoch{epoch+1}.pth"))

    torch.save(model.state_dict(),
               os.path.join(output_folder, f"{name}_final.pth"))
    return model, losses


def score_frames(model, dataset, device):
    loader     = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4,
                            pin_memory=True)
    all_scores = []
    all_paths  = []
    model.eval()
    with torch.no_grad():
        for frames, paths in tqdm(loader, desc="    Scoring", leave=False):
            frames = frames.to(device, non_blocking=True)
            all_scores.extend(model.anomaly_score(frames).cpu().numpy())
            all_paths.extend(paths)
    return np.array(all_scores), all_paths


def smooth_max(scores, window=2):
    smoothed = np.zeros_like(scores)
    for i in range(len(scores)):
        smoothed[i] = np.max(
            scores[max(0, i-window):min(len(scores), i+window+1)])
    return smoothed


def normalise(scores):
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)


def load_ground_truth(csv_path, video_name):
    df       = pd.read_csv(csv_path)
    df_video = df[df['Video'] == video_name]
    positive = set(f"frame_{(r['Frame']-1):05d}.jpg"
                   for _, r in df_video.iterrows() if r['Truth'] == 'Positive')
    negative = set(f"frame_{(r['Frame']-1):05d}.jpg"
                   for _, r in df_video.iterrows() if r['Truth'] == 'Negative')
    return positive, negative


def evaluate(scores, all_paths, positive_frames, negative_frames, video_name):
    threshold = scores.mean() + THRESHOLD_STD * scores.std()

    labelled_indices   = []
    unlabelled_indices = []
    y_true, y_pred     = [], []

    for i, (score, path) in enumerate(zip(scores, all_paths)):
        fn         = os.path.basename(path)
        is_flagged = 1 if score > threshold else 0
        if fn in positive_frames:
            labelled_indices.append(i); y_true.append(1); y_pred.append(is_flagged)
        elif fn in negative_frames:
            labelled_indices.append(i); y_true.append(0); y_pred.append(is_flagged)
        else:
            unlabelled_indices.append(i)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.sum() == 0:
        print(f"  WARNING: no bird frames found in labelled set for {video_name}")
        return None

    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0)
    auc = roc_auc_score(y_true, [scores[i] for i in labelled_indices])
    ap  = average_precision_score(y_true, [scores[i] for i in labelled_indices])

    total_flagged      = int((scores > threshold).sum())
    total_flag_rate    = total_flagged / len(all_paths)
    labelled_flag_rate = y_pred.sum() / len(y_true)
    birds_found        = int(y_true[y_pred == 1].sum())
    workload_reduction = 1 - total_flag_rate

    return {
        'video': video_name,
        'total_frames': len(all_paths),
        'bird_frames': int(y_true.sum()),
        'labelled_frames': len(y_true),
        'auc': auc, 'ap': ap,
        'precision': p, 'recall': r, 'f1': f,
        'total_flagged': total_flagged,
        'total_flag_rate': total_flag_rate,
        'labelled_flag_rate': labelled_flag_rate,
        'bias_ratio': labelled_flag_rate / total_flag_rate if total_flag_rate > 0 else 0,
        'workload_reduction': workload_reduction,
        'birds_found': birds_found,
        'threshold': threshold,
        'y_true': y_true, 'y_pred': y_pred,
        'labelled_indices': labelled_indices,
    }


def print_video_results(r):
    print(f"\n  {'='*58}")
    print(f"  RESULTS — {r['video']}")
    print(f"  {'='*58}")
    print(f"  Total frames        : {r['total_frames']:>7,}")
    print(f"  Bird frames         : {r['bird_frames']:>7,}")
    print(f"  Labelled frames     : {r['labelled_frames']:>7,}  "
          f"({r['labelled_frames']/r['total_frames']:.1%} of total)")
    print(f"")
    print(f"  ── ALL frames ──────────────────────────────────")
    print(f"  Total flagged       : {r['total_flagged']:>7,}  "
          f"({r['total_flag_rate']:.1%} of all frames)")
    print(f"  Workload reduction  : {r['workload_reduction']:.1%}")
    print(f"  Birds found         : {r['birds_found']} / {r['bird_frames']}  "
          f"(recall {r['recall']:.1%})")
    print(f"")
    print(f"  ── LABELLED frames (metrics) ───────────────────")
    print(f"  AUC-ROC             : {r['auc']:>7.3f}  ← main metric")
    print(f"  Avg Precision       : {r['ap']:>7.3f}")
    print(f"  Precision           : {r['precision']:>7.3f}")
    print(f"  Recall              : {r['recall']:>7.3f}")
    print(f"  F1 score            : {r['f1']:>7.3f}")
    print(f"  Threshold           : {r['threshold']:>7.4f}")
    print(f"  Bias ratio          : {r['bias_ratio']:>7.1f}x")
    cm = confusion_matrix(r['y_true'], r['y_pred'])
    print(f"\n  Confusion matrix:")
    print(f"                    Predicted")
    print(f"                    Normal    Bird")
    print(f"  Actual Normal   {cm[0][0]:>8,}  {cm[0][1]:>6,}")
    print(f"  Actual Bird     {cm[1][0]:>8,}  {cm[1][1]:>6,}")
    print(f"  {'='*58}")


def plot_video_results(scores, all_paths, positive_frames, results,
                       output_folder, video_name):
    bird_indices = [i for i, p in enumerate(all_paths)
                    if os.path.basename(p) in positive_frames]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
    fig.suptitle(f"Anomaly detection — {video_name}  |  "
                 f"AUC={results['auc']:.3f}  Recall={results['recall']:.3f}",
                 fontsize=13)

    # scores over time
    ax1.plot(scores, linewidth=0.5, color='steelblue', alpha=0.8)
    ax1.axhline(y=results['threshold'], color='red', linestyle='--',
                linewidth=1, label=f"threshold ({results['threshold']:.4f})")
    for idx in bird_indices:
        ax1.axvline(x=idx, color='green', alpha=0.25, linewidth=0.5)
    ax1.plot([], [], color='green', alpha=0.5,
             linewidth=2, label='ground truth bird')
    ax1.set_ylabel("Anomaly score")
    ax1.set_title("Scores over time")
    ax1.legend(fontsize=8)

    # histogram
    y_true          = results['y_true']
    scores_labelled = scores[results['labelled_indices']]
    ax2.hist(scores_labelled[y_true == 0], bins=40, alpha=0.6,
             color='steelblue', label='normal', density=True)
    ax2.hist(scores_labelled[y_true == 1], bins=20, alpha=0.6,
             color='green', label='bird', density=True)
    ax2.axvline(x=results['threshold'], color='red', linestyle='--',
                linewidth=1.5)
    ax2.set_xlabel("Anomaly score")
    ax2.set_ylabel("Density")
    ax2.set_title("Score distribution — bird vs normal")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "results.png"), dpi=120)
    plt.close()


def plot_comparison(all_results, output_folder):
    """Bar chart comparing all videos side by side."""
    videos = [r['video'] for r in all_results]
    aucs   = [r['auc']     for r in all_results]
    recalls= [r['recall']  for r in all_results]
    precs  = [r['precision'] for r in all_results]
    birds  = [r['birds_found'] for r in all_results]
    total  = [r['bird_frames'] for r in all_results]
    wl     = [r['workload_reduction'] for r in all_results]

    x = np.arange(len(videos))
    w = 0.25

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle("Multi-video comparison — v4 dual model (per-video training)",
                 fontsize=13)

    # AUC / Recall / Precision
    axes[0].bar(x - w, aucs,   w, label='AUC-ROC',   color='steelblue', alpha=0.8)
    axes[0].bar(x,     recalls,w, label='Recall',     color='green',     alpha=0.8)
    axes[0].bar(x + w, precs,  w, label='Precision',  color='orange',    alpha=0.8)
    for i, (a, r, p) in enumerate(zip(aucs, recalls, precs)):
        axes[0].text(i-w, a+0.01, f"{a:.2f}", ha='center', fontsize=8)
        axes[0].text(i,   r+0.01, f"{r:.2f}", ha='center', fontsize=8)
        axes[0].text(i+w, p+0.01, f"{p:.2f}", ha='center', fontsize=8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(videos)
    axes[0].set_ylabel("Score"); axes[0].set_ylim(0, 1.1)
    axes[0].set_title("AUC / Recall / Precision per video")
    axes[0].legend(); axes[0].grid(True, alpha=0.3, axis='y')

    # Birds found
    axes[1].bar(x, total, label='Total birds', color='lightcoral', alpha=0.8)
    axes[1].bar(x, birds, label='Birds found', color='green',      alpha=0.8)
    for i, (b, t) in enumerate(zip(birds, total)):
        axes[1].text(i, t+0.1, f"{b}/{t}", ha='center', fontsize=9, fontweight='bold')
    axes[1].set_xticks(x); axes[1].set_xticklabels(videos)
    axes[1].set_ylabel("Frames")
    axes[1].set_title("Birds found vs total per video")
    axes[1].legend(); axes[1].grid(True, alpha=0.3, axis='y')

    # Workload reduction
    axes[2].bar(x, [w*100 for w in wl], color='steelblue', alpha=0.8)
    for i, w in enumerate(wl):
        axes[2].text(i, w*100+0.3, f"{w:.1%}", ha='center', fontsize=9)
    axes[2].set_xticks(x); axes[2].set_xticklabels(videos)
    axes[2].set_ylabel("Workload reduction (%)")
    axes[2].set_title("Manual review workload reduction per video")
    axes[2].set_ylim(0, 105)
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "comparison_plot.png"), dpi=150)
    plt.close()
    print(f"\nComparison plot saved: {output_folder}/comparison_plot.png")


def print_final_comparison(all_results):
    print(f"\n{'='*90}")
    print(f"  FINAL COMPARISON — ALL VIDEOS")
    print(f"{'='*90}")
    print(f"  {'Video':<12} {'Frames':>8} {'Birds':>6} {'AUC':>7} "
          f"{'AP':>7} {'Prec':>7} {'Recall':>8} {'F1':>7} "
          f"{'Found':>7} {'Workload↓':>10}")
    print(f"  {'-'*84}")

    aucs = []
    for r in all_results:
        aucs.append(r['auc'])
        print(f"  {r['video']:<12} {r['total_frames']:>8,} "
              f"{r['bird_frames']:>6} {r['auc']:>7.3f} "
              f"{r['ap']:>7.3f} {r['precision']:>7.3f} "
              f"{r['recall']:>8.3f} {r['f1']:>7.3f} "
              f"{r['birds_found']:>3}/{r['bird_frames']:<3} "
              f"{r['workload_reduction']:>9.1%}")

    print(f"  {'-'*84}")
    avg_auc    = np.mean([r['auc']    for r in all_results])
    avg_recall = np.mean([r['recall'] for r in all_results])
    avg_prec   = np.mean([r['precision'] for r in all_results])
    avg_f1     = np.mean([r['f1']     for r in all_results])
    avg_wl     = np.mean([r['workload_reduction'] for r in all_results])
    total_birds_found = sum(r['birds_found'] for r in all_results)
    total_birds       = sum(r['bird_frames'] for r in all_results)

    print(f"  {'AVERAGE':<12} {'':>8} {'':>6} {avg_auc:>7.3f} "
          f"{'':>7} {avg_prec:>7.3f} {avg_recall:>8.3f} {avg_f1:>7.3f} "
          f"{total_birds_found:>3}/{total_birds:<3} {avg_wl:>9.1%}")
    print(f"{'='*90}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # load CSV once
    df_gt = pd.read_csv(CSV_PATH)
    print(f"\nGround truth loaded: {len(df_gt)} labelled frames")
    print(f"Videos in CSV: {sorted(df_gt['Video'].unique())}")

    all_results = []

    # ── loop over each video ───────────────────────────────────
    for video_idx, video_name in enumerate(VIDEOS):
        print(f"\n{'#'*65}")
        print(f"  VIDEO {video_idx+1}/{len(VIDEOS)}: {video_name}")
        print(f"{'#'*65}")

        frames_folder = os.path.join(FRAMES_BASE, video_name)
        output_folder = os.path.join(OUTPUT_BASE, video_name)
        os.makedirs(output_folder, exist_ok=True)

        # check frames folder exists
        if not os.path.exists(frames_folder):
            print(f"  WARNING: frames folder not found: {frames_folder}")
            print(f"  Skipping {video_name}")
            continue

        # load ground truth for this video
        positive_frames, negative_frames = load_ground_truth(CSV_PATH, video_name)
        if not positive_frames:
            print(f"  WARNING: no positive frames in CSV for {video_name}, skipping")
            continue

        # save config
        with open(os.path.join(output_folder, "config.txt"), "w") as f:
            f.write(f"""
v4 Multi-Video Pipeline — {video_name}
======================================
Timestamp     : {TIMESTAMP}
Video         : {video_name}
Frames folder : {frames_folder}
Latent dim    : {LATENT_DIM}
Epochs        : {EPOCHS}
Batch size    : {BATCH_SIZE}
Learning rate : {LEARNING_RATE}
Threshold std : {THRESHOLD_STD}
Smooth window : {SMOOTH_WINDOW}
Weight raw    : {WEIGHT_RAW}
Weight diff   : {WEIGHT_DIFF}
Device        : {device}
""")

        # ── Step 1: datasets ──────────────────────────────────
        print(f"\n  Step 1 — Loading datasets")
        dataset_raw  = CameraTrapDataset(frames_folder, IMG_SIZE, use_diff=False)
        dataset_diff = CameraTrapDataset(frames_folder, IMG_SIZE, use_diff=True)

        # ── Step 2: train Model A (raw) ───────────────────────
        print(f"\n  Step 2 — Training Model A (raw frames)")
        model_raw, losses_raw = train_model(
            dataset_raw, device, "model_A_raw", output_folder)

        # ── Step 3: train Model B (diff) ─────────────────────
        print(f"\n  Step 3 — Training Model B (diff frames)")
        model_diff, losses_diff = train_model(
            dataset_diff, device, "model_B_diff", output_folder)

        # plot loss curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(losses_raw,  linewidth=2); ax1.set_title(f"Model A loss — {video_name}")
        ax2.plot(losses_diff, linewidth=2); ax2.set_title(f"Model B loss — {video_name}")
        for ax in [ax1, ax2]:
            ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "loss_curves.png"), dpi=120)
        plt.close()

        # ── Step 4: score all frames ──────────────────────────
        print(f"\n  Step 4 — Scoring frames")
        scores_raw,  all_paths = score_frames(model_raw,  dataset_raw,  device)
        scores_diff, _         = score_frames(model_diff, dataset_diff, device)

        # smooth + normalise + combine
        scores_diff_smooth = smooth_max(scores_diff, SMOOTH_WINDOW)
        scores_raw_norm    = normalise(scores_raw)
        scores_diff_norm   = normalise(scores_diff_smooth)
        scores_combined    = WEIGHT_RAW * scores_raw_norm + WEIGHT_DIFF * scores_diff_norm

        print(f"  Raw   — mean: {scores_raw.mean():.4f}  std: {scores_raw.std():.4f}")
        print(f"  Diff  — mean: {scores_diff.mean():.4f}  std: {scores_diff.std():.4f}")
        print(f"  Combined — mean: {scores_combined.mean():.4f}  std: {scores_combined.std():.4f}")

        # ── Step 5: evaluate ──────────────────────────────────
        print(f"\n  Step 5 — Evaluating")
        results = evaluate(scores_combined, all_paths,
                           positive_frames, negative_frames, video_name)
        if results is None:
            continue

        print_video_results(results)
        all_results.append(results)

        # ── Step 6: plot ──────────────────────────────────────
        plot_video_results(scores_combined, all_paths, positive_frames,
                           results, output_folder, video_name)

        # ── Step 7: save suspicious frames ───────────────────
        print(f"\n  Step 7 — Saving suspicious frames")
        susp_folder = os.path.join(output_folder, "suspicious_frames")
        os.makedirs(susp_folder, exist_ok=True)
        flagged = sorted(
            [(s, p) for s, p in zip(scores_combined, all_paths)
             if s > results['threshold']],
            reverse=True
        )
        for i, (score, path) in enumerate(flagged):
            fn      = os.path.basename(path)
            is_bird = "BIRD" if fn in positive_frames else "normal"
            shutil.copy(path, os.path.join(
                susp_folder,
                f"rank{i+1:04d}_{is_bird}_{score:.4f}_{fn}"))
        print(f"  {len(flagged)} frames saved")

        # free GPU memory between videos
        del model_raw, model_diff
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════
    # FINAL COMPARISON
    # ══════════════════════════════════════════════════════════
    if not all_results:
        print("No results to compare — check frame folders and CSV paths")
        return

    print_final_comparison(all_results)
    plot_comparison(all_results, OUTPUT_BASE)

    # save CSV
    df_results = pd.DataFrame([{
        'video':              r['video'],
        'total_frames':       r['total_frames'],
        'bird_frames':        r['bird_frames'],
        'labelled_frames':    r['labelled_frames'],
        'auc':                round(r['auc'], 3),
        'avg_precision':      round(r['ap'], 3),
        'precision':          round(r['precision'], 3),
        'recall':             round(r['recall'], 3),
        'f1':                 round(r['f1'], 3),
        'birds_found':        r['birds_found'],
        'total_flagged':      r['total_flagged'],
        'flag_rate':          round(r['total_flag_rate'], 3),
        'workload_reduction': round(r['workload_reduction'], 3),
        'bias_ratio':         round(r['bias_ratio'], 1),
        'threshold':          round(r['threshold'], 4),
        'epochs':             EPOCHS,
        'latent_dim':         LATENT_DIM,
        'weight_raw':         WEIGHT_RAW,
        'weight_diff':        WEIGHT_DIFF,
    } for r in all_results])

    csv_path = os.path.join(OUTPUT_BASE, f"comparison_{TIMESTAMP}.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nResults CSV saved: {csv_path}")
    print(f"All outputs in:    {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
