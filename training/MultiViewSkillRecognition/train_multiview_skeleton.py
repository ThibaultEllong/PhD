#!/usr/bin/env python3
"""
Training script for multi-view skeleton-based skill classification.
"""
import os
import argparse
import logging
import contextlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from mmengine import Config
from mmaction.apis import init_recognizer, inference_skeleton, pose_inference, detection_inference

from egoexo_dataloader_2 import EgoExoDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a multi-view skeleton-based classifier"
    )
    # Dataset paths
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Base path to Ego4D dataset")
    parser.add_argument("--takes-json", type=str, required=True,
                        help="Path to takes.json annotation file")
    # Model config
    parser.add_argument("--pose-config", type=str, required=True,
                        help="Path to MMACTION skeleton config file")
    parser.add_argument("--pose-checkpoint", type=str, required=True,
                        help="Path to MMACTION skeleton checkpoint file")
    # Detection & pose inference for preprocessing
    parser.add_argument("--det-config", type=str, required=True,
                        help="Path to detection config for preprocessing")
    parser.add_argument("--det-checkpoint", type=str, required=True,
                        help="Path to detection checkpoint for preprocessing")
    parser.add_argument("--skel-config", type=str, required=True,
                        help="Path to pose config for preprocessing")
    parser.add_argument("--skel-checkpoint", type=str, required=True,
                        help="Path to pose checkpoint for preprocessing")
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Optimizer weight decay")
    parser.add_argument("--num-classes", type=int, default=5,
                        help="Number of skill classes")
    parser.add_argument("--seq-len", type=int, default=20,
                        help="Temporal sequence length for model")
    parser.add_argument("--feat-dim", type=int, default=512,
                        help="Skeleton feature dimension")
    parser.add_argument("--hidden-dim", type=int, default=512,
                        help="Hidden dimension")
    parser.add_argument("--num-views", type=int, default=4,
                        help="Number of views (ego+exo)")
    parser.add_argument("--frame-rate", type=int, default=3,
                        help="Frame sampling rate for dataset loader")
    parser.add_argument("--window-size", type=int, default=128,
                        help="Window size for skeleton extraction")
    parser.add_argument("--sampling-rate", type=int, default=3,
                        help="Sampling rate for skeleton extraction")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Compute device")
    return parser.parse_args()


def collate_objects(batch):
    labels = torch.tensor([s['skill'] for s in batch], dtype=torch.long)
    return batch, labels


def frame_windows(video_path: str, window_size: int, sampling_rate: int):
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")
    frames = []
    raw_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if raw_idx % sampling_rate == 0:
            frames.append(frame)
            if len(frames) == window_size:
                yield frames
                frames = []
        raw_idx += 1
    if frames:
        yield frames
    cap.release()


def extract_skeleton(video_path, window_size, sampling_rate, det_config,
                     det_checkpoint, skel_config, skel_checkpoint):
    from mmaction.apis import detection_inference, pose_inference
    skeleton_chunks = []
    total = 0
    for frames in frame_windows(video_path, window_size, sampling_rate):
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), \
             contextlib.redirect_stderr(devnull):
            det_res, _ = detection_inference(
                str(det_config), str(det_checkpoint), frames)
            pose_res, _ = pose_inference(
                str(skel_config), str(skel_checkpoint), frames, det_res)
            skeleton_chunks.append(np.array(pose_res))
            total += len(pose_res)
    if skeleton_chunks:
        return np.concatenate(skeleton_chunks, axis=0)
    return np.zeros((0,))


def process_skeletons(batch, skeleton_model, args):
    batch_views = []
    for item in batch:
        paths = item['samples']['exo'][:] + [item['samples']['ego']]
        views = []
        for p in paths:
            skel = extract_skeleton(
                p, args.window_size, args.sampling_rate,
                args.det_config, args.det_checkpoint,
                args.skel_config, args.skel_checkpoint)
            rep = inference_skeleton(
                skeleton_model, skel, (1920,1080), test_pipeline=None)
            views.append(torch.tensor(rep))
        batch_views.append(torch.stack(views))
    return batch_views


class MultiViewSkeletonClassifier(nn.Module):
    def __init__(self, feat_dim, seq_len, num_views,
                 hidden_dim, num_classes):
        super().__init__()
        self.attn_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim//2, 1)
        )
        self.view_weights = nn.Parameter(torch.ones(num_views))
        self.fusion_proj = nn.Linear(feat_dim, hidden_dim)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(hidden_dim)
        self.fc    = nn.Sequential(
            nn.Linear(hidden_dim * seq_len, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x: list of view tensors -> stack to (B,V,T,D)
        x = torch.stack(x).to(self.fc[0].weight.device)
        B, V, T, D = x.shape
        summary = x.mean(dim=2)
        raw_w   = self.attn_head(summary).squeeze(-1)
        w       = F.softmax(raw_w, dim=1)
        fused   = (w.view(B,V,1,1) * x).sum(dim=1)
        fused = F.relu(self.fusion_proj(fused))
        fused = fused.permute(0,2,1)
        out = F.relu(self.bn1(self.conv1(fused)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = out.reshape(B, -1)
        return self.fc(out)


def train_epoch(model, loader, criterion, optimizer, device, skeleton_model, args):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch, labels in loader:
        labels = labels.to(device)
        views = process_skeletons(batch, skeleton_model, args)
        outputs = model(views)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), 100*correct/total
#!/usr/bin/env python3
"""
Training script for multi-view skeleton-based skill classification.
"""
import os
import argparse
import logging
import contextlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from mmengine import Config
from mmaction.apis import init_recognizer, inference_skeleton, pose_inference, detection_inference

from egoexo_dataloader_2 import EgoExoDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a multi-view skeleton-based classifier"
    )
    # Dataset paths
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Base path to Ego4D dataset")
    parser.add_argument("--takes-json", type=str, required=True,
                        help="Path to takes.json annotation file")
    # Model config
    parser.add_argument("--pose-config", type=str, required=True,
                        help="Path to MMACTION skeleton config file")
    parser.add_argument("--pose-checkpoint", type=str, required=True,
                        help="Path to MMACTION skeleton checkpoint file")
    # Detection & pose inference for preprocessing
    parser.add_argument("--det-config", type=str, required=True,
                        help="Path to detection config for preprocessing")
    parser.add_argument("--det-checkpoint", type=str, required=True,
                        help="Path to detection checkpoint for preprocessing")
    parser.add_argument("--skel-config", type=str, required=True,
                        help="Path to pose config for preprocessing")
    parser.add_argument("--skel-checkpoint", type=str, required=True,
                        help="Path to pose checkpoint for preprocessing")
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Optimizer weight decay")
    parser.add_argument("--num-classes", type=int, default=5,
                        help="Number of skill classes")
    parser.add_argument("--seq-len", type=int, default=20,
                        help="Temporal sequence length for model")
    parser.add_argument("--feat-dim", type=int, default=512,
                        help="Skeleton feature dimension")
    parser.add_argument("--hidden-dim", type=int, default=512,
                        help="Hidden dimension")
    parser.add_argument("--num-views", type=int, default=4,
                        help="Number of views (ego+exo)")
    parser.add_argument("--frame-rate", type=int, default=3,
                        help="Frame sampling rate for dataset loader")
    parser.add_argument("--window-size", type=int, default=128,
                        help="Window size for skeleton extraction")
    parser.add_argument("--sampling-rate", type=int, default=3,
                        help="Sampling rate for skeleton extraction")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Compute device")
    return parser.parse_args()


def collate_objects(batch):
    labels = torch.tensor([s['skill'] for s in batch], dtype=torch.long)
    return batch, labels


def frame_windows(video_path: str, window_size: int, sampling_rate: int):
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")
    frames = []
    raw_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if raw_idx % sampling_rate == 0:
            frames.append(frame)
            if len(frames) == window_size:
                yield frames
                frames = []
        raw_idx += 1
    if frames:
        yield frames
    cap.release()


def extract_skeleton(video_path, window_size, sampling_rate, det_config,
                     det_checkpoint, skel_config, skel_checkpoint):
    from mmaction.apis import detection_inference, pose_inference
    skeleton_chunks = []
    total = 0
    for frames in frame_windows(video_path, window_size, sampling_rate):
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), \
             contextlib.redirect_stderr(devnull):
            det_res, _ = detection_inference(
                str(det_config), str(det_checkpoint), frames)
            pose_res, _ = pose_inference(
                str(skel_config), str(skel_checkpoint), frames, det_res)
            skeleton_chunks.append(np.array(pose_res))
            total += len(pose_res)
    if skeleton_chunks:
        return np.concatenate(skeleton_chunks, axis=0)
    return np.zeros((0,))


def process_skeletons(batch, skeleton_model, args):
    batch_views = []
    for item in batch:
        paths = item['samples']['exo'][:] + [item['samples']['ego']]
        views = []
        for p in paths:
            skel = extract_skeleton(
                p, args.window_size, args.sampling_rate,
                args.det_config, args.det_checkpoint,
                args.skel_config, args.skel_checkpoint)
            rep = inference_skeleton(
                skeleton_model, skel, (1920,1080), test_pipeline=None)
            views.append(torch.tensor(rep))
        batch_views.append(torch.stack(views))
    return batch_views


class MultiViewSkeletonClassifier(nn.Module):
    def __init__(self, feat_dim, seq_len, num_views,
                 hidden_dim, num_classes):
        super().__init__()
        self.attn_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim//2, 1)
        )
        self.view_weights = nn.Parameter(torch.ones(num_views))
        self.fusion_proj = nn.Linear(feat_dim, hidden_dim)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(hidden_dim)
        self.fc    = nn.Sequential(
            nn.Linear(hidden_dim * seq_len, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x: list of view tensors -> stack to (B,V,T,D)
        x = torch.stack(x).to(self.fc[0].weight.device)
        B, V, T, D = x.shape
        summary = x.mean(dim=2)
        raw_w   = self.attn_head(summary).squeeze(-1)
        w       = F.softmax(raw_w, dim=1)
        fused   = (w.view(B,V,1,1) * x).sum(dim=1)
        fused = F.relu(self.fusion_proj(fused))
        fused = fused.permute(0,2,1)
        out = F.relu(self.bn1(self.conv1(fused)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = out.reshape(B, -1)
        return self.fc(out)


def train_epoch(model, loader, criterion, optimizer, device, skeleton_model, args):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch, labels in loader:
        labels = labels.to(device)
        views = process_skeletons(batch, skeleton_model, args)
        outputs = model(views)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), 100*correct/total


def validate_epoch(model, loader, criterion, device, skeleton_model, args):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch, labels in loader:
            labels = labels.to(device)
            views = process_skeletons(batch, skeleton_model, args)
            outputs = model(views)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), 100*correct/total


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Init pose recognition model
    cfg = Config.fromfile(args.pose_config)
    skeleton_model = init_recognizer(
        cfg, args.pose_checkpoint, device=device)

    # Datasets and loaders
    train_ds = EgoExoDataset(
        args.dataset_path, args.takes_json, split='train', skill=True,
        get_frames=False, get_pose=True, get_hands_pose=False,
        frame_rate=args.frame_rate)
    val_ds   = EgoExoDataset(
        args.dataset_path, args.takes_json, split='val', skill=True,
        get_frames=False, get_pose=True, get_hands_pose=False,
        frame_rate=args.frame_rate)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_objects)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_objects)

    # Model, loss, optimizer, scheduler
    model = MultiViewSkeletonClassifier(
        feat_dim=args.feat_dim,
        seq_len=args.seq_len,
        num_views=args.num_views,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # Training loop
    for epoch in range(1, args.epochs+1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer,
            device, skeleton_model, args)
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device,
            skeleton_model, args)
        scheduler.step(val_loss)
        logging.info(
            f"Epoch {epoch}/{args.epochs} "
            f"Train: loss={train_loss:.4f}, acc={train_acc:.2f}% | "
            f"Val: loss={val_loss:.4f}, acc={val_acc:.2f}%"
        )
        torch.save(model.state_dict(),
                   f"multiview_model_epoch_{epoch}.pth")

if __name__ == '__main__':
    main()


def validate_epoch(model, loader, criterion, device, skeleton_model, args):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch, labels in loader:
            labels = labels.to(device)
            views = process_skeletons(batch, skeleton_model, args)
            outputs = model(views)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), 100*correct/total


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Init pose recognition model
    cfg = Config.fromfile(args.pose_config)
    skeleton_model = init_recognizer(
        cfg, args.pose_checkpoint, device=device)

    # Datasets and loaders
    train_ds = EgoExoDataset(
        args.dataset_path, args.takes_json, split='train', skill=True,
        get_frames=False, get_pose=True, get_hands_pose=False,
        frame_rate=args.frame_rate)
    val_ds   = EgoExoDataset(
        args.dataset_path, args.takes_json, split='val', skill=True,
        get_frames=False, get_pose=True, get_hands_pose=False,
        frame_rate=args.frame_rate)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_objects)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_objects)

    # Model, loss, optimizer, scheduler
    model = MultiViewSkeletonClassifier(
        feat_dim=args.feat_dim,
        seq_len=args.seq_len,
        num_views=args.num_views,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # Training loop
    for epoch in range(1, args.epochs+1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer,
            device, skeleton_model, args)
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device,
            skeleton_model, args)
        scheduler.step(val_loss)
        logging.info(
            f"Epoch {epoch}/{args.epochs} "
            f"Train: loss={train_loss:.4f}, acc={train_acc:.2f}% | "
            f"Val: loss={val_loss:.4f}, acc={val_acc:.2f}%"
        )
        torch.save(model.state_dict(),
                   f"multiview_model_epoch_{epoch}.pth")

if __name__ == '__main__':
    main()
