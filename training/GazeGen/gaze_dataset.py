
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn as nn 
from transformers import (
    VivitImageProcessor,
    VivitForVideoClassification,
    VivitModel,
    TrainingArguments,
    Trainer,
)
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
import cv2 as cv
import matplotlib.pyplot as plt

from tqdm import tqdm
import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class GazeVideoDataset(Dataset):
    def __init__(self, metadata_path, frames=32, prediction_size=10, transform=None, gaze_only=False):
        self.metadata_path = metadata_path
        self.video_paths, self.gaze_data_dict = self.get_paths(metadata_path)
        self.frames = frames
        self.prediction_size = prediction_size
        self.chunk_length = self.frames + self.prediction_size
        self.transform = transform
        self.gaze_only = gaze_only

        # Dictionaries to store cleaned gaze data and valid frame indices.
        self.gaze_data = {}
        self.valid_indices = {}  # Will hold the original frame numbers with valid gaze data

        with tqdm(total=len(self.video_paths), desc="Processing gaze data") as pbar:
            for class_name, gaze_path in self.gaze_data_dict.items():
                
                if os.path.exists(gaze_path):
                    # Read CSV, select "x" and "y" columns.
                    gaze_df = pd.read_csv(gaze_path)[["x", "y"]]
                    # Create a mask for rows without NaN values.
                    valid_mask = gaze_df.notnull().all(axis=1)
                    # Record the original indices of the valid rows.
                    valid_idx = gaze_df[valid_mask].index.to_numpy()
                    # Drop NaNs and reset index so the remaining rows are contiguous.
                    gaze_df = gaze_df[valid_mask].reset_index(drop=True)
                    self.gaze_data[class_name] = gaze_df.to_numpy()
                    self.valid_indices[class_name] = valid_idx
                else:
                    raise FileNotFoundError(f"Gaze data file not found: {gaze_path}")
                pbar.update(1)
        # Precompute valid chunks for each video based on cleaned gaze data.
        self.chunk_indices = []
        for vid_idx, video_path in enumerate(self.video_paths):
            class_name = video_path.split('/')[-2]
            if class_name in self.gaze_data:
                total_valid_frames = self.gaze_data[class_name].shape[0]
                # Ensure we have enough contiguous valid frames.
                for start in range(0, total_valid_frames - self.chunk_length + 1):
                    self.chunk_indices.append((vid_idx, start))
    
    def __len__(self):
        return len(self.chunk_indices)
    
    def __getitem__(self, idx):
        # Retrieve video index and the start index (in cleaned gaze data) for the chunk.
        vid_idx, start_idx = self.chunk_indices[idx]
        video_path = self.video_paths[vid_idx]
        class_name = video_path.split('/')[-2]

        # Get the corresponding gaze data and the valid original frame indices.
        gaze_array = self.gaze_data[class_name]
        valid_idx_array = self.valid_indices[class_name]
        
        # Determine the original video frame numbers for this chunk.
        chunk_valid_frame_indices = valid_idx_array[start_idx : start_idx + self.chunk_length]
        
        if not self.gaze_only:
            frames = []
            cap = cv2.VideoCapture(video_path)
            for frame_index in chunk_valid_frame_indices:
                # Jump to the desired frame in the video.
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()
            # Convert frames to tensor: (chunk_length, C, H, W)
            frames = np.stack(frames)
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        else:
            frames = []
            
        # Extract the corresponding gaze chunk from the cleaned array.
        gaze_chunk = gaze_array[start_idx : start_idx + self.chunk_length]
        gaze_chunk = torch.from_numpy(gaze_chunk).float()
        
        sample = {'frames': frames, 'gaze': gaze_chunk}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_paths(self, metadata_path):
        print("Getting paths...")
        metadata = pd.read_csv(os.path.join(metadata_path, "metadata.csv"))
        video_paths = []
        gaze_paths = {}
        for i in range(metadata.shape[0]):
            video_paths.append(os.path.join(metadata_path, metadata["Label"][i],
                                            metadata["Class Name"][i], metadata["Original Name"][i]))
            gaze_paths[metadata["Class Name"][i]] = os.path.join(metadata_path, metadata["Label"][i],
                                                                  metadata["Class Name"][i],
                                                                  "general_eye_gaze_2d.csv")
        return video_paths, gaze_paths

def custom_collate_fn(batch):
    """
    Collates a list of samples into a batch.
    
    Each sample is a dictionary with:
        'frames': Tensor of shape [chunk_length, C, H, W]
        'gaze': Tensor of shape [chunk_length, 2]
    
    The collate function stacks these into batched tensors.
    """
    # Stack frames: resulting shape [batch_size, chunk_length, C, H, W]
    frames = torch.stack([sample['frames'] for sample in batch])
    # Stack gaze: resulting shape [batch_size, chunk_length, 2]
    gaze = torch.stack([sample['gaze'] for sample in batch])
    
    return {'frames': frames, 'gaze': gaze}

