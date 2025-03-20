from pathlib import Path
import os
import torch
import numpy as np
import imageio
import pytorchvideo.data
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)
from transformers import (
    VivitImageProcessor,
    VivitForVideoClassification,
    TrainingArguments,
    Trainer,
)
import evaluate
from IPython.display import Image
import time

def count_videos(dataset_root_path):
    """Count the number of videos in the dataset."""
    video_count_train = len(list(dataset_root_path.glob("train/*/*.mp4")))
    video_count_val = len(list(dataset_root_path.glob("val/*/*.mp4")))
    video_count_test = len(list(dataset_root_path.glob("test/*/*.mp4")))
    return video_count_train, video_count_val, video_count_test

def get_class_labels(all_video_file_paths):
    """Extract unique class labels from video file paths."""
    class_labels = sorted({str(path).split("/")[-2] for path in all_video_file_paths})
    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label

def load_model(label2id, id2label, freeze_feature_extractor=True):
    """Load and configure the ViViT model for video classification.
    
    Args:
        label2id (dict): Mapping from class labels to IDs.
        id2label (dict): Mapping from IDs to class labels.
        freeze_feature_extractor (bool): Whether to freeze the base model parameters.
        
    Returns:
        model: The configured ViViT model ready for fine-tuning.
    """
    model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400").to("cuda")
    model.config.label2id = label2id
    model.config.id2label = id2label
    model.classifier = torch.nn.Linear(in_features=768, out_features=len(label2id), bias=True).to("cuda")
    model.num_labels = len(label2id)
    
    if freeze_feature_extractor:
        for name, param in model.named_parameters():
            # Only allow the classifier (and any added head layers) to be trained.
            if "classifier" not in name:
                param.requires_grad = False
    return model

def prepare_transforms(image_processor, num_frames_to_sample, resize_to):
    """Prepare data transformations for training and validation."""
    mean = image_processor.image_mean
    std = image_processor.image_std

    train_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        RandomShortSideScale(min_size=256, max_size=320),
                        RandomCrop(resize_to),
                        RandomHorizontalFlip(p=0.5),
                    ]
                ),
            ),
        ]
    )

    val_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        Resize(resize_to),
                    ]
                ),
            ),
        ]
    )

    return train_transform, val_transform

def create_datasets(dataset_root_path, train_transform, val_transform, clip_duration):
    """Create training and test datasets."""
    train_dataset = pytorchvideo.data.Ucf101(
        data_path=os.path.join(dataset_root_path, "train"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
        decode_audio=False,
        transform=train_transform,
    )

    test_dataset = pytorchvideo.data.Ucf101(
        data_path=os.path.join(dataset_root_path, "test"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
    )

    return train_dataset, test_dataset

def unnormalize_img(img, mean, std):
    """Un-normalizes the image pixels."""
    img = (img * std) + mean
    img = (img * 255).astype("uint8")
    return img.clip(0, 255)

def create_gif(video_tensor, filename="sample.gif", mean=None, std=None):
    """Prepares a GIF from a video tensor."""
    frames = []
    for video_frame in video_tensor:
        frame_unnormalized = unnormalize_img(video_frame.permute(1, 2, 0).numpy(), mean, std)
        frames.append(frame_unnormalized)
    kargs = {"duration": 0.25}
    imageio.mimsave(filename, frames, "GIF", **kargs)
    return filename

def display_gif(video_tensor, gif_name="sample.gif", mean=None, std=None):
    """Prepares and displays a GIF from a video tensor."""
    video_tensor = video_tensor.permute(1, 0, 2, 3)
    gif_filename = create_gif(video_tensor, gif_name, mean, std)
    return Image(filename=gif_filename)

def collate_fn(examples):
    """Collate function for data loading."""
    pixel_values = torch.stack([example["video"].permute(1, 0, 2, 3) for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def compute_metrics(eval_pred):
    """Compute accuracy metric."""
    metric = evaluate.load("accuracy")
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

def main():
    dataset_root_path = Path("/home/cerisnadm/Bureau/Thibault/Ego4d/dataset/skill")

    video_count_train, video_count_val, video_count_test = count_videos(dataset_root_path)
    video_total = video_count_train + video_count_val + video_count_test
    print(f"Total videos: {video_total}")

    all_video_file_paths = (
        list(dataset_root_path.glob("train/*/*.mp4"))
        + list(dataset_root_path.glob("val/*/*.mp4"))
        + list(dataset_root_path.glob("test/*/*.mp4"))
    )

    label2id, id2label = get_class_labels(all_video_file_paths)
    print(f"Unique classes: {list(label2id.keys())}.")

    model = load_model(label2id, id2label, freeze_feature_extractor=True)
    image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")

    num_frames_to_sample = model.config.num_frames
    sample_rate = 4
    fps = 30
    clip_duration = num_frames_to_sample * sample_rate / fps

    resize_to = (image_processor.size["shortest_edge"], image_processor.size["shortest_edge"])
    train_transform, val_transform = prepare_transforms(image_processor, num_frames_to_sample, resize_to)

    train_dataset, test_dataset = create_datasets(dataset_root_path, train_transform, val_transform, clip_duration)

    print(train_dataset.num_videos, test_dataset.num_videos)

    sample_video = next(iter(train_dataset))
    video_tensor = sample_video["video"]
    display_gif(video_tensor, mean=image_processor.image_mean, std=image_processor.image_std)

    model_name = "ViViT"
    new_model_name = f"{model_name}-finetuned-egoexo4D-subset_{time.time()}"
    num_epochs = 2
    batch_size = 1

    args = TrainingArguments(
        new_model_name,
        remove_unused_columns=False,
        num_train_epochs=num_epochs,
        learning_rate=1e-4,
        save_strategy="epoch",
        eval_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,
        logging_steps=10,
        logging_dir="./logs",
        metric_for_best_model="accuracy",
        max_steps=(train_dataset.num_videos // batch_size) * num_epochs,
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    train_results = trainer.train()
    print(train_results)

if __name__ == "__main__":
    main()
