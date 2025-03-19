from pathlib import Path
import os
import torch
import numpy as np
import imageio
import pytorchvideo.data
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    UniformTemporalSubsample,
)
from torchvision.transforms import Compose, Lambda, Resize
from transformers import (
    VivitImageProcessor,
    VivitForVideoClassification,
    Trainer,
    TrainingArguments
)

import evaluate
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image

def load_model(model_path, label2id, id2label):
    """Load the trained model for evaluation."""
    model = VivitForVideoClassification.from_pretrained(model_path).to("cuda")
    model.config.label2id = label2id
    model.config.id2label = id2label
    return model

def prepare_transforms(image_processor, num_frames_to_sample, resize_to):
    """Prepare data transformations for testing."""
    mean = image_processor.image_mean
    std = image_processor.image_std

    test_transform = Compose(
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

    return test_transform

def create_dataset(dataset_root_path, test_transform, clip_duration):
    """Create the test dataset."""
    test_dataset = pytorchvideo.data.Ucf101(
        data_path=os.path.join(dataset_root_path, "test"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=test_transform,
    )

    return test_dataset

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
    pixel_values = torch.stack([example["video"].permute(1, 0, 2, 3) for example in examples]).to("cuda")
    labels = torch.tensor([example["label"] for example in examples]).to("cuda")
    return {"pixel_values": pixel_values, "labels": labels}

def collate_pred_fn(examples):
    labels = torch.tensor([example["label"] for example in examples]).to("cuda")
    return {"labels": labels}

def compute_metrics(eval_pred):
    """Compute accuracy metric."""
    metric = evaluate.load("accuracy")
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

def plot_confusion_matrix(cm, class_names):
    """Plot the confusion matrix."""
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

def get_class_labels(all_video_file_paths):
    """Extract unique class labels from video file paths."""
    class_labels = sorted({str(path).split("/")[-2] for path in all_video_file_paths})
    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


def main():
    dataset_root_path = Path("/media/thibault/DATA/these_thibault/Dataset/data/EgoExo4D/dataset/")
    model_path = "/home/thibault/Documents/PhD/training/ViViT/checkpoints/ViViT-finetuned-egoexo4D-subset/checkpoint-536/"  # Update with your model path

    all_video_file_paths = (
        list(dataset_root_path.glob("train/*/*.mp4"))
        + list(dataset_root_path.glob("val/*/*.mp4"))
        + list(dataset_root_path.glob("test/*/*.mp4"))
    )
    
    image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
    label2id,id2label = get_class_labels(all_video_file_paths)  # Load or define your id2label mapping

    model = load_model(model_path, label2id, id2label)

    num_frames_to_sample = model.config.num_frames
    sample_rate = 4
    fps = 30
    clip_duration = num_frames_to_sample * sample_rate / fps

    resize_to = (image_processor.size["shortest_edge"], image_processor.size["shortest_edge"])
    test_transform = prepare_transforms(image_processor, num_frames_to_sample, resize_to)

    test_dataset = create_dataset(dataset_root_path, test_transform, clip_duration)

    # Display a random sample from the test dataset
    sample_video = next(iter(test_dataset))
    video_tensor = sample_video["video"]
    display_gif(video_tensor, mean=image_processor.image_mean, std=image_processor.image_std)

    # Predict the class for the sample
    inputs = collate_fn([sample_video])
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    print(f"Predicted class: {id2label[predictions.item()]}")

    # Evaluate the model on the entire test dataset
    

if __name__ == "__main__":
    main()
