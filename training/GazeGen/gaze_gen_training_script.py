gimport os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 
from tqdm import tqdm
from transformers import VivitModel, VivitImageProcessor, VivitConfig

from gaze_gen import GazeGen
from gaze_encoder import GazeEmbed
from gaze_dataset import GazeVideoDataset, custom_collate_fn


# Function to train the model
def train(model, train_loader, processor, optimizer, criterion, device, epochs, save_path, writer, scheduler):
    model.train()  # Set model to training mode
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        mean_dist = 0

        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch') as pbar:
            for batch_idx, batch in enumerate(pbar):
                frames = batch["frames"]
                frames = processor(frames.numpy(), do_resize=True, size=224, return_tensors="pt")
                gaze = batch["gaze"]
                targets = gaze[:, 16:].to(device)  # Get target gaze locations

                optimizer.zero_grad()

                # Forward pass
                outputs = model(frames.pixel_values[:,:16].to(device), gaze[:,:16].to(device))[0]
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                scheduler.step(loss)

                running_loss += loss.item()
                mean_dist += torch.mean(torch.norm(outputs - targets, dim=1))
                print("Mean Distance: ", mean_dist)
                correct += (outputs.argmax(dim=1) == targets.argmax(dim=1)).sum().item()
                total += targets.size(0)

                # Log to TensorBoard every 10 batches
                if batch_idx % 10 == 0:
                    writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + batch_idx)
                    writer.add_scalar('Training Accuracy', 100. * correct / total, epoch * len(train_loader) + batch_idx)

                # Update progress bar
                pbar.set_postfix(loss=running_loss / (batch_idx + 1), accuracy=100. * correct / total)

        # Logging at the end of each epoch
        avg_mean_dist = mean_dist / len(train_loader)
        avg_loss = running_loss / len(train_loader)
        accuracy = 100. * correct / total
        writer.add_scalar('Epoch Loss', avg_loss, epoch)
        writer.add_scalar('Epoch Accuracy', accuracy, epoch)

        print(f'Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%')

        # Save the model checkpoint after each epoch
        model_path = os.path.join(save_path, f"gazegen_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_path)
        writer.add_text("Checkpoint Path", model_path, epoch)  # Log checkpoint path

    writer.close()


# Main function to initialize dataset, model, and training loop
def main(args):
    # Hyperparameters
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TensorBoard Logger
    writer = SummaryWriter(log_dir=args.log_dir)

    # Model Save Path
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)  # Ensure save directory exists

    # Load ViViT model (Pretrained)
    config = VivitConfig(num_frames=16)
    vivit = VivitModel(config).from_pretrained(args.vivit_checkpoint, config=config, ignore_mismatched_sizes=True).to(device)
    vivit.classifier = nn.Identity()  # Remove classification head
    vivit.eval()

    # Load gaze encoder
    gaze_encoder = GazeEmbed(2, 768, 16, 10).to(device)
    gaze_encoder.load_state_dict(torch.load("./checkpoints/gaze_embed_768_0.001_15_epoch_14.pth"))
    gaze_encoder.eval()
        
    
    # Initialize GazeGen model
    model = GazeGen(gaze_encoder, vivit, batch_size, frame_num=16).to(device)

    # Load dataset and DataLoader
    gaze_dataset = GazeVideoDataset(args.metadata_root_path, frames=16, prediction_size=10)
    dataloader = DataLoader(gaze_dataset, num_workers=64, collate_fn=custom_collate_fn, batch_size=batch_size, shuffle=True)

    processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


    # Train model with TensorBoard logging
    train(model, dataloader, processor, optimizer, criterion, device, epochs, save_path, writer, scheduler)


# Argument parser for command-line execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GazeGen Model on EgoExo4D dataset.")

    parser.add_argument("--metadata_root_path", type=str, required=True, help="Path to dataset metadata.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--save_path", type=str, default="./checkpoints", help="Directory to save model checkpoints.")
    parser.add_argument("--vivit_checkpoint", type=str, required=True, help="Path to pretrained ViViT checkpoint.")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory to save TensorBoard logs.")

    args = parser.parse_args()

    main(args)

