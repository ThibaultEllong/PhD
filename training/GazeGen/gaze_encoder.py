import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
import glob 
import pandas as pd
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm, trange
from time import time
import matplotlib.pyplot as plt
from gaze_dataset import GazeVideoDataset

class GazeEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, sequence_length):
        super(GazeEncoder, self).__init__()
        
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        
        # First Layer
        self.fc1 = nn.Linear(self.input_dim, 64)
        self.bn1 = nn.BatchNorm1d(self.sequence_length)  # Ensure the batch size is compatible
        self.relu1 = nn.ReLU()
        
        # Second Layer
        self.fc2 = nn.Linear(64, 256)
        self.bn2 = nn.BatchNorm1d(self.sequence_length)
        self.relu2 = nn.ReLU()
        
        # Output Layer
        self.fc3 = nn.Linear(256, self.latent_dim)
        
        
    def normalize(self, x, screen_width=1920, screen_height=1080):
        x[:,:, 0] = x[:,:, 0] / screen_width
        x[:,:, 1] = x[:,:,1] / screen_height
        return x

    def forward(self, x):
        x = self.normalize(x)
        x = self.fc1(x)  # Only use the first 16 frames
        x = self.bn1(x)  # Make sure input to BatchNorm1d is correctly shaped
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.fc3(x)
       
        return x

class GazeDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, prediction_length):
        super(GazeDecoder, self).__init__()
        # Define your decoder layers (mirroring the encoder structure is common)
        self.prediction_length = prediction_length
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        self.layers = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.BatchNorm1d(self.prediction_length),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(self.prediction_length),
            nn.ReLU(),
            nn.Linear(64, self.output_dim) # Output should match input dimensions
        )
        
    def denormalize_gaze(self, x, screen_width=1920, screen_height=1080):
        x[:, :, 0] = x[:, :, 0] * screen_width
        x[:, :, 1] = x[:, :, 1] * screen_height
        return x

    def forward(self, z):  # z is the latent vector
        
        return self.denormalize_gaze(self.layers(z[:,:self.prediction_length,:]))  # Ensure output shape matches input shape

class GazeEmbed(nn.Module):
    def __init__(self, input_dim, latent_dim, sequence_length, prediction_length):
        super(GazeEmbed, self).__init__()
        self.encoder = GazeEncoder(input_dim, latent_dim, sequence_length)
        self.decoder = GazeDecoder(latent_dim, input_dim, prediction_length)

    def forward(self, x):
        z = self.encoder(x)  # Encode
        x_reconstructed = self.decoder(z)  # Decode
        return x_reconstructed
    
def check_for_nans(model):
    # Check parameters
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN detected in parameter: {name}")

    # Check gradients
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"NaN detected in gradient of parameter: {name}")


def train_gaze_embed(n_epochs, lr, l_dim, in_dim, num_frames, prediction_length):
    print(os.getcwd())
    
    
    train_root_path = "/media/thibault/DATA/these_thibault/Dataset/data/EgoExo4D/dataset/train/"
    test_root_path = "/media/thibault/DATA/these_thibault/Dataset/data/EgoExo4D/dataset/test/"
    train_dataset = GazeVideoDataset(train_root_path, frames=num_frames, prediction_size=prediction_length, gaze_only=True)
    test_dataset = GazeVideoDataset(test_root_path, frames=num_frames, prediction_size=prediction_length, gaze_only=True)
  
    
    train_loader = DataLoader(train_dataset, 256, True)
    test_loader = DataLoader(test_dataset, 256, True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    
    
    
   
    
    model = GazeEmbed(input_dim=in_dim, latent_dim=l_dim, sequence_length=num_frames, prediction_length=prediction_length).to(device)
    
    checkpoint_path = Path(f'./Models/GazeEncoder/checkpoints/gaze_embed_{num_frames}_{prediction_length}_{time()}_{l_dim}_{lr}_{n_epochs}')
    os.mkdir(checkpoint_path)
    
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    criterion = nn.MSELoss()
    
    writer = SummaryWriter(log_dir='/home/thibault/Documents/Code/LogDir/autoencoder')
    
    for epoch in trange(n_epochs, desc="Training Autoencoder !"):
        train_loss = 0.0
        avg_dist = 0.0
        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x = batch["gaze"][:, :num_frames, :].to(device)
            y = model(x)
            loss = criterion(x[:,:prediction_length], y)

            train_loss += loss.detach().cpu().item() / len(train_loader)
            avg_dist += torch.abs(x[:,:prediction_length]-y).mean().detach().cpu().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch + 1}/{n_epochs} loss: {train_loss:.2f}")
        scheduler.step(train_loss)
        print(f"Average distance error (in pixel): {avg_dist / len(train_loader)}")
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)
        
        torch.save(model.state_dict(), checkpoint_path / f'gaze_encoder_epoch_{epoch}.pth') 
    writer.close() 
    

    with torch.no_grad():
        checkpoint = torch.load('./Models/GazeEncoder/checkpoints/gaze_encoder_1739293104.678117_512_0.001_50/gaze_encoder_epoch_49.pth')
        # model.load_state_dict(checkpoint)
        model.eval()  
        correct, total = 0, 0
        test_loss = 0.    
        input = torch.tensor([])
        preds = torch.tensor([])
        avg_dist = 0.0
        i = 0
        for batch in tqdm(test_loader, desc="Testing"):
            x = batch["gaze"][:, :num_frames, :].to(device)
            y = model(x)   
            loss = criterion(x[:,:prediction_length], y)
            test_loss += loss.detach().cpu().item() / len(test_loader)        
            correct += torch.sum(torch.abs(x[:,:prediction_length]-y)<=2).detach().cpu().item()
            total += x.numel()        
            input = torch.cat((input,x[:,:prediction_length].cpu()), dim=0)
            preds = torch.cat((preds,y.cpu()), dim=0)
            avg_dist += torch.abs(x[:,:prediction_length]-y).mean().detach().cpu().item()
            i += 1
        accuracies = []
        for i in range(0,10):
            acc = torch.sum(torch.abs(input-preds)<=i).detach().cpu().item()
            accuracies.append(acc/total*100)      
        plt.hist(accuracies, bins=10)
        plt.show()           
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")
        print(f"Average distance error (in pixel): {avg_dist / len(test_loader)}")
    return model


if __name__ == "__main__":
    main()
        
