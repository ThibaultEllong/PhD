import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn as nn 

from gaze_encoder import GazeEmbed
from cross_attention import CrossAttentionBlock
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt

class GazeGen(nn.Module):
    def __init__(self, gaze_embed,vivit,batch_size, representation_dim=768, frame_num=32, num_heads=8, prediction_size=10):
        super(GazeGen,self).__init__()
        
        self.representation_dim = representation_dim
        self.frame_num = frame_num
        self.num_heads = num_heads
        self.prediction_size = prediction_size
        self.batch_size = batch_size
        
        self.cls = nn.Parameter(torch.rand(self.batch_size, self.prediction_size, self.representation_dim)).to("cuda")
        
        self.connect_gaze_from_512 = nn.Linear(512, self.representation_dim).to("cuda")
        self.connect_gaze_to_512 = nn.Linear(self.representation_dim,512).to("cuda")
        self.vivit = vivit
        self.gaze_encoder = gaze_embed.encoder
        self.gaze_decoder = gaze_embed.decoder
        self.cross_attention = CrossAttentionBlock(representation_dim, self.num_heads, 0.1, 4).to("cuda")
        
        self.pos_encoding = nn.Embedding(self.frame_num, self.representation_dim).to("cuda")
        
    def forward(self, img_seq, gaze_seq):
        pos = torch.arange(self.frame_num).unsqueeze(0).repeat(img_seq.size(0), 1).to("cuda") #positional encoding
        
        img_embed = self.vivit(img_seq) #image embedding
        
        gaze_embed = self.gaze_encoder(gaze_seq) + self.pos_encoding(pos).squeeze() # gaze embedding + positional encoding
        #the original 512 features-sized gaze vector is converted to 768 features to match the image embedding
        gaze_embed = torch.cat((gaze_embed.unsqueeze(0).transpose(0,1), self.cls.unsqueeze(0).transpose(0,1)), dim=-2)
        x, attn_weights = self.cross_attention(query = gaze_embed.squeeze(1).transpose(0,1) , key = img_embed.last_hidden_state.transpose(0,1), value = img_embed.last_hidden_state.transpose(0,1))
        
        results = self.gaze_decoder(x.transpose(0,1))
        return results, attn_weights
    
def main(gaze_encoder, vivit):
    with torch.no_grad():
        model= GazeGen(gaze_encoder, vivit)
        model.train()
        # Summing attention weights across heads and layers (simplified)
        

        img_seq = torch.rand(1, 32, 3, 224, 224).to("cuda")
        gaze_seq = torch.empty(32, 2).uniform_(0, 1080).to("cuda")
        results, attn_weights  = model(img_seq, gaze_seq)
        return results.transpose(0,1), attn_weights

if __name__ == "__main__":
    print("gaze_gen.py is being run directly")
