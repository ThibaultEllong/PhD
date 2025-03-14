import torch.nn as nn
import torch
from torch.nn import MultiheadAttention



class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(CrossAttentionBlock, self).__init__()
        self.mha = MultiheadAttention(embed_size, heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        
    def forward(self, value, key, query, mask=None):
        attention, attn_weights = self.mha(query, key, value, attn_mask=mask)
        
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        
        return out, attn_weights
