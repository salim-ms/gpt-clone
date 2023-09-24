import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers.attention_head import MyAttentionHead


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, n_heads, head_size, max_context_length, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        # create a list of multiple heads
        self.multi_heads = nn.ModuleList([MyAttentionHead(embed_size=embed_size, head_size=head_size, max_context_length=max_context_length) for _ in range(n_heads)])
        # projection layer which takes B T, C=embed_size from multiple heads concatenated
        self.projection_layer = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        # run x over each head and concatenat
        x = torch.cat([head(x) for head in self.multi_heads], dim=-1)
        # print(x.shape)
        x = self.projection_layer(x)
        x = self.dropout(x)
        return x
    

if __name__ == "__main__":
    example = torch.randn([2, 3, 16])
    print(example)
    
    multi_head_attention = MultiHeadAttention(embed_size=16, n_heads=4, max_context_length=8, head_size=4)
    
    x = multi_head_attention(example)
    
    print(x)
    print(x.shape)