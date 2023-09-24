import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers.layer_norm import MyLayerNorm
from model.layers.multi_head_attention import MultiHeadAttention
from model.layers.feed_forward import FeedForward


class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_size, n_heads, max_context_length):
        super(SelfAttentionBlock, self).__init__()
        # compute head_size
        head_size = embed_size // n_heads
        # layer norm
        self.layer_norm1 = MyLayerNorm(dim_size=embed_size)
        # multi head attention
        self.multi_head_attention = MultiHeadAttention(embed_size=embed_size,
                                                       n_heads=n_heads,
                                                       head_size=head_size,
                                                       max_context_length=max_context_length)
        # layer norm 2
        self.layer_norm2 = MyLayerNorm(dim_size=embed_size)
        # feedforward
        self.feed_forward = FeedForward(embed_size=embed_size)
        
            
    def forward(self, x):
        # we need to do skip connections
        x = x + self.multi_head_attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x


if __name__ == "__main__":
    x = torch.randn([2, 3, 16])
    
    block = SelfAttentionBlock(embed_size=16, n_heads=4, max_context_length=8)
    y = block(x)
    # y = 2, 3, 16
    print(y.shape)
    