import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EfficientCausal(nn.Module):
    def __init__(self, embed_size, n_heads, block_size, is_bias=False, dropout=0.0):
        super(EfficientCausal, self).__init__()
        self.embed_size = embed_size
        self.n_heads = n_heads
        self.block_size = block_size
        self.dropout = dropout
        # K, Q, V in single linear layer
        self.c_attn = nn.Linear(embed_size, 3 * embed_size, bias=is_bias)
        self.c_proj = nn.Linear(embed_size, embed_size, bias=is_bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # check if we have flash attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("warning no flash attention, resulting to slow computation")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("masking", torch.tril(torch.ones(block_size, block_size))
                                        .view(1, 1, block_size, block_size))
        
        
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # generate k,q,v of size (B, T, C) note C includes heads
        q,k,v = self.c_attn(x).split(self.embed_size, dim=2)
        
        # change view to include heads
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # B, nh, T, hs
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # B, nh, T, hs
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # B, nh, T, hs
        
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True) # B, nh, T, hs
        else:
            att = q @ k.transpose(-2, -1) # B, nh, T, T
            # scale it
            att = att * (1.0 / math.sqrt(k.size(-1)))
            # mask it
            att = att.masked_fill(self.masking[:, :, :T, :T] == 0, float('-inf'))
            # Softmax so each token knows how much value to get from other tokens
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
            
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        
        y = self.resid_dropout(self.c_proj(y))
        return y
    
    
if __name__ == "__main__":
    x = torch.randn([2, 8, 64])
    multi_head_attention = EfficientCausal(embed_size=64, n_heads=4, block_size=8)
    
    print(multi_head_attention(x).shape) # 2, 8, 64
    
    