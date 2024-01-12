import torch
import torch.nn as nn
import torch.nn.functional as F

"""
LABML Implementation treat shapes as [seq_len, dimension]
then transform it to [seq_len, batch, heads, dimension]

we like to work with batch, seq_len, dimension
so this implementation is modified
refer to original for reference
https://nn.labml.ai/transformers/rope/index.html
"""


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class RotaryEmbeddingLabML(nn.Module):
    def __init__(self, d: int, base: int = 10_000) -> None:
        super(RotaryEmbeddingLabML, self).__init__()
        
        # d is integer representing the number of features in last dimension "d" of vector token which will receive rotary computation
        self.d = d
        self.base = base

        self.cos_cached = None # shape [1, seq_len, 1, d] where 1, 1, is batch and n_heads
        self.sin_cached = None # shape [1, seq_len, 1, d] where 1, 1, is batch and n_heads
        
        
    def _build_cache(self, x: torch.Tensor):
        # goal is to get sin() and cos() of Matrices of shape [seq_len, theta*2]
        # theta has shape of [d/2] half the dimension because we apply to pairs of features positioned [0, 1,......, d/2, 1+d/2]
        
        # if cos_cached/sin_cached already computed return right away and shape of x sequence length must
        # less or equal to size of computed cos_cached which is based on original x tensor fed to this function
        if self.cos_cached is not None and self.shape[1] <= self.cos_cached.shape[0]:
            return
        
        # desired sequence length
        seq_len = x.shape[1]
        # generate sequence positions
        seq_idx = torch.arange(0, seq_len, device=device).float().to(device) # shape [seq_len]
        
        # compute thetas, 10_000 power of 2(i - 1)/d -> 2(i - 1) means only even numbers 0, 2, 4, ....
        theta = 1. / (self.base ** (torch.arange(0, self.d, 2)).float() / self.d ).to(device) # shape [d / 2]
        
        # generate a matrix of size (seq_idx, theta)
        #  idx_theta = torch.einsum('n,d->nd', seq_idx, theta) outer product
        # i like normal python broadcasting
        # reshape seq_idx to [n, 1] and theta to [1, d/2] -> [n, d/2]
        
        idx_theta = seq_idx.view(-1, 1) @ theta.view(1, -1) # shape is [seq_length, d/2]
        
        #idx_theta_org = torch.einsum('n,d->nd', seq_idx, theta)
        # assert torch.all(idx_theta == idx_theta_org) == True

        # expand idx_theta to shape of [seq_len, [theta] + [theta]] so we can apply the transformations to pairs of features
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=-1) # shape [seq_length, d]
        # print(idx_theta2.shape) 
        
        # compute cos and sin and reshape to [1, seq_len, 1, d] 
        self.cos_cached = idx_theta2.cos()[None, :, None, :]
        self.sin_cached = idx_theta2.sin()[None, :, None, :]
        
        self.cos_cached.to(device=device)
        self.sin_cached.to(device=device)
        
        # print(self.cos_cached.shape)
        # print(self.sin_cached.shape)
        
        
    def _neg_half(self, x: torch.Tensor):
        # x is 4d [batch, seq_len, n_heads, d_dim]
        # split x into [-second_half, first_half] where half is determined by d
        d_2 = self.d // 2
        return torch.cat([ -x[:, :, :, d_2:], x[:, :, :, :d_2] ], dim=-1)
        
    
    def forward(self, x: torch.Tensor):
        # x is 4d [batch, seq_len, n_heads, d_dim]
        self._build_cache(x)
        
        # rotary can be applied to a portion of features specified by d, hence we split the tensor
        x_rope, x_pass = x[..., :self.d], x[..., self.d:] # [batch, seq, n_heads, :d]
        
        neg_x_rope = self._neg_half(x_rope)
        # print(x_rope.shape) # [batch, seq, n_heads, :d]
        # print(neg_x_rope.shape) # [batch, seq, n_heads, :d]
        # print(x_pass.shape) # [batch, seq, n_heads, d:] # d usually 0

        # apply cos and sin only up to x sequence length
        # print(self.cos_cached.shape)
        x_rope_final = (x_rope * self.cos_cached[:, :x.shape[1], :, :]) + (neg_x_rope * self.sin_cached[:, :x.shape[1], :, :])
        # print(f"x_rope final is {x_rope_final.shape}") [batch, seq, n_heads, dim]
        
        # re-stitch the tensor together
        return torch.cat((x_rope_final, x_pass), dim=-1)
    
    
if __name__ == "__main__":
    x = torch.randn([2, 8, 32], dtype=torch.float16).to(device)
    print(x.shape)
    
    # x must be transformed to 4d by adding n_head dimension to third position batch, seq, n_heads, d_dim
    x = x[:, :, None, :]
    print(x.shape)
    
    rotay_layer = RotaryEmbeddingLabML(d=x.shape[3])
    
    rotated_x = rotay_layer(x)
    
    print(f"shape of original x is {x.shape} must match rotated shape {rotated_x.shape} [batch, seq, n_heads, d]")