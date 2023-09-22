import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, embed_size, dropout=0.0):
        super(FeedForward, self).__init__()
        # this layer expands embedding size by 4x then project back to embed_size
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout)
        )
        
    
    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    x = torch.randn([2, 3, 16])
    feed_forward = FeedForward(embed_size=16)
    y = feed_forward(x)
    print(y.shape)
    
    