import torch
import torch.nn as nn
import torch.nn.functional as F


class MyBatchNorm1d(nn.Module):
    def __init__(self, dim_features):
        super(MyBatchNorm1d, self).__init__()
        # scalables and shifters gamman and beta
        self.weights = nn.Parameter(torch.ones(dim_features))
        self.biases = nn.Parameter(torch.zeros(dim_features))
        # momentum and epsilon
        self.momentum = 0.1
        self.epsilon = 1e-5
        # running mean and variance are part of model state but don't get updated by optimizer
        self.register_buffer('running_mean', torch.zeros(dim_features))
        self.register_buffer('running_var', torch.ones(dim_features))
        
        
    def forward(self, x):
        if self.training:
            # forward training
            # compute mean for every feature across the batch, e.g. 2x3 -> 1x3 mean
            # compute var for every feature across the batch e.g. 2x3 -> 1x3 var
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            # print(mean)
            # print(var)
            # compute running mean and running var
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            
        else:
            # if during eval, just use the stored running mean and var
            mean = self.running_mean
            var = self.running_var
        
        # normalize the input x - mean / sqrt(var + eps)        
        normalized_x = (x - mean) / torch.sqrt(var + self.epsilon)
        # apply weights and biases normalized * weights + biases
        return normalized_x * self.weights + self.biases
        



if __name__ == "__main__":
    x = torch.tensor([[1, 2, 3],
                      [4, 5, 6]], dtype=torch.float32)
    
    
    batch_norm1d = nn.BatchNorm1d(3)
    v = batch_norm1d(x)
    print(v)
    
    print("scratch implementation")
    mybatchnorm = MyBatchNorm1d(dim_features=3)
    vprime = mybatchnorm(x)
    print(vprime)
    
    
    # x3d = torch.tril(torch.ones((2, 3, 4), dtype=torch.float32))    
    # print(x3d)
    # # for this to work we need to reshape the tensor to have features on the second dimension
    # x3d_permuted = x3d.permute(0, 2, 1)
    # batch_norm1d = nn.BatchNorm1d(4)
    
    # v = batch_norm1d(x3d_permuted)
    # print(v)
    
    x4d = torch.randn([2, 5, 3, 3], dtype=torch.float32)
    
    x4d_mean = x4d.mean([0, 2, 3], keepdims=True)
    print(x4d_mean)
    print(x4d_mean.size())
    
    
    
    