import torch
import torch.nn as nn
import torch.nn.functional as F


class MyLayerNorm(nn.Module):
    def __init__(self, dim_size):
        super(MyLayerNorm, self).__init__()
        # each feature will be scaled and shifted via weights and biases or gamma and beta
        self.weights = nn.Parameter(torch.ones(dim_size))
        self.biases = nn.Parameter(torch.zeros(dim_size))
        self.epsilon = 1e-5

        
    def forward(self, x):
        print("hello norm")
        # compute mean
        x_mean = x.mean(dim=-1, keepdim=True)
        # compute variance
        x_var = x.var(dim=-1, keepdim=True, unbiased=False)
        # compute normalized_x
        normalized_x = (x - x_mean) / torch.sqrt(x_var + self.epsilon) 
        # compute layer_normed_x
        layer_normed_x = normalized_x * self.weights + self.biases
        return layer_normed_x
            
        
class LayerNorm1d: # (used to be BatchNorm1d)
  
  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)
  
  def __call__(self, x):
    # calculate the forward pass
    xmean = x.mean(1, keepdim=True) # batch mean
    xvar = x.var(1, keepdim=True) # batch variance
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    return self.out
  
  def parameters(self):
    return [self.gamma, self.beta]


class LayerNormBiasEnabled(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
    


if __name__ == "__main__":
    x_tensor = torch.tensor([[1, 2, 3],
                             [4, 5, 6]], dtype=torch.float32)
    
    another_tensor = torch.tril(torch.ones(2,3,4))
    # print(x_tensor)
    # print(x_tensor.mean(dim=1, keepdim=True))
    # print(x_tensor.var(dim=1, keepdim=True))
    layer_norm = MyLayerNorm(dim_size=3)
    my_normalized_x = layer_norm(x_tensor)
    print(my_normalized_x)
    print()
        
    # use normal layer norm
    print(x_tensor)
    orig_norm = nn.LayerNorm(3)
    print(f"norm shape {orig_norm.normalized_shape}")
    orig_normalized = orig_norm(x_tensor)
    
    print(orig_normalized)
    
    
    karpathy = LayerNorm1d(dim=3)
    print(karpathy(x_tensor))
    
    print(F.layer_norm(x_tensor, layer_norm.weights.shape, layer_norm.weights, layer_norm.biases, layer_norm.epsilon))
     
     
    no_bias_norm = LayerNormBiasEnabled(ndim=3, bias=False)
    print(no_bias_norm(x_tensor))