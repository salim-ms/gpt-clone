import torch
import torch.nn as nn
import torch.nn.functional as F


class Logistic(nn.Module):
    def __init__(self):
        super(Logistic, self).__init__()
        self.a = nn.Parameter(torch.rand(1))
        self.b = nn.Parameter(torch.rand(1))
        
        
    def forward(self, x, y=None):
        z = self.a * x + self.b
        y_prime = torch.sigmoid(z)
        
        loss = None
        if y is not None:
            loss = F.cross_entropy(y_prime, y)
        
        return y_prime, loss

if __name__ == "__main__":
    
    x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    y = torch.tensor([0, 0, 1, 1, 1], dtype=torch.float32)
    
    logistic = Logistic()
    
    optimizer = torch.optim.AdamW(logistic.parameters(), lr=0.001)
    for i in range(10000):
        
        y_prime, loss = logistic(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(logistic.a)
    print(logistic.b)
    
    print(y_prime)