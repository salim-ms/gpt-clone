import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

class MyAttentionHead(nn.Module):
    def __init__(self, embed_size, head_size, max_context_length, dropout=0.1):
        super(MyAttentionHead, self).__init__()
        # keys, values, queries 
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        # masking matrix
        self.register_buffer("masking", torch.tril(torch.ones(max_context_length, max_context_length)))
        # dropout
        self.dropout = nn.Dropout(dropout)
        
    
    def forward(self, x):
        
        # x shape B, T, C (batch, sequence/Time, Channels/dims)
        B, T, C = x.shape 
        # matrix multiply with keys, values, and query
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        # multiply query with key to get B, T, T (which tells how each token will attend by how much to other tokens in the sequence)
        # query B, T, C key B, T, C, transpose key (B, C, T)
        weights = query @ key.transpose(2, 1)
        # mask the weights before softmax, replace all 0s with -inf so softmax treats them as 0 during computation
        weights = weights.masked_fill(self.masking[:T, :T] == 0, float('-inf')) # why :T, because total sequence length can be greater than this particular batch T length
        # scale the weights, divide by sqrt(embedding size)
        weights = weights / torch.sqrt(torch.tensor(C)) 
        # apply softmax
        soft_weights = F.softmax(weights, dim=-1) # apply on last dimension B, T, T
        # print(f"softmax weights {soft_weights}")
        soft_weights = self.dropout(soft_weights)
        # multiply softmax (B, T, T) by values (B, T, C) -> B, T, C
        res = soft_weights @ value
        # return 
        return res




## karpathy
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        n_embd = 16
        block_size=2
        k_dropout = 0.0

        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(k_dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
    
    
if __name__ == "__main__":
    import math
    x = torch.ones([1, 2, 16])
    print(x.size()) # B=1, T=2, C=16
    
    B, T, C = x.shape
    print(B)
    print(T)
    head_size = int(math.sqrt(C))
    print(head_size)
    m_head_attention = MyAttentionHead(embed_size=C, head_size=head_size, max_context_length=T, dropout=0.0)
    m_head_attention2 = MyAttentionHead(embed_size=C, head_size=head_size, max_context_length=T, dropout=0.0)
    
    res = m_head_attention(x) # output B, T, head_size=4
    res2 = m_head_attention2(x)
    print(res.size())
    print(res)
    print()
    print(res2)
    print()

    khead = Head(head_size=head_size)
    k_res = khead(x)
    print(k_res)
    
    