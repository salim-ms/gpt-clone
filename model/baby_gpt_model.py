import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers.self_attention_block import SelfAttentionBlock
from model.layers.layer_norm import MyLayerNorm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BabyGPT(nn.Module):
    def __init__(self, config, vocab_size):
        super(BabyGPT, self).__init__()
        
        # configs
        embed_size = config["Model"].getint("embed_size")
        n_layers = config["Model"].getint("n_layers")
        sequence_length = config["Model"].getint("sequence_length")
        n_heads = config["Model"].getint("n_heads")
        
        # 2 embedding layers token + positional
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(sequence_length, embed_size)
        # attention blocks for n_layers
        self.layers = nn.Sequential(*[SelfAttentionBlock(embed_size=embed_size, n_heads=n_heads, sequence_length=sequence_length) for _ in range(n_layers)])
        # layer norm
        self.layer_norm = MyLayerNorm(dim_size=embed_size) 
        # linear layer head
        self.lm_head = nn.Linear(embed_size, vocab_size)
        
        # do weight tying
        self.lm_head.weight = self.token_embedding.weight
        
        # loss
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, x_indices, targets=None):
        B, T = x_indices.shape
        x_token_embeddings =  self.token_embedding(x_indices) # B, T, C
        x_pos_embeddings = self.positional_embedding(torch.arange(T, device=device)) # T, C
        x = x_pos_embeddings + x_token_embeddings # B, T, C
        
        # go through attention blocks
        x = self.layers(x)
        
        # apply layer norm
        x = self.layer_norm(x)
        # apply final lm head
        logits = self.lm_head(x) # B, T, vocab-size
        
        if targets is None:
            loss = None
        else:
            # compute loss
            # we need to flatten the logits to B*T,C from B,T,C
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # print(targets.shape)
            loss = self.loss_fn(logits, targets)
            
            
        return logits, loss
        
    
    def generate(self, max_new_tokens=100, idx=None):
        # if starting token not provided, generate one at random based on 65 vocab size
        if not idx:
            idx=torch.randint(low=0,high=self.vocab_size, size=(1,1), dtype=torch.long)
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
        

if __name__ == "__main__":
    rr = torch.arange(5)
    print(rr)
    x= torch.randint(low=0,high=65, size=(2,3), dtype=torch.long)
    print(x)
    x = x.to(device)
    # create a fake config
    config = {
        "Model":
            {
                "embed_size": 16,
                "n_layers": 2,
                "sequence_length": 8,
                "n_heads": 4
            },
        "Dataset":
            {
                "vocab_size": 65,
            }
    }
    baby_gpt = BabyGPT(config=config, vocab_size=65)
    baby_gpt = baby_gpt.to(device)
    # call with loss
    y, loss = baby_gpt(x, x)
    print(y.shape) # B*T, vocab_size # for every example in the batch for every time step we make a prediction
    print(loss.item())
    
    
    # call no loss
    y, loss = baby_gpt(x)
    print(y.shape)
    
        