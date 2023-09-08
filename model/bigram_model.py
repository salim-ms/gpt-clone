import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # create layers
        self.vocab_size = vocab_size
        self.embedding_layer = nn.Embedding(vocab_size, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, x, y_targets=None):
        logits = self.embedding_layer(x) # out shape: B T vocab_size
        if y_targets is None:
            loss = None
        else:
            """ the thing to remmeber is that in transformers, each time step is a prediction of its own.
            hence we flattent the logits and the targets, it's a classification problem one token to another.
            """
            # given that the model is bigram and our inputs of shape B T "time sequence"
            # we need to flatten it
            # reshape logits to B*T, vocab_size
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # shape [32, 65] (batch=4, context=8)
            y_targets = y_targets.view(B*T) # shape [32]
            loss = self.loss_fn(logits, y_targets)
            
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
    from data.char_shake_dataset import CharShakeDataset
    
    bigram_model = BigramModel(vocab_size=65)
    # generate random input
    x_rand = torch.randint(low=0, high=65, size=(4, 8), dtype=torch.long)
    y_rand = torch.randint(low=0, high=65, size=(4, 8), dtype=torch.long)
    print(x_rand)
    print(x_rand.shape)
    
    logits, loss = bigram_model(x_rand, y_rand)
    print(logits)
    print(loss)
    
    shake_dataset = CharShakeDataset(path_to_file="./data/input.txt")
    
    generated_shakespeare = bigram_model.generate(idx=torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0]
    print(generated_shakespeare)
    print(shake_dataset.decode(generated_shakespeare))
    
    
    