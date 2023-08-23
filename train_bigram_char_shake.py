import torch
import torch.nn as nn
from torch.nn import functional as F
from data.char_shake_dataset import CharShakeDataset
from model.bigram_model import BigramModel

torch.manual_seed(1337)


# create a path to store trained model
BATCH_SIZE = 4
CONTEXT_LENGTH= 8
BIGRAM_MODEL_PATH = "bigram_model_saved.pth"

if __name__ == "__main__":
    shake_dataset = CharShakeDataset(path_to_file="./data/input.txt")
    bigram_model = BigramModel(vocab_size=len(shake_dataset.vocab))
    train_dataset, eval_dataset = shake_dataset.load_datasets()
    # create an optimizer
    optimizer = torch.optim.AdamW(bigram_model.parameters(), lr=1e-3)
    
    # training loop
    running_loss = 0
    for i in range(5000):
        # get training data batch
        x, y = shake_dataset.get_batch(split="train", batch_size=BATCH_SIZE, context_length=CONTEXT_LENGTH)
        # forward
        logits, loss = bigram_model(x, y)
        # set optimizer grad to 0
        optimizer.zero_grad(set_to_none=True)
        # compute loss backprop
        loss.backward()
        # step optimizer
        optimizer.step()
        
        running_loss += loss.item()
        # output loss every now and then
        if i % 100 == 99:
            print(running_loss / 100)
            running_loss = 0
    
    # save model
    torch.save(bigram_model.state_dict(), BIGRAM_MODEL_PATH)