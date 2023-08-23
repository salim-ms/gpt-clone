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
    
    # load trained model
    bigram_model.load_state_dict(torch.load(BIGRAM_MODEL_PATH))
    bigram_model.eval()
    # shake_text = bigram_model.generate(idx=torch.zeros((1,1), dtype=torch.long), max_new_tokens=200)[0]
    # print(shake_dataset.decode(shake_text))
    
    print(torch.randint(low=0,high=65, size=(1,1), dtype=torch.long))
    shake_text = bigram_model.generate(idx=torch.randint(low=0,high=65, size=(1,1), dtype=torch.long), max_new_tokens=200)[0]
    print(shake_dataset.decode(shake_text))