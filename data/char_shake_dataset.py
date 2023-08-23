import random
import torch


random.seed(1233)


class CharShakeDataset:
    def __init__(self, path_to_file: str) -> None:
        
        # read the file and load the text
        with open(path_to_file, "r") as f:
            text = f.read()

        # generate vocab
        self.vocab = sorted(list(set(text)))
        assert len(self.vocab) == 65
        # generate mappings from char to index
        self.char_to_index = {ch:index for index, ch in enumerate(self.vocab)}
        # generate mappings from index to char
        self.index_to_char = {index:ch for index, ch in enumerate(self.vocab)}
        
        # generate train dataset and eval dataset
        # 90% train, 10% eval
        train_index_cutoff = int(0.9 * len(text))
        train_text = text[:train_index_cutoff]
        eval_text = text[train_index_cutoff:]

        # encode
        self.train_data = self.encode(train_text)
        self.eval_data = self.encode(eval_text)

    
    def load_datasets(self):
        return self.train_data, self.eval_data

    def encode(self, text: str):
        return torch.tensor([self.char_to_index[ch] for ch in text], dtype=torch.long)
        
    def decode(self, indices: torch.Tensor):
        return "".join([self.index_to_char[idx] for idx in indices.tolist()])

    def get_batch(self, split: str, batch_size=4, context_length=8):
        dataset = self.train_data if split == "train" else self.eval_data
        # generate batch-size random indices 
        rand_indices = torch.randint(len(dataset) - context_length, (batch_size,))
       
        x = torch.stack([dataset[rand_idx:rand_idx+context_length] for rand_idx in rand_indices])
        # target y are shifted by 1
        y = torch.stack([dataset[rand_idx+1:rand_idx+context_length+1] for rand_idx in rand_indices])
        return x, y
        
    
if __name__ == "__main__":
    shake_dataset = CharShakeDataset(path_to_file="./data/input.txt")
    train_dataset, eval_dataset = shake_dataset.load_datasets()
    print(train_dataset[:100])    
    print(shake_dataset.decode(train_dataset[:100]))
    
    x, y = shake_dataset.get_batch(split="train", batch_size=4, context_length=8)
    print(x)
    print(y)
    print(x.shape)
    