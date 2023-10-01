from loguru import logger
import random
import torch
import tokenizers
from tokenizers import Tokenizer
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents, Replace, Strip
from tokenizers.pre_tokenizers import Whitespace, WhitespaceSplit, Split
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import decoders

from utils.config_parser import create_directory
import os

"""
Run from root directory
"""

random.seed(1233)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class TokenShakeDataset:
    def __init__(self, path_to_file: str) -> None:
        
        # read the file and load the text
        with open(path_to_file, "r") as f:
            text = f.read()

        # BPE tokenizer
        self.tokenizer = self.build_tokenizer(path_to_file)
        
        # generate train dataset and eval dataset
        # 90% train, 10% eval
        train_index_cutoff = int(0.9 * len(text))
        train_text = text[:train_index_cutoff]
        eval_text = text[train_index_cutoff:]

        # encode "In Memory dataset"
        self.train_data = self.encode(train_text)
        self.eval_data = self.encode(eval_text)

        # get vocab_size
        self.vocab_size = self.tokenizer.get_vocab_size()
    
    def build_tokenizer(self, path_to_file):
        # make sure we have dirs in place
        tokenizer_dir_path = "/workspace/data_outputs/"
        tokenizer_file_name = "bpe_tokenizer_shake.json"
        create_directory(tokenizer_dir_path)
        
        # make sure path is a list of files
        if isinstance(path_to_file, str):
            # convert to list
            path_to_file = [path_to_file]
        
        
        # check if we have trained a tokenizer before
        if os.path.isfile(f"{tokenizer_dir_path}/{tokenizer_file_name}".replace("//", "/")):
            tokenizer = Tokenizer.from_file(f"{tokenizer_dir_path}/{tokenizer_file_name}".replace("//", "/"))
            logger.info("Reading Tokenizer From Disk ...")
            logger.info(f"\n Tokenizer Vocab Size {tokenizer.get_vocab_size()} \n")
            return tokenizer
        
        logger.info("Building A Tokenizer from scratch ....")
        # Create the tokenizer and the trainer
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(special_tokens=[
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[PAD]",
            "[MASK]"
            ])
        
        # add normalizers and preprocessor
        tokenizer.normalizer = normalizers.Sequence([NFD(), StripAccents(), Replace(r'(?<=\n)[ \t]+', "\n")])
        
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence([Split(pattern=' ', behavior="removed"), 
                                                                      Split(pattern='\n', behavior="isolated")])
        # train the tokenizer
        tokenizer.train(path_to_file, trainer)
        
        # save the tokenizer
        tokenizer.save(f"{tokenizer_dir_path}/{tokenizer_file_name}".replace("//", "/"))
        
        logger.info(f"\n Tokenizer Vocab Size {tokenizer.get_vocab_size()} \n")
        
        return tokenizer
    
    
    def load_datasets(self):
        return self.train_data, self.eval_data

    def encode(self, text: str):
        output = self.tokenizer.encode(text)
        ids = output.ids # type list
        # convert to tensor
        return torch.tensor(ids, device=device)
        
    def decode(self, indices: torch.Tensor):
        return self.tokenizer.decode(indices.tolist())

    def get_batch(self, split: str, batch_size=4, max_context_length=8):
        dataset = self.train_data if split == "train" else self.eval_data
        # generate batch-size random indices 
        rand_indices = torch.randint(len(dataset) - max_context_length, (batch_size,))
       
        x = torch.stack([dataset[rand_idx:rand_idx+max_context_length] for rand_idx in rand_indices])
        # target y are shifted by 1
        y = torch.stack([dataset[rand_idx+1:rand_idx+max_context_length+1] for rand_idx in rand_indices])
        x = x.to(device)
        y = y.to(device)
        return x, y
        
    
if __name__ == "__main__":
    shake_dataset = TokenShakeDataset(path_to_file="./data/input.txt")
    train_dataset, eval_dataset = shake_dataset.load_datasets()
    print(train_dataset[:100])    
    print(shake_dataset.decode(train_dataset[:100]))
    
    x, y = shake_dataset.get_batch(split="train", batch_size=4, max_context_length=8)
    print(x)
    print(y)
    print(x.shape)
    
    
    ids = shake_dataset.encode("hello \n this is the king  it's amazing\n")
    print(ids)
    print(shake_dataset.decode(ids))
    
    
    ids = shake_dataset.encode("king\n\n\nking\n")
    print(ids)
    print(shake_dataset.decode(ids))
    
    print(shake_dataset.encode("\n"))
    print(shake_dataset.encode(" \n"))
    print(shake_dataset.encode("  \n"))
    
    
    output = shake_dataset.tokenizer.encode("king\n\n\nking\n")
    ids = output.ids # type list
    print(output.tokens)

    output = shake_dataset.tokenizer.encode("king \n\n \nking \n")
    ids = output.ids # type list
    print(output.tokens)