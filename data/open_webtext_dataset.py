import tiktoken
import numpy as np
from datasets import load_dataset
import os

DATASET_NAME = "stas/openwebtext-10k"
GPT_TOKENIZER = tiktoken.get_encoding("gpt2")

# print(GPT_TOKENIZER.eot_token) # 50256


def tokenize_text(example):
    token_ids = GPT_TOKENIZER.encode_ordinary(example["text"])
    # append end of token
    token_ids.append(GPT_TOKENIZER.eot_token)
    
    out = {"ids": token_ids, "len": len(token_ids)}
    return out

    
class OpenWebTextDataset:
    pass


def main():
    dataset = load_dataset(DATASET_NAME)
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=224, shuffle=True)
    # change test to val key
    dataset['val'] = dataset.pop('test')
    """
    DatasetDict({
        train: Dataset({
            features: ['text'],
            num_rows: 8000
        })
        val: Dataset({
            features: ['text'],
            num_rows: 2000
        })
    })
    """
    # convert text to ids via tokenizer gpt2 bpe
    dataset = dataset.map(tokenize_text, remove_columns=["text"], num_proc=8)
    
    """
    DatasetDict({
        train: Dataset({
            features: ['ids', 'len'],
            num_rows: 8000
        })
        val: Dataset({
            features: ['ids', 'len'],
            num_rows: 2000
        })
    })
    """
    
    # create single np array for each split which spans the entire ids to be used for training by sampling at random from it
    for split, ds in dataset.items():
        # file names are (__file__)_train.bin, (__file__)_val.bin 
        filename = os.path.join(f"{__file__}_{split}.bin")
        total_input_size = np.sum(ds["len"])
        arr = np.memmap(filename=filename, dtype=np.uint16, mode="w+", shape=(total_input_size,))

        # fill the array in batches
        idx = 0
        total_batches = 12
        for i in range(total_batches):
            sharded_ds = ds.shard(num_shards=total_batches, index=i, contiguous=True).with_format('numpy')
            ids = np.concatenate(sharded_ds["ids"])
            arr[idx: idx + len(ids)] = ids
            idx += len(ids)
            
        arr.flush()
            
            
if __name__ == "__main__":
    main()
