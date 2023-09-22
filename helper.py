import configparser
from data.char_shake_dataset import CharShakeDataset
from model.bigram_model import BigramModel
from model.baby_gpt_model import BabyGPT
import os, errno


def parse_dataset_config(config: configparser.ConfigParser):
    dataset_name = config["Dataset"]["dataset_name"]
    print(f"dataset_name: {dataset_name}")
    tokenizer_type = config["Tokenizer"]["tokenizer_type"]
    print(f"tokenizer_type: {dataset_name}")
    
    if str(dataset_name).strip() == "shakespeare" and str(tokenizer_type).strip() == "char":
        m_dataset = CharShakeDataset(path_to_file="./data/input.txt")
    else:
        raise Exception("Cannot Parse Config")
    return m_dataset

def parse_model_config(config: configparser.ConfigParser, vocab_size: int):
    model_type = config["Model"]["model_type"]
    print(f"model_type: {model_type}")
    
    if model_type == "bigram":
        model = BigramModel(vocab_size=vocab_size)
    elif model_type == "baby_gpt":
        model = BabyGPT(config=config, vocab_size=vocab_size)
    else:
        raise Exception("Cannot Parse Config")
    return model


def create_directory(dir_path: str):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise