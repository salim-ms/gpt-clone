import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
from utils.config_parser import parse_config
from helper import parse_dataset_config, parse_model_config

torch.manual_seed(1337)

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="config name only, can be found under configurations/")
parser.add_argument("--max_tokens", default=200, required=False, help="number of tokens to generate")


MODEL_SAVED_PATH = "model_saved.pth"

if __name__ == "__main__":
    args = parser.parse_args()
    config = parse_config(args.config)
    
    BATCH_SIZE = config["Model"].getint("batch_size")
    CONTEXT_LENGTH= config["Model"].getint("context_length")
    
    # parse and load
    m_dataset = parse_dataset_config(config)
    m_model = parse_model_config(config, m_dataset.vocab_size)
    RESTORED_MODEL_PATH = f'/workspace/model_outputs/{config["Model"]["model_type"]}/{MODEL_SAVED_PATH}'
    # load trained model
    m_model.load_state_dict(torch.load(RESTORED_MODEL_PATH))
    m_model.eval()
    
    generated_text = m_model.generate(max_new_tokens=int(args.max_tokens))[0]
    print(m_dataset.decode(generated_text))