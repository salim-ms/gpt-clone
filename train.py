import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
from utils.config_parser import parse_config
from helper import parse_dataset_config, parse_model_config, create_directory


torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, 
                    help="config name only, can be found under configurations/")


# create a path to store trained model
MODEL_SAVED_PATH = "model_saved.pth"

# esitmate loss during training based on training and eval loss
@torch.no_grad()
def estimate_loss(model, dataset, batch_size, max_context_length, eval_iterations):
    out = {}
    # entry eval model
    model.eval()
    # for both training and evaluation set, compute average loss
    for split in ['train', 'eval']:
        # create a 1d tensor to store values of losses, so we can average it after
        losses = torch.zeros(eval_iterations)
        for i in range(eval_iterations):
            # get data
            x, y = dataset.get_batch(split, batch_size, max_context_length)
            logits, loss = model(x, y)
            losses[i] = loss.item()
        
        avg_loss = losses.mean()
        out[split] = avg_loss
    
    print(f'Train Loss: {out["train"]}, Eval Loss: {out["eval"]}')
    # when done evaluation, switch back to train mode
    model.train()
    return out
    

if __name__ == "__main__":
    # parse the config
    args = parser.parse_args()
    config = parse_config(args.config)
    
    BATCH_SIZE = config["Model"].getint("batch_size")
    MAX_CONTEXT_LENGTH= config["Model"].getint("max_context_length")
    
    TRAINING_STEPS = config["Training"].getint("training_steps")
    EVAL_ITERATIONS = config["Training"].getint("eval_iterations")
    LEARNING_RATE = config["Training"].getfloat("learning_rate")
    
    # Dataset & tokenizer configs
    m_dataset = parse_dataset_config(config)
    print(f"Vocab Size {m_dataset.vocab_size}")
    
    # Model Config
    m_model = parse_model_config(config, int(m_dataset.vocab_size))
    
    # move model to device
    m_model = m_model.to(device)
    
    train_dataset, eval_dataset = m_dataset.load_datasets()
    # create an optimizer
    optimizer = torch.optim.AdamW(m_model.parameters(), lr=LEARNING_RATE)
    
    # output total number of parameters
    trainable_parameters = sum(p.numel() for p in m_model.parameters()) / 1e6
    print(f"{trainable_parameters} M Parameters")
    
    # training loop
    running_loss = 0
    for i in range(TRAINING_STEPS):
        # get training data batch
        x, y = m_dataset.get_batch(split="train", batch_size=BATCH_SIZE, max_context_length=MAX_CONTEXT_LENGTH)
        # forward
        logits, loss = m_model(x, y)
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

        if i % 200 == 199:
            estimate_loss(m_model, m_dataset, BATCH_SIZE, MAX_CONTEXT_LENGTH, EVAL_ITERATIONS)
        
    # save model
    model_directory_path = f'/workspace/model_outputs/{config["Model"]["model_type"]}'
    create_directory(model_directory_path)
    trained_model_path = f'/workspace/model_outputs/{config["Model"]["model_type"]}/{MODEL_SAVED_PATH}'
    torch.save(m_model.state_dict(), trained_model_path)