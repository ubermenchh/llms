import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle, time
import pandas as pd
import numpy as np
import argparse

from models.gpt import GPTLanguageModel
from models.llama import Llama 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument("--file", default='sherlock.txt', type=str, help='the text file to train the model on (default is sherlock.txt)')
parser.add_argument("--model", default='gpt', type=str, help='select a model: gpt(default), llama')

batch_size = 32
epochs = 1000
learning_rate = 3e-4
log_interval = 100 
n_embd = 384 
n_heads = 8
n_layers = 4
dropout = 0.2  
context_window = 16
d_model = 512

args = parser.parse_args()

with open(args.file, 'r', encoding='utf-8') as f:
    text = f.read().lower()
    chars = sorted(list(set(text)))

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for ch, i in enumerate(chars)} 

def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

dataset = torch.tensor(encode(text), dtype=torch.int8)
vocab_size = len(chars)

def get_batches(data, split, batch_size, context_window):
    train = data[:int(0.8 * len(data))]
    val = data[int(0.8 * len(data)):int(0.9 * len(data))]
    test = data[int(0.9 * len(data)):]

    batch_data = train
    if split == 'val': batch_data = val
    if split == 'test': batch_data = test 

    ix = torch.randint(0, batch_data.size(0) - context_window - 1, (batch_size,))
    x = torch.stack([batch_data[i:i+context_window] for i in ix]).long()
    y = torch.stack([batch_data[i+1:i+context_window+1] for i in ix]).long()

    return x, y

@torch.inference_mode()
def evaluate_loss(model):
    out = {}
    model.eval()

    for split in ['train', 'val', 'test']:
        losses = []
        for _ in range(10):
            xb, yb = get_batches(dataset, split, batch_size, context_window)
            xb, yb = xb.to(device), yb.to(device)

            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = np.mean(losses)
    model.train()
    return out

def train(model, optimizer, scheduler=None, print_logs=False):
    losses = []
    start_time = time.time()
    
    for epoch in range(epochs):
        optimizer.zero_grad()

        xs, ys = get_batches(dataset, 'train', batch_size, context_window)
        xs, ys = xs.to(device), ys.to(device)

        logits, loss = model(xs, ys)
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        if epoch % log_interval == 0:
            batch_time = time.time() - start_time
            x = evaluate_loss(model)
            losses += [x]
            if print_logs:
                print(f"Epoch {epoch} | Val Loss: {x['val']:.3f} | Time {batch_time:.3f} | ETA in seconds {batch_time * (epochs - epoch) / log_interval:.3f}")
            start_time = time.time()

            if scheduler:
                print("lr: ", scheduler.get_lr())

    print("Validation Loss: ", losses[-1]['val'])
    return pd.DataFrame(losses).plot()

def generate(model, max_new_tokens=1024):
    idx = torch.zeros(5, 1).long().to(device)
    
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -context_window:])
        last_time_step_logits = logits[:, -1, :]
        p = F.softmax(last_time_step_logits, dim=-1)
        idx_next = torch.multinomial(p, num_samples=1)
        idx = torch.cat([idx, idx_next], dim=-1)

    return [decode(x) for x in idx.tolist()]

if args.model == 'gpt':
    model = GPTLanguageModel(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    train(model, optimizer)
    print(generate(model)[0])

if args.model == 'llama':
    model = Llama(vocab_size, d_model, n_layers, n_heads, context_window).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), betas=(.9, .95), weight_decay=.1, eps=1e-9, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300, eta_min=1e-5)

    train(model, optimizer, scheduler)
    print(generate(model)[0])


# with open('gpt-model.pkl', 'wb') as f:
#    pickle.dump(model, f)
# print('Model Saved!!!')
