import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle, time
import pandas as pd
import numpy as np

from data import get_batch, decode, encode, get_batches
from model import GPTLanguageModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 32
block_size = 128
epochs = 1000
learning_rate = 3e-4
log_interval = 10
n_embd = 384
n_head = 4
n_layer = 4
dropout = 0.2
context_window = 16

with open('sherlock.txt', 'r', encoding='utf-8') as f:
    text = f.read().lower()
    chars = sorted(list(set(text)))
dataset = torch.tensor(encode(text), dtype=torch.int8)

vocab_size = len(chars)

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

model = GPTLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

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
        logits = model(idx[:, -context_window:])
        last_time_step_logits = logits[:, -1, :]
        p = F.softmax(last_time_step_logits, dim=-1)
        idx_next = torch.multinomial(p, num_samples=1)
        idx = torch.cat([idx, idx_next], dim=-1)

    return [decode(x) for x in idx.tolist()]

train(model, optimizer, scheduler=None, print_logs=True)
print(generate(model)[0])

with open('gpt-model.pkl', 'wb') as f:
    pickle.dump(model, f)
print('Model Saved!!!')
