import torch
import torch.nn as nn
import torch.nn.functional as F
import mmap
import random
import pickle

file_path = 'sherlock.txt'

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))

vocab_size = len(chars)

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

# print(text[:1000])

def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

#print(encode('Hello There'))
#print(decode([29, 53, 60, 60, 63, 1, 41, 56, 53, 66, 53]))

batch_size = 32
block_size = 128
max_iters = 500
learning_rate = 3e-4
eval_iters = 100
n_embd = 384
n_head = 4
n_layer = 4
dropout = 0.2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_random_chunk():
    with open(file_path, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size * batch_size)

            mm.seek(start_pos)
            block = mm.read(block_size * batch_size - 1)

            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            data = torch.tensor(encode(decoded_block), dtype=torch.long)

    return data

def get_batch(split):
    data = get_random_chunk()
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


