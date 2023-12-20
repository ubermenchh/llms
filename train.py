import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

from data import get_batch, decode, encode
from model import GPTLanguageModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 32
block_size = 128
max_iters = 500
learning_rate = 3e-4
eval_iters = 100
n_embd = 384
n_head = 4
n_layer = 4
dropout = 0.2

with open('sherlock.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))

vocab_size = len(chars)

@torch.inference_mode()
def estimate_loss():
    out = {}
    model.eval()

    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = GPTLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % 10 == 0:
        losses = estimate_loss()
        print(f"Step: {iter} | Train Loss: {losses['train']:.3f} | Test Loss: {losses['test']:.3f}")

    xb, yb = get_batch('train')

    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


with open('gpt-model.pkl', 'wb') as f:
    pickle.dump(model, f)
print('Model Saved!!!')

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=4096)[0].tolist()))
