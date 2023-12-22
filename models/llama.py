import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 32
d_model = 512
n_heads = 8
context_window = 16
n_layers = 4

class RMSNorm(nn.Module):
    def __init__(self, layer_shape, eps=1e-8, bias=False):
        super().__init__()
        self.register_buffer("scale", nn.Parameter(torch.ones(layer_shape)))

    def forward(self, x):
        ff_rms = torch.linalg.norm(x, dim=(1, 2)) * x[0].numel()**-0.5
        raw = x / ff_rms.unsqueeze(-1).unsqueeze(-1)
        return self.scale[:x.shape[1], :].unsqueeze(0) * raw 


class SwiGLU(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear_gate = nn.Linear(size, size)
        self.linear = nn.Linear(size, size)
        self.beta = torch.randn(1, requires_grad=True)

        self.beta = nn.Parameter(torch.ones(1))
        self.register_parameter('beta', self.beta)

    def forward(self, x):
        swish_gate = self.linear_gate(x) * torch.sigmoid(self.beta * self.linear_gate(x))
        out = swish_gate * self.linear(x)
        return out

class RoPEMaskedAttentionHead(nn.Module):
    def __init__(self, d_model, context_window):
        super().__init__()
        self.w_q = nn.Linear(d_model, d_model, bias=False).to(device)
        self.w_k = nn.Linear(d_model, d_model, bias=False).to(device)
        self.w_v = nn.Linear(d_model, d_model, bias=False).to(device)
    
    @staticmethod
    def get_rotary_matrix(context_window, embedding_dim):
        R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)
        for position in range(context_window):
            for i in range(embedding_dim // 2):
                theta = 10000. ** (-2 * (i - 1) / embedding_dim)
                m_theta = position * theta
                R[position, 2*i, 2*i] = np.cos(m_theta)
                R[position, 2*i, 2*i+1] = -np.sin(m_theta)
                R[position, 2*i+1, 2*i] = np.sin(m_theta)
                R[position, 2*i+1, 2*i+1] = np.cos(m_theta)
        return R

    def set_rotary_matrix(self):
        self.R = self.get_rotary_matrix(context_window, d_model).to(device)

    def forward(self, x, return_attn_weights=False):
        x = x.to(device)
        b, m, d = x.shape

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q_rotated = (torch.bmm(q.transpose(0, 1), self.R[:m])).transpose(0, 1)
        k_rotated = (torch.bmm(k.transpose(0, 1), self.R[:m])).transpose(0, 1)

        activations = F.scaled_dot_product_attention(q_rotated, k_rotated, v, dropout_p=0.1)

        if return_attn_weights:
            attn_mask = torch.tril(torch.ones((m, m)), diagonal=0).to(device)
            attn_weights = torch.bmm(q_rotated, k_rotated.transpose(1, 2)) / np.sqrt(d) + attn_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            return activations, attn_weights
        return activations

class RoPEMaskedMultiheadAttention(nn.Module):
    def __init__(self, n_heads, d_model, context_window):
        super().__init__()
        self.heads = nn.ModuleList([RoPEMaskedAttentionHead(d_model, context_window) for _ in range(n_heads)])
        for head in self.heads:
            head.set_rotary_matrix()
        self.linear = nn.Linear(n_heads * d_model, d_model)
        self.dropout = nn.Dropout(.2)

    def forward(self, x):
        heads = [h(x) for h in self.heads]
        x = torch.cat(heads, dim=-1)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class LlamaBlock(nn.Module):
    def __init__(self, context_window, d_model, n_heads):
        super().__init__()
        self.rms = RMSNorm((context_window, d_model))
        self.attention = RoPEMaskedMultiheadAttention(n_heads, d_model, context_window)
        self.feedforward = nn.Sequential(
                nn.Linear(d_model, d_model),
                SwiGLU(d_model)
        )

    def forward(self, x):
        x = self.rms(x)
        x = x + self.attention(x)

        x = self.rms(x)
        x = x + self.feedforward(x)
        return x

class Llama(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, context_window):
        super().__init__()
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.llama_blocks = nn.Sequential(
                OrderedDict([(f"llama_{i}", LlamaBlock(context_window, d_model, n_heads)) for i in range(n_layers)])
        )
        self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model),
                SwiGLU(d_model),
                nn.Linear(d_model, vocab_size)
        )

    def forward(self, idx, targets=None):
        x = self.embeddings(idx)
        x = self.llama_blocks(x)
        logits = self.ffn(x)

        if targets is None:
            return logits, None
        else:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            return logits, loss 
