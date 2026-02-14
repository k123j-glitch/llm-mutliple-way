import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    """Llama uses RMSNorm instead of LayerNorm for speed and stability."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # RMS = sqrt(mean(x^2) + eps)
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm_x * self.weight


class RotaryEmbedding(nn.Module):
    """RoPE helps the model understand the position of words in a circle/rotation."""

    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary_pos_emb(x, cos, sin):
    """Rotates the Query and Key vectors."""
    # Split x into two halves for rotation math
    x1, x2 = x.chunk(2, dim=-1)
    # Standard 2D rotation matrix application: [x1*cos - x2*sin, x1*sin + x2*cos]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)


class TinyLlamaBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # Attention layers
        self.attention_norm = RMSNorm(dim)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        # Feed-forward layers (SwiGLU)
        self.ffn_norm = RMSNorm(dim)
        self.w1 = nn.Linear(dim, dim * 4, bias=False)  # Gate
        self.w2 = nn.Linear(dim * 4, dim, bias=False)  # Down
        self.w3 = nn.Linear(dim, dim * 4, bias=False)  # Up

    def forward(self, x, cos, sin, mask=None):
        # 1. Self-Attention
        h = self.attention_norm(x)
        q, k, v = self.wq(h), self.wk(h), self.wv(h)

        # Reshape for multi-head
        q = q.view(q.shape[0], q.shape[1], self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # Scaled Dot-Product Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        # Concatenate heads and project out
        out = out.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1)
        x = x + self.wo(out)  # Residual connection

        # 2. Feed-Forward (SwiGLU)
        h = self.ffn_norm(x)
        # SwiGLU: (Swish(W1*x) * W3*x) * W2
        swish_gate = F.silu(self.w1(h))
        x = x + self.w2(swish_gate * self.w3(h))  # SwiGLU + Residual

        return x


class TinyLlama(nn.Module):
    def __init__(self, vocab_size, dim=512, n_layers=6, n_heads=8):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TinyLlamaBlock(dim, n_heads) for _ in range(n_layers)])
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.rope = RotaryEmbedding(dim // n_heads)

    def forward(self, tokens):
        _bsz, seq_len = tokens.shape
        h = self.token_embedding(tokens)

        # Compute RoPE once for all layers
        cos, sin = self.rope(seq_len, h.device)

        # Create Causal Mask (so tokens can't look at the future)
        mask = torch.full((seq_len, seq_len), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=1)

        for layer in self.layers:
            h = layer(h, cos, sin, mask)

        return self.output(self.norm(h))