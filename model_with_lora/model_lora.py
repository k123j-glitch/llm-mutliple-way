import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# 1. THE PURE LORA LAYER
class LoRALinear(nn.Module):
    """
    Wraps a standard nn.Linear with LoRA matrices.
    Math: Output = (x @ W_base) + (x @ A @ B) * (alpha / rank)
    """

    def __init__(self, base_layer, rank=16, alpha=32):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # LoRA Matrices
        self.lora_A = nn.Parameter(torch.zeros((in_features, rank)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(torch.zeros((rank, out_features)))

        # FREEZE the base weight
        self.base_layer.weight.requires_grad = False

    def forward(self, x):
        # Base linear path
        result = self.base_layer(x)
        # LoRA adapter path: (x @ A) @ B
        adapter = (x @ self.lora_A @ self.lora_B) * self.scaling
        return result + adapter


# 2. RMSNorm (Standard Llama Normalization)
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed


# 3. TINYLLAMA BLOCK WITH LORA
class TinyLlamaBlock(nn.Module):
    def __init__(self, dim, n_heads, r=16):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # Attention Layers (LoRA-wrapped)
        self.wq = LoRALinear(nn.Linear(dim, dim, bias=False), rank=r)
        self.wk = LoRALinear(nn.Linear(dim, dim, bias=False), rank=r)
        self.wv = LoRALinear(nn.Linear(dim, dim, bias=False), rank=r)
        self.wo = LoRALinear(nn.Linear(dim, dim, bias=False), rank=r)

        # FeedForward Layers (LoRA-wrapped)
        # 5632 is the intermediate size for TinyLlama 1.1B
        self.w1 = LoRALinear(nn.Linear(dim, 5632, bias=False), rank=r)  # Gate
        self.w2 = LoRALinear(nn.Linear(5632, dim, bias=False), rank=r)  # Down
        self.w3 = LoRALinear(nn.Linear(dim, 5632, bias=False), rank=r)  # Up

        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x):
        # Self Attention + Residual
        h = x + self._attention_path(self.attention_norm(x))
        # Feed Forward + Residual
        out = h + self._ffn_path(self.ffn_norm(h))
        return out

    def _attention_path(self, x):
        bsz, seqlen, _ = x.shape

        # Get Q, K, V
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Reshape for multi-head: [Batch, Heads, Seq, Head_Dim]
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention: [Batch, Heads, Seq, Seq]
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal Mask (Corrected for 4D Tensors)
        mask = torch.triu(torch.ones(seqlen, seqlen, device=x.device), diagonal=1).bool()
        mask = mask.view(1, 1, seqlen, seqlen)  # Expansion for Batch and Heads
        scores.masked_fill_(mask, float("-inf"))

        # Softmax and context mixing
        attn = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(attn, xv)  # [Batch, Heads, Seq, Head_Dim]

        # Restore shape: [Batch, Seq, Dim]
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

    def _ffn_path(self, x):
        # SwiGLU: (SiLU(W1 * x) * (W3 * x)) * W2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# 4. FULL TINYLLAMA MODEL
class TinyLlama(nn.Module):
    def __init__(self, vocab_size=32000, dim=2048, n_layers=22, n_heads=32, r=16):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TinyLlamaBlock(dim, n_heads, r) for _ in range(n_layers)])
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens):
        # 1. Embed tokens
        x = self.token_embedding(tokens)

        # 2. Run through Transformer blocks
        for layer in self.layers:
            x = layer(x)

        # 3. Final norm and projection to vocab
        return self.output(self.norm(x))