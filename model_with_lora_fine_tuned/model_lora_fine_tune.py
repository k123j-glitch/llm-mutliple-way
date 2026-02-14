import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# 1. THE LORA LAYER
class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=16, alpha=32):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.lora_A = nn.Parameter(torch.zeros((in_features, rank)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(torch.zeros((rank, out_features)))

        # Freeze base weights
        self.base_layer.weight.requires_grad = False

    def forward(self, x):
        result = self.base_layer(x)
        adapter = (x @ self.lora_A @ self.lora_B) * self.scaling
        return result + adapter


# 2. RMSNORM
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed


# 3. TINYLLAMA BLOCK (The missing piece)
class TinyLlamaBlock(nn.Module):
    def __init__(self, dim, n_heads, r=16):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # Attention path (LoRA wrapped)
        self.wq = LoRALinear(nn.Linear(dim, dim, bias=False), rank=r)
        self.wk = LoRALinear(nn.Linear(dim, dim, bias=False), rank=r)
        self.wv = LoRALinear(nn.Linear(dim, dim, bias=False), rank=r)
        self.wo = LoRALinear(nn.Linear(dim, dim, bias=False), rank=r)

        # Feed Forward (SwiGLU) path (LoRA wrapped)
        self.w1 = LoRALinear(nn.Linear(dim, 5632, bias=False), rank=r)  # Gate
        self.w2 = LoRALinear(nn.Linear(5632, dim, bias=False), rank=r)  # Down
        self.w3 = LoRALinear(nn.Linear(dim, 5632, bias=False), rank=r)  # Up

        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x):
        # Attention + Residual
        h = x + self._attention_path(self.attention_norm(x))
        # MLP + Residual
        out = h + self._ffn_path(self.ffn_norm(h))
        return out

    def _attention_path(self, x):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        # Matmul (Context Mixing)
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal Mask
        mask = torch.triu(torch.ones(seqlen, seqlen, device=x.device), 1).bool()
        scores.masked_fill_(mask[None, None, :, :], float("-inf"))

        attn = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(attn, xv)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

    def _ffn_path(self, x):
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
        x = self.token_embedding(tokens)
        for layer in self.layers:
            x = layer(x)
        return self.output(self.norm(x))

    def load_pretrained(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("model.", "").replace("self_attn.q_proj", "wq.base_layer")
            new_key = new_key.replace("self_attn.k_proj", "wk.base_layer")
            new_key = new_key.replace("self_attn.v_proj", "wv.base_layer")
            new_key = new_key.replace("self_attn.o_proj", "wo.base_layer")
            new_key = new_key.replace("mlp.gate_proj", "w1.base_layer")
            new_key = new_key.replace("mlp.down_proj", "w2.base_layer")
            new_key = new_key.replace("mlp.up_proj", "w3.base_layer")
            new_key = new_key.replace("input_layernorm", "attention_norm")
            new_key = new_key.replace("post_attention_layernorm", "ffn_norm")
            if "embed_tokens" in key: new_key = "token_embedding.weight"
            if "lm_head.weight" in key: new_key = "output.weight"
            if "model.norm.weight" in key: new_key = "norm.weight"
            new_state_dict[new_key] = value

        self.load_state_dict(new_state_dict, strict=False)

    def merge_lora_weights(self):
        for module in self.modules():
            if isinstance(module, LoRALinear):
                # Apply LoRA math: W = W + (A @ B) * scaling
                # Note: We transpose A @ B to match Linear weight shape [out, in]
                delta_w = (module.lora_A @ module.lora_B).t() * module.scaling
                module.base_layer.weight.data += delta_w
                # Reset LoRA params
                module.lora_A.data.zero_()
                module.lora_B.data.zero_()

    def save_fine_tuned_checkpoint(self, path, only_lora=True):
        if only_lora:
            state_dict = {n: p for n, p in self.named_parameters() if "lora_" in n}
        else:
            state_dict = self.state_dict()
        torch.save(state_dict, path)