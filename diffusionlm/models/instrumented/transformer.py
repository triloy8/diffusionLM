import torch
import torch.nn as nn
from profiling import nvtx

from diffusionlm.models.instrumented.layers import Embedding, RMSNorm, SwiGLU, Linear
from diffusionlm.models.instrumented.attention import MultiheadSelfAttentionRoPE


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
        self.attn = MultiheadSelfAttentionRoPE(d_model, num_heads, max_seq_len, theta, device, dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.long)
        # Attention path
        with nvtx.range("model/block/attn"):
            if nvtx.enabled("fine"):
                with nvtx.range("model/block/ln1"):
                    ln1x = self.ln1(x)
            else:
                ln1x = self.ln1(x)
            x = x + self.attn(ln1x, token_positions)
        # FFN path
        with nvtx.range("model/block/ffn"):
            if nvtx.enabled("fine"):
                with nvtx.range("model/block/ln2"):
                    ln2x = self.ln2(x)
            else:
                ln2x = self.ln2(x)
            x = x + self.ffn(ln2x)
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, device=None, dtype=None):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = torch.nn.ModuleList([TransformerBlock(d_model, num_heads, context_length, rope_theta, d_ff, device, dtype) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        with nvtx.range("model/forward"):
            with nvtx.range("model/embedding"):
                output_seq = self.token_embeddings(in_indices)
            for i, layer in enumerate(self.layers):
                with nvtx.range(f"model/layer[{i}]"):
                    output_seq = layer(output_seq)
            if nvtx.enabled("fine"):
                with nvtx.range("model/ln_final"):
                    normed_output_seq = self.ln_final(output_seq)
            else:
                normed_output_seq = self.ln_final(output_seq)
            with nvtx.range("model/lm_head"):
                logits = self.lm_head(normed_output_seq)
            return logits
