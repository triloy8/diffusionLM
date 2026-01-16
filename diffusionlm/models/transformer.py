import torch
import torch.nn as nn

from diffusionlm.models.layers import Embedding, RMSNorm, SwiGLU, Linear
from diffusionlm.models.attention import MultiheadSelfAttentionRoPE


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        d_ff: int,
        attention_backend: str = "custom",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
        self.attn = MultiheadSelfAttentionRoPE(
            d_model,
            num_heads,
            max_seq_len,
            theta,
            attention_backend=attention_backend,
            device=device,
            dtype=dtype,
        )
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        token_positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.long)
        ln1x = self.ln1(x)
        x = x + self.attn(ln1x, token_positions, attention_mask=attention_mask)
        ln2x = self.ln2(x)
        x = x + self.ffn(ln2x)
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        attention_backend: str = "custom",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = torch.nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    context_length,
                    rope_theta,
                    d_ff,
                    attention_backend=attention_backend,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

    def forward(self, in_indices: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        output_seq = self.token_embeddings(in_indices)
        for layer in self.layers:
            output_seq = layer(output_seq, attention_mask=attention_mask)
        normed_output_seq = self.ln_final(output_seq)
        logits = self.lm_head(normed_output_seq)
        return logits
