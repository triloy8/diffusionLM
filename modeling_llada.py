from typing import (
    Optional,
    Tuple,
)

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

class Dropout(nn.Dropout):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0:
            return input
        else:
            return F.dropout(input, self.p, self.training, self.inplace)

class RMSLayerNorm(nn.Module):
    """
    RMS layer norm, a simplified :class:`LayerNorm` implementation
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device = "cuda",
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device))

        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        og_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = x.to(og_dtype)

        return self.weight * x

class RotaryEmbedding(nn.Module):
    def __init__(
            self,
            rope_theta:float, 
            d_model: int,
            n_heads: int,
            max_sequence_length: int, 
            device: torch.device
        ):
        super().__init__()
        self.rope_theta = rope_theta
        self.d_model = d_model
        self.n_heads = n_heads
        self.get_rotary_embedding(max_sequence_length, device)

    def get_rotary_embedding(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:

        dim = self.d_model // self.n_heads
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim))
        seq = torch.arange(seq_len, device=device, dtype=torch.float)
        freqs = einsum("i , j -> i j", seq, inv_freq)
        positions = torch.cat((freqs, freqs), dim=-1)
        pos_sin, pos_cos = positions.sin()[None, None, :, :], positions.cos()[None, None, :, :]

        return pos_sin, pos_cos

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)

        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return ((t * pos_cos) + (self.rotate_half(t) * pos_sin)).to(t.dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        query_len, key_len = q.shape[-2], k.shape[-2]  # could be different if layer_past not None
        pos_sin, pos_cos = self.get_rotary_embedding(key_len, q.device)
        pos_sin = pos_sin.type_as(q)
        pos_cos = pos_cos.type_as(q)
        q = self.apply_rotary_pos_emb(
            pos_sin[:, :, key_len - query_len : key_len, :],
            pos_cos[:, :, key_len - query_len : key_len, :],
            q,
        )
        k = self.apply_rotary_pos_emb(pos_sin, pos_cos, k)

        return q.type_as(q), k.type_as(k)

class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

    @property
    def output_multiplier(self) -> float:
        return 0.5

class LLaDASequentialBlock(nn.Module):
    """
    This is a typical transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection).
    """

    def __init__(
            self, 
            layer_id: int, 
            mlp_ratio: int,
            d_model: int,
            n_heads: int, 
            residual_dropout: float,
            attention_dropout: float,
            rope_theta: float,
            max_sequence_length: int,
            device: torch.device,
        ):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = (
            mlp_ratio * d_model
        )
        assert d_model % n_heads == 0

        self.attention_dropout = attention_dropout
        self.n_heads = n_heads

        # Dropout.
        self.dropout = Dropout(residual_dropout)

        # Activation function.
        self.act = SwiGLU()
        assert (self.act.output_multiplier * self.hidden_size) % 1 == 0

        # Attention output projection.
        self.attn_out = nn.Linear(
            d_model, d_model, bias=False, device=device
        )

        # Feed-forward output projection.
        self.ff_out = nn.Linear(
            int(self.act.output_multiplier * self.hidden_size),
            d_model,
            bias=False,
            device=device,
        )
        self.ff_out._is_residual = True

        # Rotary embeddings.
        self.rotary_emb = RotaryEmbedding(rope_theta=rope_theta, d_model=d_model, n_heads=n_heads, max_sequence_length=max_sequence_length, device=device)

        # Layer norms.
        self.attn_norm = RMSLayerNorm(d_model=d_model, device=device)
        self.ff_norm = RMSLayerNorm(d_model=d_model, device=device)
        
        # Attention input projection. Projects x -> (q, k, v)
        self.fused_dims = (
            d_model,
            d_model,
            d_model,
        )
        self.att_proj = nn.Linear(
            d_model, sum(self.fused_dims), bias=False, device=device
        )
        
        # Feed-forward input projection.
        self.ff_proj = nn.Linear(
            d_model, self.hidden_size, bias=False, device=device
        )

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = q.size()  # batch size, sequence length, d_model

        # Move head forward to be next to the batch dim.
        # shape: (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        q, k = self.rotary_emb(q, k)

        # Get the attention scores.
        # shape: (B, nh, T, hs)
        att = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            is_causal=False,
        )

        # Re-assemble all head outputs side-by-side.
        att = att.transpose(1, 2).contiguous().view(B, T, C)

        # Apply output projection.
        return self.attn_out(att)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        q, k, v = self.att_proj(self.attn_norm(x)).split(self.fused_dims, dim=-1)

        att = self.attention(q, k, v)

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x
        x = self.ff_norm(x)
        x = self.ff_proj(x)

        x = self.act(x)
        x = self.ff_out(x)
        x = self.dropout(x)
        x = og_x + x

        return x

class LLaDAModel(nn.Module):
    def __init__(
            self, 
            mlp_ratio: int,
            d_model: int,
            n_heads: int, 
            residual_dropout: float,
            attention_dropout: float,
            rope_theta: float,
            max_sequence_length: int,
            vocab_size: int,
            embedding_dropout: float,
            n_layers: int,
            device: torch.device,
        ):
        super().__init__()
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    vocab_size, d_model, device=device
                ),
                emb_drop=Dropout(embedding_dropout),
                ln_f=RMSLayerNorm(d_model=d_model, device=device),
            )
        )

        blocks = [
            LLaDASequentialBlock(
                layer_id=i, 
                mlp_ratio=mlp_ratio,
                d_model=d_model,
                n_heads=n_heads, 
                residual_dropout=residual_dropout,
                attention_dropout=attention_dropout,
                rope_theta=rope_theta,
                max_sequence_length=max_sequence_length,
                device=device,
            ) 
            for i in range(n_layers)
        ]
        self.transformer.update({"blocks": nn.ModuleList(blocks)})

        self.transformer.update(
            {
                "ff_out": nn.Linear(
                    d_model,
                    vocab_size,
                    bias=False,
                    device=device,
                )
            }
        )

    @property
    def device(self) -> torch.device:
        device: torch.device = self.transformer.wte.weight.device  # type: ignore
        return device

    def forward(
        self,
        input_ids: torch.LongTensor,
        last_logits_only: bool = False,
    ) -> torch.Tensor:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param last_logits_only: If `True`, only compute the logits for the last token of each sequence.
            This can speed up decoding when you only care about the next token.
        """

        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.wte(input_ids)

        # Add input + positional embeddings and apply dropout.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.emb_drop(x)  # type: ignore

        for block_idx, block in enumerate(self.transformer.blocks):
            x = block(x)

        if last_logits_only:
            # shape: (batch_size, 1, d_model)
            x = x[:, -1, :].unsqueeze(1)

        # Apply final layer norm.
        # shape: (batch_size, seq_len or 1, d_model)
        x = self.transformer.ln_f(x)  # type: ignore

        logits = self.transformer.ff_out(x)  # type: ignore

        return logits