import torch
import torch.nn as nn

from transformerlm.models.layers import Embedding, RMSNorm, SwiGLU, Linear
from transformerlm.models.attention import (
    MultiheadSelfAttentionRoPE,
    MultiheadCrossAttentionRoPE,
    MultiheadSelfAttentionRoPE2D,
    MultiheadCrossAttentionRoPE2D,
)


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


class TransformerImageBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        max_height: int | None,
        max_width: int | None,
        theta: float,
        d_ff: int,
        attention_backend: str = "custom",
        use_rope_2d: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
        self.use_rope_2d = use_rope_2d
        if self.use_rope_2d:
            if max_height is None or max_width is None:
                raise ValueError("max_height/max_width must be provided when use_rope_2d is True")
            self.self_attn = MultiheadSelfAttentionRoPE2D(
                d_model,
                num_heads,
                max_height,
                max_width,
                theta,
                attention_backend=attention_backend,
                device=device,
                dtype=dtype,
            )
            self.cross_attn = MultiheadCrossAttentionRoPE2D(
                d_model,
                num_heads,
                max_height,
                max_width,
                theta,
                attention_backend=attention_backend,
                device=device,
                dtype=dtype,
            )
        else:
            self.self_attn = MultiheadSelfAttentionRoPE(
                d_model,
                num_heads,
                max_seq_len,
                theta,
                attention_backend=attention_backend,
                device=device,
                dtype=dtype,
            )
            self.cross_attn = MultiheadCrossAttentionRoPE(
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
        self.ln3 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
        context: torch.Tensor,
        context_token_positions: torch.Tensor,
        row_positions: torch.Tensor | None = None,
        col_positions: torch.Tensor | None = None,
        context_row_positions: torch.Tensor | None = None,
        context_col_positions: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ln1x = self.ln1(x)
        if self.use_rope_2d:
            if row_positions is None or col_positions is None:
                raise ValueError("row_positions/col_positions must be provided when use_rope_2d is True")
            x = x + self.self_attn(ln1x, row_positions, col_positions, attention_mask=attention_mask)
        else:
            x = x + self.self_attn(ln1x, token_positions, attention_mask=attention_mask)
        ln2x = self.ln2(x)
        if self.use_rope_2d:
            if context_row_positions is None or context_col_positions is None:
                raise ValueError("context_row_positions/context_col_positions must be provided when use_rope_2d is True")
            x = x + self.cross_attn(
                ln2x,
                context,
                row_positions,
                col_positions,
                context_row_positions,
                context_col_positions,
                attention_mask=None,
            )
        else:
            x = x + self.cross_attn(
                ln2x,
                context,
                token_positions,
                context_token_positions,
                attention_mask=None,
            )
        ln3x = self.ln3(x)
        x = x + self.ffn(ln3x)
        return x


class TransformerImage(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        label_vocab_size: int,
        attention_backend: str = "custom",
        image_height: int | None = None,
        image_width: int | None = None,
        use_rope_2d: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.context_length = context_length
        self.use_rope_2d = use_rope_2d
        if self.use_rope_2d:
            if image_height is None or image_width is None:
                side = int(context_length ** 0.5)
                if side * side != context_length:
                    raise ValueError("image_height/image_width must be set when context_length is not a square")
                image_height = side
                image_width = side
        self.image_height = image_height
        self.image_width = image_width
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.label_embeddings = Embedding(label_vocab_size, d_model, device, dtype)
        self.layers = torch.nn.ModuleList(
            [
                TransformerImageBlock(
                    d_model,
                    num_heads,
                    context_length,
                    self.image_height,
                    self.image_width,
                    rope_theta,
                    d_ff,
                    attention_backend=attention_backend,
                    use_rope_2d=self.use_rope_2d,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

    def forward(
        self,
        in_indices: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context is None:
            raise ValueError("context must be provided for TransformerImage")
        output_seq = self.token_embeddings(in_indices)
        context_emb = self.label_embeddings(context).unsqueeze(-2)
        token_positions = torch.arange(output_seq.shape[-2], device=output_seq.device, dtype=torch.long)
        context_token_positions = torch.arange(context_emb.shape[-2], device=output_seq.device, dtype=torch.long)
        row_positions = None
        col_positions = None
        context_row_positions = None
        context_col_positions = None
        if self.use_rope_2d:
            if self.image_height is None or self.image_width is None:
                raise ValueError("image_height/image_width must be set when use_rope_2d is True")
            seq_len = output_seq.shape[-2]
            expected = int(self.image_height) * int(self.image_width)
            if seq_len != expected:
                raise ValueError(f"sequence length {seq_len} does not match image_height*image_width {expected}")
            row_positions = torch.arange(int(self.image_height), device=output_seq.device, dtype=torch.long)
            row_positions = row_positions.repeat_interleave(int(self.image_width))
            col_positions = torch.arange(int(self.image_width), device=output_seq.device, dtype=torch.long)
            col_positions = col_positions.repeat(int(self.image_height))
            context_row_positions = torch.zeros(context_emb.shape[-2], device=output_seq.device, dtype=torch.long)
            context_col_positions = torch.zeros(context_emb.shape[-2], device=output_seq.device, dtype=torch.long)
        for layer in self.layers:
            output_seq = layer(
                output_seq,
                token_positions,
                context_emb,
                context_token_positions,
                row_positions=row_positions,
                col_positions=col_positions,
                context_row_positions=context_row_positions,
                context_col_positions=context_col_positions,
                attention_mask=attention_mask,
            )
        normed_output_seq = self.ln_final(output_seq)
        logits = self.lm_head(normed_output_seq)
        return logits
