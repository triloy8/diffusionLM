from einops import einsum
import torch
import torch.nn as nn
from profiling import nvtx


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        mean = 0.0
        std = 2 / (in_features + out_features)
        a = mean - 3 * std
        b = mean + 3 * std
        nn.init.trunc_normal_(self.weight, mean=mean, std=std, a=a, b=b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if nvtx.enabled("verbose"):
            with nvtx.range("linear"):
                y = einsum(self.weight, x, "out_features in_features, ... in_features -> ... out_features")
                return y
        else:
            y = einsum(self.weight, x, "out_features in_features, ... in_features -> ... out_features")
            return y


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        with nvtx.range("embedding/lookup"):
            embeds = self.weight[token_ids]
            return embeds


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.weight = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if nvtx.enabled("fine"):
            with nvtx.range("rmsnorm/compute"):
                in_dtype = x.dtype
                x = x.to(torch.float32)
                rms = torch.sqrt(torch.mean(x ** 2, dim=-1) + self.eps).unsqueeze(-1)
                x = (1 / rms) * (x * self.weight)
                return x.to(in_dtype)
        else:
            in_dtype = x.dtype
            x = x.to(torch.float32)
            rms = torch.sqrt(torch.mean(x ** 2, dim=-1) + self.eps).unsqueeze(-1)
            x = (1 / rms) * (x * self.weight)
            return x.to(in_dtype)


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with nvtx.range("ffn"):
            if nvtx.enabled("fine"):
                with nvtx.range("ffn/w1"):
                    w1x = self.w1(x)
                with nvtx.range("ffn/w3"):
                    w3x = self.w3(x)
                with nvtx.range("ffn/silu"):
                    silu = w1x * torch.sigmoid(w1x)
                with nvtx.range("ffn/glu"):
                    glu = silu * w3x
                with nvtx.range("ffn/w2"):
                    w2x = self.w2(glu)
                return w2x
            else:
                w1x = self.w1(x)
                w3x = self.w3(x)
                silu = w1x * torch.sigmoid(w1x)
                glu = silu * w3x
                w2x = self.w2(glu)
                return w2x
