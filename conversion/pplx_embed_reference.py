"""Golden fp32 reference oracle for pplx-embed (Perplexity) on CoreML-LLM.

This is the ground truth every CoreML fidelity comparison is measured against:

    HF fp32 forward  ->  masked-mean (plain) / pool_matrix matmul (context)
                     ->  st_quantize int8 / binary / ubinary

CRITICAL — quantizer parity. We mirror the model's own ``st_quantize.py`` EXACTLY:

    int8   = clamp(round(tanh(x) * 127), -128, 127)   # torch.round = HALF-TO-EVEN
    binary = where(x >= 0, +1.0, -1.0)                 # float32 +/-1
    ubinary= packbits(x >= 0)                          # uint8 [..., dim/8]

NOTE the two traps, both deliberately followed here:
  * **torch.round** (round-half-to-even / banker's), NOT the paper's / the parallel
    effort's HALF-UP ``floor(127*tanh+0.5)`` — they differ by +/-1 at exact halves.
  * **qmin = -128** (not -127). Note -128 is never actually reached: tanh(x)*127 in
    (-127, 127), so round() bottoms out at -127; the -128 clamp is purely defensive.

Pooling matches the reference ``modeling.py``:
  * plain   : mean over valid (non-pad) tokens.
  * context : late chunking -- encode the whole window once (bidirectional), then
    mean-pool each chunk's token span. Chunks are joined with the tokenizer's
    sep_token; the SEP token itself and padding are excluded from every chunk.
    We express per-chunk pooling as a single matmul with a ``pool_matrix`` so the
    same formulation drops straight into the CoreML graph (ANE-friendly); the plain
    embed is the degenerate one-chunk case (row 0 = 1/L over all valid tokens).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

Quantization = Literal["int8", "binary", "ubinary"]
N_MAX_CHUNKS = 32


# --------------------------------------------------------------------------- #
# Quantizers — bit-for-bit mirrors of st_quantize.py (operate in torch float32).
# --------------------------------------------------------------------------- #
def int8_tanh_quant(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """clamp(round(tanh(x) * 127), -128, 127) via torch.round. Returns int8 ndarray."""
    t = torch.as_tensor(x, dtype=torch.float32)
    soft = torch.tanh(t)
    q = torch.clamp(torch.round(soft * 127.0), -128, 127)
    return q.to(torch.int8).cpu().numpy()


def binary_tanh_quant(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """where(x >= 0, +1.0, -1.0). Returns float32 ndarray of +/-1."""
    t = torch.as_tensor(x, dtype=torch.float32)
    return torch.where(t >= 0, 1.0, -1.0).cpu().numpy().astype(np.float32)


def ubinary_pack(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """packbits(x >= 0) along the last axis. Returns uint8 ndarray [..., dim/8]."""
    t = torch.as_tensor(x, dtype=torch.float32)
    bits = (t.cpu().numpy() >= 0)
    return np.packbits(bits, axis=-1)


def quantize(x: torch.Tensor | np.ndarray, quantization: Quantization = "int8") -> np.ndarray:
    if quantization == "int8":
        return int8_tanh_quant(x)
    if quantization == "binary":
        return binary_tanh_quant(x)
    if quantization == "ubinary":
        return ubinary_pack(x)
    raise ValueError(f"Invalid quantization '{quantization}'; expected int8/binary/ubinary.")


# --------------------------------------------------------------------------- #
# Pooling.
# --------------------------------------------------------------------------- #
def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """[B,L,D] x [B,L] -> [B,D] mean over valid tokens (clamped denom, like modeling.py)."""
    m = mask.unsqueeze(-1).to(hidden.dtype)            # [B,L,1]
    summed = (hidden * m).sum(dim=1)                   # [B,D]
    counts = m.sum(dim=1).clamp(min=1e-9)              # [B,1]
    return summed / counts


def make_pool_matrix(spans: list[tuple[int, int]], L: int, n_max: int = N_MAX_CHUNKS) -> np.ndarray:
    """[n_max, L] float32; row k = normalized mean weights over chunk k's [start,end) span.

    Unused rows (>= len(spans)) are all-zero -> tanh(0)=0 -> zero vector; callers MUST
    skip them (NaN under cosine). The plain case is one span (0, n_valid)."""
    P = np.zeros((n_max, L), dtype=np.float32)
    for k, (start, end) in enumerate(spans[:n_max]):
        n = end - start
        if n > 0:
            P[k, start:end] = 1.0 / float(n)
    return P


def embed_context(hidden: torch.Tensor | np.ndarray, pool_matrix: np.ndarray,
                  quantization: Quantization = "int8") -> np.ndarray:
    """Late-chunking pool + quant: (pool_matrix @ hidden) -> quantize. hidden [L,D]."""
    h = torch.as_tensor(hidden, dtype=torch.float32)
    P = torch.as_tensor(pool_matrix, dtype=torch.float32)
    pooled = P @ h                                     # [n_max, D]
    return quantize(pooled, quantization)


# --------------------------------------------------------------------------- #
# Fidelity helper.
# --------------------------------------------------------------------------- #
def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Row-wise cosine for [N,D] arrays. Near-zero-norm rows -> NaN (caller excludes)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim == 1:
        a, b = a[None], b[None]
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    sim = np.full(a.shape[0], np.nan, dtype=np.float32)
    valid = (na > eps) & (nb > eps)
    if valid.any():
        sim[valid] = (a[valid] * b[valid]).sum(-1) / (na[valid] * nb[valid])
    return sim


# --------------------------------------------------------------------------- #
# The HF fp32 oracle.
# --------------------------------------------------------------------------- #
@dataclass
class _Loaded:
    model: torch.nn.Module
    tokenizer: object


class Reference:
    """Loads a pplx-embed checkpoint (fp32, CPU) and produces golden embeddings.

    >>> ref = Reference("perplexity-ai/pplx-embed-v1-0.6b")
    >>> ref.embed(["hello world"]).shape          # (1, 1024), dtype int8
    """

    def __init__(self, hf_repo: str = "perplexity-ai/pplx-embed-v1-0.6b",
                 device: str = "cpu", dtype: torch.dtype = torch.float32):
        from transformers import AutoModel, AutoTokenizer

        self.hf_repo = hf_repo
        self.device = device
        self.dtype = dtype
        model = AutoModel.from_pretrained(hf_repo, trust_remote_code=True, dtype=dtype)
        model.eval().to(device)
        tokenizer = AutoTokenizer.from_pretrained(hf_repo, trust_remote_code=True)
        self._l = _Loaded(model=model, tokenizer=tokenizer)

    @property
    def model(self) -> torch.nn.Module:
        return self._l.model

    @property
    def tokenizer(self):
        return self._l.tokenizer

    @torch.inference_mode()
    def hidden_states(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize + bidirectional forward. Returns (last_hidden_state [B,L,D], mask [B,L])."""
        enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc)
        return out.last_hidden_state.float(), enc["attention_mask"].float()

    @torch.inference_mode()
    def embed(self, texts: list[str], quantization: Quantization = "int8") -> np.ndarray:
        """Plain embed: masked-mean over valid tokens -> quantize. Returns [B, 1024]."""
        hidden, mask = self.hidden_states(texts)
        pooled = masked_mean(hidden, mask)             # [B, D]
        return quantize(pooled, quantization)

    @torch.inference_mode()
    def embed_chunks(self, documents: list[list[str]],
                     quantization: Quantization = "int8") -> list[np.ndarray]:
        """Context embed (late chunking): join chunks with sep_token, encode the whole
        window once, mean-pool each chunk's span. Returns one [n_chunks, 1024] array per doc.

        Mirrors modeling.py: SEP tokens and padding are excluded from every chunk."""
        sep = self.tokenizer.sep_token
        sep_id = self.tokenizer.sep_token_id
        joined = [sep.join(chunks) for chunks in documents]
        enc = self.tokenizer(joined, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc)
        hidden = out.last_hidden_state.float()         # [B,L,D]
        input_ids = enc["input_ids"]
        mask = enc["attention_mask"]

        results: list[np.ndarray] = []
        for b in range(input_ids.shape[0]):
            valid = mask[b].bool()
            n_valid = int(valid.sum().item())
            sep_pos = ((input_ids[b] == sep_id) & valid).nonzero(as_tuple=True)[0].tolist()
            spans: list[tuple[int, int]] = []
            start = 0
            for sp in sep_pos:
                spans.append((start, sp))              # chunk is [start, sep) — SEP excluded
                start = sp + 1
            spans.append((start, n_valid))             # final chunk to last valid token
            L = hidden.shape[1]
            P = make_pool_matrix(spans, L, n_max=len(spans))
            results.append(embed_context(hidden[b], P, quantization))
        return results


__all__ = [
    "Reference", "Quantization", "N_MAX_CHUNKS",
    "int8_tanh_quant", "binary_tanh_quant", "ubinary_pack", "quantize",
    "masked_mean", "make_pool_matrix", "embed_context", "cosine_similarity",
]
