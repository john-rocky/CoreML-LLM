#!/usr/bin/env python3
"""Re-capture Gemma 3n layer-0 activations on a much larger mixed corpus.

The original capture only got 788 tokens because the inline CORPUS string
was short. Co-firing clustering needs more samples per neuron for the
matrix to be reliable. This script targets 4000-6000 tokens spanning
multiple categories: technical narrative, Python code, JavaScript code,
math reasoning, instruction-following examples, dialog patterns.

Output:
  /tmp/l0_activations_large.npz  -- (X, Y, W_down) numpy arrays
"""
from __future__ import annotations
import argparse
import time

# wandb stub
import sys as _sys
import types as _types
import importlib.machinery as _machinery
if "wandb" not in _sys.modules:
    _w = _types.ModuleType("wandb")
    _w.__path__ = []  # type: ignore[attr-defined]
    _w.__spec__ = _machinery.ModuleSpec("wandb", loader=None, is_package=True)
    _sys.modules["wandb"] = _w

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ~6000 tokens of mixed-domain text. Generous covers since we want the
# co-firing matrix to capture real semantic clustering, not artefacts.

CORPUS_PARTS = [
"""The history of computing is filled with paradigm shifts that redefined
what machines could do. Early mechanical calculators gave way to vacuum
tube computers, then transistors, then integrated circuits. Each
revolution brought orders of magnitude in speed, miniaturisation, and
accessibility. Today the dominant trend is parallel computing,
specifically the kind of massively parallel computation that GPUs and
neural processing units enable. Large language models have become the
flagship workload of this era, with billions of parameters trained on
trillions of tokens to capture the statistics of human language.""",

"""Yet running these models on personal devices remains a challenge. A
modern smartphone has perhaps eight gigabytes of unified memory and a
power budget measured in single-digit watts. The model weights alone
can occupy a significant fraction of that memory. Bandwidth between
DRAM and the compute units is often the binding constraint rather
than compute throughput itself. Optimisations that reduce the amount
of data moved per token — quantisation, sparsity, speculative
decoding, prefix caching — have therefore taken centre stage in the
last few years. Mobile inference frameworks compete on how
aggressively they can pack compute into the available bandwidth.""",

"""Apple's Neural Engine occupies an interesting niche in this landscape.
It was originally designed for small computer-vision models that fit
comfortably in its on-chip SRAM. The introduction of the Foundation
Models framework in iOS 26 pushed it into territory it was not built
for, and developers have been discovering the boundaries ever since.
Recent reverse engineering work has illuminated much of the previously
opaque architecture, revealing both constraints and opportunities.
Native sparsity, dynamic routing, and block-quantised weights are all
on the table.""",

"""class BinarySearchTree:
    def __init__(self):
        self.root = None
        self.size = 0

    def insert(self, value):
        if self.root is None:
            self.root = TreeNode(value)
            self.size += 1
            return
        node = self.root
        while True:
            if value < node.value:
                if node.left is None:
                    node.left = TreeNode(value)
                    self.size += 1
                    return
                node = node.left
            elif value > node.value:
                if node.right is None:
                    node.right = TreeNode(value)
                    self.size += 1
                    return
                node = node.right
            else:
                return  # duplicate

    def contains(self, value):
        node = self.root
        while node is not None:
            if value == node.value:
                return True
            node = node.left if value < node.value else node.right
        return False

    def inorder(self):
        result = []
        def walk(node):
            if node is None:
                return
            walk(node.left)
            result.append(node.value)
            walk(node.right)
        walk(self.root)
        return result""",

"""function quicksort(arr) {
    if (arr.length <= 1) return arr;
    const pivot = arr[Math.floor(arr.length / 2)];
    const left = arr.filter(x => x < pivot);
    const middle = arr.filter(x => x === pivot);
    const right = arr.filter(x => x > pivot);
    return [...quicksort(left), ...middle, ...quicksort(right)];
}

function debounce(fn, delay) {
    let timer = null;
    return function (...args) {
        if (timer) clearTimeout(timer);
        timer = setTimeout(() => fn.apply(this, args), delay);
    };
}

async function fetchAll(urls) {
    const results = await Promise.all(urls.map(url => fetch(url)));
    return Promise.all(results.map(r => r.json()));
}

const memoize = (fn) => {
    const cache = new Map();
    return (...args) => {
        const key = JSON.stringify(args);
        if (cache.has(key)) return cache.get(key);
        const result = fn(...args);
        cache.set(key, result);
        return result;
    };
};""",

"""Find the derivative of f(x) = x^3 - 2x^2 + 5x - 7 using basic rules.
Step 1: apply the power rule term by term.
The derivative of x^3 is 3*x^(3-1) = 3*x^2.
The derivative of -2*x^2 is -2 * 2*x = -4*x.
The derivative of 5*x is just 5 (constant times the variable).
The derivative of the constant -7 is 0.
Step 2: combine the terms.
f'(x) = 3*x^2 - 4*x + 5.

Now consider integration. The antiderivative of g(x) = 2*x + 1 is
G(x) = x^2 + x + C, where C is the constant of integration. To
evaluate the definite integral of g from 0 to 3, compute
G(3) - G(0) = (9 + 3) - 0 = 12. The fundamental theorem of calculus
guarantees that this matches the area under the curve of g on [0, 3].""",

"""User: Could you summarise the main idea of speculative decoding?
Assistant: Speculative decoding speeds up large language model inference
by having a small "drafter" model propose several tokens at once, then
the large "target" model verifies them in parallel. If the drafter's
guesses match what the target would have produced, all the proposed
tokens are accepted in one pass instead of one at a time. The speedup
depends on how often the drafter agrees with the target.

User: What's the catch?
Assistant: The drafter has to actually be a good predictor of the
target, which is non-trivial. For mismatched vocab pairs or domain
gaps, the drafter rarely agrees and the verification overhead becomes
larger than the savings. Quality of speculative decoding is bounded by
the drafter's per-token accept rate.""",

"""Say yes ten times: yes yes yes yes yes yes yes yes yes yes.
Say no five times: no no no no no.
Count from one to twenty: one two three four five six seven eight nine
ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen
nineteen twenty.
Recite the alphabet: a b c d e f g h i j k l m n o p q r s t u v w x
y z.
The quick brown fox jumps over the lazy dog. The quick brown fox jumps
over the lazy dog. The quick brown fox jumps over the lazy dog.
Repeat after me: hello world. Hello world. Hello world.""",

"""Linear algebra notation review. A matrix is a rectangular array of
numbers arranged in rows and columns. We write a m-by-n matrix A as a
collection of entries a_{ij} where i ranges over rows and j over
columns. Matrix multiplication is defined when the inner dimensions
match: an m-by-n matrix times an n-by-p matrix yields an m-by-p
matrix whose entry at row i column k is the sum over j of a_{ij}
times b_{jk}. The transpose of a matrix swaps rows and columns. The
identity matrix has ones on the diagonal and zeros elsewhere. A
matrix is invertible if it admits a matrix B such that AB = BA = I,
in which case we write B as A inverse.""",

"""The transformer architecture revolutionised natural language
processing by replacing recurrence with self-attention. Each layer
applies a multi-head attention mechanism that lets every token attend
to every other token, computing weighted averages of value vectors
based on similarity between query and key vectors. After attention,
a position-wise feed forward network applies two linear projections
with a non-linear activation in between. Residual connections and
layer normalisation stabilise training and let the network grow very
deep. Most modern large language models follow this template with
small variations such as rotary positional encodings or grouped
query attention.""",

"""def fibonacci(n):
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a""",

"""When the user asks a question, first read it carefully. Then think
about what information you need. If the question is ambiguous, ask
for clarification rather than guessing. Avoid making up facts. If you
do not know something, say so. Keep responses concise and focused on
the question. Use examples when they help illustrate a concept. Cite
sources when relevant. Acknowledge uncertainty when present. Adapt
the tone to match the context: formal for technical questions, casual
for conversational ones. Always proofread before sending.""",

"""Quantum computing exploits superposition and entanglement to process
information in ways classical computers cannot. A qubit can be in a
state that is a complex linear combination of zero and one, and
multiple qubits can become entangled so that their states are
correlated regardless of distance. Algorithms like Shor's factoring
algorithm and Grover's search algorithm achieve speedups over their
classical counterparts by leveraging these phenomena. However,
maintaining quantum coherence is extremely difficult, requiring near
absolute zero temperatures and aggressive error correction.""",
]

CORPUS = "\n\n".join(CORPUS_PARTS)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/tmp/gemma3n-e2b")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--tokens", type=int, default=6000)
    p.add_argument("--out", default="/tmp/l0_activations_large.npz")
    args = p.parse_args()

    device = get_device()
    print(f"[capture] device={device}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device).eval()

    # Locate layer
    root = getattr(model, "model", None) or model
    text = getattr(root, "language_model", None) or root
    layer = text.layers[args.layer]
    W = layer.mlp.down_proj.weight.detach().to(torch.float32).cpu().numpy()
    print(f"[capture] W_down {W.shape}")

    X_buf, Y_buf = [], []

    def pre_mlp(_m, inputs):
        x = inputs[0].detach()
        flat = x.reshape(-1, x.shape[-1]).to(torch.float32).cpu().numpy()
        X_buf.append(flat)

    def pre_down(_m, inputs):
        y = inputs[0].detach()
        flat = y.reshape(-1, y.shape[-1]).to(torch.float32).cpu().numpy()
        Y_buf.append(flat)

    h1 = layer.mlp.register_forward_pre_hook(pre_mlp)
    h2 = layer.mlp.down_proj.register_forward_pre_hook(pre_down)

    enc = tok(CORPUS, return_tensors="pt", truncation=True,
              max_length=args.tokens)
    input_ids = enc["input_ids"].to(device)
    print(f"[capture] corpus tokens: {input_ids.shape[1]}")

    t0 = time.time()
    with torch.no_grad():
        _ = model(input_ids=input_ids)
    print(f"[capture] forward {time.time()-t0:.1f}s")

    h1.remove()
    h2.remove()

    X = np.concatenate(X_buf, axis=0)
    Y = np.concatenate(Y_buf, axis=0)
    print(f"[capture] X{X.shape}  Y{Y.shape}")
    # Drop non-finite
    finite = np.isfinite(Y).all(axis=-1) & np.isfinite(X).all(axis=-1)
    print(f"[capture] dropping {int((~finite).sum())} non-finite tokens")
    X = X[finite]
    Y = Y[finite]
    print(f"[capture] final X{X.shape}  Y{Y.shape}")

    np.savez(args.out, X=X, Y=Y, W_down=W)
    print(f"[capture] wrote {args.out}")


if __name__ == "__main__":
    main()
