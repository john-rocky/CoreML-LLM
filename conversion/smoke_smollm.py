#!/usr/bin/env python3
"""SmolLM 135M CoreML smoke test on Mac.

Loads the converted model and generates a few tokens. Verifies the
weight loading + tracing path produces coherent output before we wire
it as a drafter.
"""
import json
import numpy as np
import coremltools as ct
from transformers import AutoTokenizer

MODEL = "/tmp/smollm135_coreml/model.mlpackage"
TOK_DIR = "/tmp/smollm2-135m"
PROMPT = "The history of computing began with"
MAX_NEW = 32

tok = AutoTokenizer.from_pretrained(TOK_DIR)
ids = tok.encode(PROMPT, add_special_tokens=False)
print(f"prompt={PROMPT!r} ids={ids}")

mlmodel = ct.models.MLModel(MODEL, compute_units=ct.ComputeUnit.CPU_AND_GPU)
print(f"loaded {MODEL}")
print(f"inputs: {mlmodel.input_description}")
print(f"outputs: {mlmodel.output_description}")

# State init for the model
state = mlmodel.make_state()

out_ids = []
for pos, tid in enumerate(ids):
    mask = np.full((1, 1, 1, 2048), -1e4, dtype=np.float16)
    mask[..., : pos + 1] = 0.0  # attend to positions 0..pos
    upd = np.zeros((1, 1, 2048, 1), dtype=np.float16)
    upd[0, 0, pos, 0] = 1.0
    feed = {
        "input_ids": np.array([[tid]], dtype=np.int32),
        "position_ids": np.array([pos], dtype=np.int32),
        "causal_mask": mask,
        "update_mask": upd,
    }
    out = mlmodel.predict(feed, state=state)

new_id = int(out["token_id"][0])
print(f"first new token: {new_id} → {tok.decode([new_id])!r}")
out_ids.append(new_id)

for i in range(1, MAX_NEW):
    pos = len(ids) + i - 1
    mask = np.full((1, 1, 1, 2048), -1e4, dtype=np.float16)
    mask[..., : pos + 1] = 0.0
    upd = np.zeros((1, 1, 2048, 1), dtype=np.float16)
    upd[0, 0, pos, 0] = 1.0
    feed = {
        "input_ids": np.array([[new_id]], dtype=np.int32),
        "position_ids": np.array([pos], dtype=np.int32),
        "causal_mask": mask,
        "update_mask": upd,
    }
    out = mlmodel.predict(feed, state=state)
    new_id = int(out["token_id"][0])
    if new_id == 2:  # EOS
        break
    out_ids.append(new_id)

print(f"\noutput ids: {out_ids}")
print(f"generated: {tok.decode(out_ids)!r}")
print(f"full text: {PROMPT}{tok.decode(out_ids)}")
