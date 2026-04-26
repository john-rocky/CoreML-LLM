#!/usr/bin/env python3
"""Build Gemma 4 E2B stateful chunks (MLState + slice_update KV).

Phase 1 of the Gemma 4 stateful migration — mirrors the Qwen3-VL
v1.5.0 pattern. Produces 4 mlpackages:

  chunk_1.mlpackage: own KV state (sliding + full), computes per_layer_combined
  chunk_2.mlpackage: own KV state, emits kv13_*/kv14_* producer aliases
  chunk_3.mlpackage: stateless, reads kv13/14
  chunk_4.mlpackage: stateless, reads kv13/14, lm_head + argmax

Sliding cache uses ring writes at slot `ring_pos = current_pos % W`,
which Swift precomputes and passes alongside `current_pos`. The
`update_mask` input that the recurrent build needed for ANE-compat
out-of-place full-layer writes is GONE — `ios18.slice_update` does
the in-place write natively.

Sidecars (embed_weight, per-layer projection, RoPE tables, tokenizer,
model_config.json) are produced by the existing
`build_gemma4_bundle.py` pipeline — this script only touches the
chunk mlpackages. After running this, copy the chunk_{1..4}.mlpackage
files alongside an existing E2B sidecar bundle to ship a complete
Gemma 4 E2B stateful build.

T=1 only (decode + slow T=1 prefill). Multifunction prefill_bN is a
follow-up (mirrors Qwen3-VL v1.5.0 → multifunction multifunction).

Usage:
  python conversion/build_gemma4_e2b_stateful_chunks.py \
      --output /tmp/gemma4-e2b-stateful \
      [--ctx 2048] [--nbits 4] [--only-chunk N]

  GEMMA4_HF_DIR or --hf-dir picks the local HF model path.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import coremltools as ct
from coremltools.optimize.coreml import (
    OpPalettizerConfig, OptimizationConfig, palettize_weights,
    linear_quantize_activations, linear_quantize_weights,
    OpLinearQuantizerConfig,
)
from coremltools.optimize.coreml.experimental import (
    OpActivationLinearQuantizerConfig,
)

from ane_ops import MODEL_DTYPE
from config import MODEL_REGISTRY
from models.gemma4 import Gemma4Model
from models.gemma4_swa_chunks import compute_chunk_boundaries
from models.gemma4_swa_stateful_chunks import (
    SWAStatefulChunk1, SWAStatefulChunk2,
    SWAStatefulChunk3, SWAStatefulChunk4,
)


DEFAULT_HF_DIR = os.environ.get(
    "GEMMA4_HF_DIR", f"{ROOT}/../output/gemma4-e2b/hf_model")
fp16 = np.float16


# ============================================================
# HF model loading (mirrors build_verify_chunks.py)
# ============================================================

def _resolve_hf_dir(model_name: str, override: str | None) -> str:
    if override:
        return override
    if model_name in MODEL_REGISTRY:
        from huggingface_hub import snapshot_download
        repo = MODEL_REGISTRY[model_name].hf_repo
        local = os.path.join(ROOT, "..", "output", model_name, "hf_model")
        if not os.path.isdir(local) or not any(
            fn.endswith(".safetensors") for fn in os.listdir(local)
        ):
            print(f"Downloading {repo} to {local}...")
            snapshot_download(
                repo, local_dir=local,
                allow_patterns=["*.safetensors", "*.json", "tokenizer*",
                                "*.txt", "*.model"],
            )
        return local
    return DEFAULT_HF_DIR


# ============================================================
# Conversion helpers
# ============================================================

def _audit_ane(pkg_path: str) -> float:
    try:
        m = ct.models.MLModel(pkg_path,
                              compute_units=ct.ComputeUnit.CPU_AND_NE)
        compiled = m.get_compiled_model_path()
        plan = ct.models.compute_plan.MLComputePlan.load_from_path(
            path=str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        dev = Counter()
        for fn in plan.model_structure.program.functions.values():
            for op in fn.block.operations:
                a = plan.get_compute_device_usage_for_mlprogram_operation(op)
                d = ("const" if (a is None and op.operator_name == "const")
                     else (a.preferred_compute_device.__class__.__name__
                           if a else "unknown"))
                dev[d] += 1
        total = sum(dev.values())
        compute = total - dev.get("const", 0)
        ane = dev.get("MLNeuralEngineComputeDevice", 0)
        pct = 100 * ane / compute if compute else 0.0
        print(f"    ANE placement: {ane}/{compute} ({pct:.1f}%)")
        return pct
    except Exception as e:
        print(f"    ANE audit skipped: {e}")
        return 0.0


def _patch_suffix_skip_missing():
    """`insert_suffix_quantize_dequantize_pair` reads `rmin`/`rmax` for the
    OUTPUT var of every dequantize→conv/add/linear/pool pattern via
    `optimize_utils.get_min_and_max_values`. If our prefix patch caused
    the calibration to skip an op (e.g. bool-typed intermediate), the
    suffix pass also crashes with KeyError.

    Wrap the suffix pass's _try_apply_transform to silently skip when
    the relevant var has no stats. Idempotent.
    """
    from coremltools.converters.mil.mil.passes.defs import (
        optimize_activation_quantization as suff_mod,
    )
    cls = suff_mod.insert_suffix_quantize_dequantize_pair
    if getattr(cls._try_apply_transform, "_missing_stats_patched", False):
        return
    original = cls._try_apply_transform

    def patched(self, last_op, _child_op, block, visited_ops, op_config):
        var_name = last_op.outputs[0].name
        stats = self._activation_stats
        if stats is None or var_name not in stats:
            return False
        if "rmin" not in stats[var_name] or "rmax" not in stats[var_name]:
            return False
        # Skip degenerate ranges where the suffix path produces scale=0,
        # tripping iOS17 quantize op's "scale cannot be 0" validation.
        # (cml9 prefix has a guard; suffix doesn't.)
        if float(stats[var_name]["rmin"]) == float(stats[var_name]["rmax"]):
            return False
        return original(self, last_op, _child_op, block, visited_ops, op_config)

    patched._missing_stats_patched = True
    cls._try_apply_transform = patched


def _patch_quant_dequant_skip_missing():
    """cml9's `insert_prefix_quantize_dequantize_pair` pass walks every
    `linear`/`conv`/`matmul`/`add`/pool op and tries to quantize its
    inputs. If an input's source op has a non-output-compatible type
    (e.g. bool intermediates from mask construction), the calibration
    pass silently SKIPS that var (cloned_output_type is None branch in
    `predict_intermediate_outputs`), so its rmin/rmax are absent. The
    insertion pass then crashes with KeyError('rmin').

    Patch `transform_op` to skip ops whose `x` input has no stats,
    rather than crash. Effect: those specific edges stay fp16 while
    everything else gets INT8 quantized — partial coverage is fine
    for memory-bandwidth gain.
    """
    from coremltools.optimize.coreml import _quantization_passes as qpasses
    cls = qpasses.insert_prefix_quantize_dequantize_pair
    if getattr(cls.transform_op, "_missing_stats_patched", False):
        return
    original_transform = cls.transform_op

    def patched_transform_op(self, op):
        from coremltools.converters.mil.mil import types as mil_types
        x = op.inputs.get("x") if hasattr(op.inputs, "get") else None
        if x is None:
            x = op.inputs["x"] if "x" in op.inputs else None
        if x is None:
            return original_transform(self, op)
        # Skip non-floating-point inputs — quantize op requires scale dtype
        # to match input dtype, but cml9 always uses fp16/fp32 scales.
        # An int32 `add` (e.g. ring_pos + 1) trips a downstream
        # ValueError during quantize op construction.
        if x.dtype not in (mil_types.fp16, mil_types.fp32):
            return
        if self._activation_stats is not None:
            if x.name not in self._activation_stats:
                return
            stats = self._activation_stats[x.name]
            if "rmin" not in stats or "rmax" not in stats:
                return
        return original_transform(self, op)

    patched_transform_op._missing_stats_patched = True
    cls.transform_op = patched_transform_op


def _patch_calibrator_for_stateful():
    """cml9's `linear_quantize_activations` calibrator clones the spec,
    appends intermediate outputs, builds a temp MLModel, and calls
    `model.predict(inputs)` per calibration sample. For STATEFUL models
    this raises "The input feature for kv_cache_full must be an MLState,
    but it was not." because no state is allocated.

    Patch `ModelDebugger.predict_intermediate_outputs` to detect a
    stateful temp model and allocate fresh state per call. Idempotent.
    """
    import coremltools.optimize.coreml.experimental._model_debugger as dbg
    if getattr(dbg.ModelDebugger.predict_intermediate_outputs,
               "_stateful_patched", False):
        return
    from coremltools.optimize.coreml.experimental._model_debugger import (
        ModelDebugger, ModelInfo, _SPECIFICATION_VERSION_IOS_16, logger,
    )

    def predict_intermediate_outputs(
        self, inputs, intermediate_output_names,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    ):
        cloned_spec = ModelDebugger.clone_spec(self.model_info.spec)
        if self.model_info.spec.specificationVersion < _SPECIFICATION_VERSION_IOS_16:
            logger.warning(
                f"The model has spec version {self.model_info.spec.specificationVersion}, "
                f"forcefully updated to {_SPECIFICATION_VERSION_IOS_16} during calibration."
            )
            cloned_spec.specificationVersion = max(
                self.model_info.spec.specificationVersion, 7)
        cloned_model_info = ModelInfo(
            ModelDebugger.get_program_info(cloned_spec.mlProgram), cloned_spec
        )
        cloned_block_info = ModelDebugger.get_any_block(cloned_model_info)

        for output_name in intermediate_output_names:
            cloned_output_type = ModelDebugger.get_output_feature_type(
                output_name, self.block_info.operations
            )
            if cloned_output_type is None:
                continue
            cloned_block_info.spec.outputs.append(output_name)
            cloned_output = ct.proto.Model_pb2.FeatureDescription()
            cloned_output.name = output_name
            cloned_output.type.multiArrayType.dataType = cloned_output_type
            cloned_model_info.spec.description.output.append(cloned_output)

        model = ct.models.MLModel(
            cloned_spec,
            weights_dir=self.weights_dir,
            compute_units=compute_units,
            skip_model_load=False,
        )
        # PATCH: allocate fresh state for stateful models so predict()
        # doesn't trip "input feature for <state> must be an MLState".
        if model._is_stateful():
            return model.predict(inputs, state=model.make_state())
        return model.predict(inputs)

    predict_intermediate_outputs._stateful_patched = True
    dbg.ModelDebugger.predict_intermediate_outputs = predict_intermediate_outputs


def _make_calibration_samples(input_specs, num_samples=4, seed=0):
    """Synthetic calibration data for `linear_quantize_activations`.

    Names + shapes are taken from the converted model's input specs;
    values are sampled from distributions that approximate the real
    activations entering each chunk:
      - mask inputs: zero (all-attend) — masks are added, so 0 is the
        "no penalty" value; -inf positions don't affect activation range
      - cos/sin: uniform(-1, 1) — RoPE outputs are bounded
      - position counters (int32): small monotone values per sample
      - everything else (hidden_states, per_layer, kv13/kv14): N(0, 0.5)
        roughly matches embedding scales after RMSNorm

    Stateful chunks: only regular tensor inputs go in the dict. The
    `predict` call inside the calibrator allocates fresh state via the
    spec's StateType; we don't pass MLState ourselves.
    """
    from coremltools.converters.mil.mil import types as mil_types
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(num_samples):
        d = {}
        for spec in input_specs:
            # ct.TensorType.shape is a `Shape` object — use .to_list() to
            # get a python list of dim ints. ct.TensorType.dtype is a MIL
            # type, not a numpy dtype, so compare against mil_types.
            shape = tuple(int(s) for s in spec.shape.to_list())
            name = spec.name
            if spec.dtype == mil_types.int32:
                d[name] = np.full(shape, i, dtype=np.int32)
            elif "mask" in name:
                d[name] = np.zeros(shape, dtype=np.float16)
            elif name.startswith(("cos_", "sin_")):
                d[name] = rng.uniform(-1.0, 1.0, shape).astype(np.float16)
            else:
                d[name] = rng.normal(0.0, 0.5, shape).astype(np.float16)
        samples.append(d)
    return samples


def _load_real_calibration_samples(npz_path: str, input_names: set) -> list:
    """Load .npz produced by `gen_calib_data_real.py` and reshape into
    the dict-of-arrays format `linear_quantize_activations` expects.
    Filters to only the input names that exist in the converted model
    (so chunk_2/3/4 paths can reuse the chunk_1 npz by ignoring extras).
    """
    data = np.load(npz_path)
    n = int(data["_meta_num_samples"][0])
    samples = []
    for i in range(n):
        d = {}
        for name in input_names:
            key = f"sample_{i:03d}__{name}"
            if key in data.files:
                d[name] = data[key]
        if len(d) == len(input_names):
            samples.append(d)
    return samples


def _trace_and_convert_stateful(
    model, sample_inputs, input_specs, output_specs, state_specs,
    out_path: str, quantize_nbits: int,
    activation_quant: bool = False,
    calib_samples: int = 4,
    calib_data_path: str | None = None,
    activation_scope: str = "linear",
    activation_mode: str = "linear_symmetric",
    joint_int8_lut: bool = False,
):
    """Trace + ct.convert with StateType, save, optionally palettize, audit."""
    t = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(model, sample_inputs, check_trace=False,
                                 strict=False)
    print(f"    traced in {time.time()-t:.1f}s")

    t = time.time()
    convert_kwargs = dict(
        convert_to="mlprogram",
        inputs=input_specs,
        outputs=output_specs,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    # Only pass states= for stateful chunks. Empty list trips ct.convert
    # on some coremltools versions ("expected at least one state").
    if state_specs:
        convert_kwargs["states"] = state_specs
    mlmodel = ct.convert(traced, **convert_kwargs)
    print(f"    converted in {time.time()-t:.1f}s")

    if quantize_nbits > 0:
        t = time.time()
        op_cfg = OpPalettizerConfig(
            mode="kmeans", nbits=quantize_nbits,
            granularity="per_grouped_channel", group_size=32,
        )
        mlmodel = palettize_weights(mlmodel, OptimizationConfig(global_config=op_cfg))
        print(f"    palettized int{quantize_nbits} in {time.time()-t:.1f}s")
        if joint_int8_lut:
            t = time.time()
            jq_cfg = OpLinearQuantizerConfig(
                mode="linear_symmetric", dtype="int8",
                granularity="per_tensor", weight_threshold=2048,
            )
            mlmodel = linear_quantize_weights(
                mlmodel, OptimizationConfig(global_config=jq_cfg),
                joint_compression=True,
            )
            print(f"    joint INT8-LUT quantized in {time.time()-t:.1f}s")

    if activation_quant:
        t = time.time()
        _patch_calibrator_for_stateful()
        _patch_quant_dequant_skip_missing()
        _patch_suffix_skip_missing()
        act_cfg = OpActivationLinearQuantizerConfig(
            mode=activation_mode,
            weight_threshold=2048,
        )
        # Two scope modes: "linear" (cml9 PR #2577's main contribution —
        # narrow scope on matmul path only) and "all" (also quantize
        # conv/add/pool inputs). The 2026-04-26 first attempt with synth
        # calibration found "linear" insufficient (cos sim 0.15), but the
        # quality root cause was the calibration data, not the scope.
        # Both modes are valid; default "linear" is the safer narrow scope.
        if activation_scope == "all":
            opt_cfg = OptimizationConfig(global_config=act_cfg)
        else:
            opt_cfg = OptimizationConfig(op_type_configs={"linear": act_cfg})

        if calib_data_path:
            input_names = {spec.name for spec in input_specs}
            sample_data = _load_real_calibration_samples(
                calib_data_path, input_names)
            if not sample_data:
                raise RuntimeError(
                    f"No samples loaded from {calib_data_path} matching "
                    f"this chunk's input names {sorted(input_names)}")
            print(f"    activation-quantizing INT8 (scope={activation_scope}) "
                  f"with {len(sample_data)} REAL calibration samples "
                  f"from {calib_data_path}")
        else:
            sample_data = _make_calibration_samples(
                input_specs, num_samples=calib_samples)
            print(f"    activation-quantizing INT8 (scope={activation_scope}) "
                  f"with {len(sample_data)} synthetic calibration samples...")
        mlmodel = linear_quantize_activations(mlmodel, opt_cfg, sample_data)
        print(f"    activation-quantized in {time.time()-t:.1f}s")

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    mlmodel.save(out_path)
    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(out_path)
        for f in fns
    ) / 1024 / 1024
    print(f"    saved {out_path} ({size_mb:.1f} MB)")
    _audit_ane(out_path)


# ============================================================
# Per-chunk converters
# ============================================================

def convert_chunk1(base, c_start, c_end, ctx, out_path, nbits, *,
                   use_linear=False, activation_quant=False,
                   calib_data_path=None, activation_scope="linear",
                   activation_mode="linear_symmetric",
                   awq_calib_data_path=None, awq_alpha=0.5,
                   joint_int8_lut=False):
    print("\n" + "=" * 60)
    print(f"CHUNK 1 (L{c_start}-{c_end-1}) — own KV state, computes PLE")
    print("=" * 60)
    cfg = base.config
    hidden = cfg.hidden_size
    pld = cfg.hidden_size_per_layer_input
    nlayers = cfg.num_hidden_layers
    W = cfg.sliding_window
    hd_s, hd_f = cfg.head_dim, cfg.global_head_dim
    max_hd = hd_f
    HKV = cfg.num_key_value_heads

    chunk = SWAStatefulChunk1(base, c_start, c_end, ctx,
                               use_linear=use_linear).eval().to(MODEL_DTYPE)
    ns, nf = max(chunk.num_sliding, 1), max(chunk.num_full, 1)

    if awq_calib_data_path:
        from awq_smoothing import apply_awq_to_chunk
        print(f"  Applying AWQ smoothing (alpha={awq_alpha}) before trace...")
        apply_awq_to_chunk(chunk, awq_calib_data_path, alpha=awq_alpha)

    sample = (
        torch.zeros(1, 1, hidden, dtype=torch.float16),                  # hidden_states
        torch.zeros(1, 1, 1, ctx, dtype=torch.float16),                   # causal_mask_full
        torch.zeros(1, 1, 1, W, dtype=torch.float16),                     # causal_mask_sliding
        torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),            # per_layer_raw
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),                  # cos_s
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),                  # sin_s
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),                  # cos_f
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),                  # sin_f
        torch.zeros(1, dtype=torch.int32),                                # current_pos
        torch.zeros(1, dtype=torch.int32),                                # ring_pos
    )
    inputs = [
        ct.TensorType(name="hidden_states",      shape=sample[0].shape, dtype=fp16),
        ct.TensorType(name="causal_mask_full",   shape=sample[1].shape, dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=sample[2].shape, dtype=fp16),
        ct.TensorType(name="per_layer_raw",      shape=sample[3].shape, dtype=fp16),
        ct.TensorType(name="cos_s",              shape=sample[4].shape, dtype=fp16),
        ct.TensorType(name="sin_s",              shape=sample[5].shape, dtype=fp16),
        ct.TensorType(name="cos_f",              shape=sample[6].shape, dtype=fp16),
        ct.TensorType(name="sin_f",              shape=sample[7].shape, dtype=fp16),
        ct.TensorType(name="current_pos",        shape=(1,),            dtype=np.int32),
        ct.TensorType(name="ring_pos",           shape=(1,),            dtype=np.int32),
    ]
    outputs = [
        ct.TensorType(name="hidden_states_out",       dtype=fp16),
        ct.TensorType(name="per_layer_combined_out",  dtype=fp16),
    ]
    states = [
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(2 * ns, HKV, W, max_hd), dtype=fp16),
            name="kv_cache_sliding",
        ),
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(2 * nf, HKV, ctx, max_hd), dtype=fp16),
            name="kv_cache_full",
        ),
    ]
    _trace_and_convert_stateful(
        chunk, sample, inputs, outputs, states, out_path, nbits,
        activation_quant=activation_quant,
        calib_data_path=calib_data_path,
        activation_scope=activation_scope,
        activation_mode=activation_mode,
        joint_int8_lut=joint_int8_lut)


def convert_chunk2(base, c_start, c_end, ctx, out_path, nbits, *,
                   use_linear=False, activation_quant=False,
                   calib_data_path=None, activation_scope="linear",
                   activation_mode="linear_symmetric"):
    print("\n" + "=" * 60)
    print(f"CHUNK 2 (L{c_start}-{c_end-1}) — own KV state, emits kv13/kv14")
    print("=" * 60)
    cfg = base.config
    hidden = cfg.hidden_size
    pld = cfg.hidden_size_per_layer_input
    nlayers = cfg.num_hidden_layers
    W = cfg.sliding_window
    hd_s, hd_f = cfg.head_dim, cfg.global_head_dim
    max_hd = hd_f
    HKV = cfg.num_key_value_heads

    chunk = SWAStatefulChunk2(base, c_start, c_end, ctx,
                               use_linear=use_linear).eval().to(MODEL_DTYPE)
    ns, nf = max(chunk.num_sliding, 1), max(chunk.num_full, 1)

    sample = (
        torch.zeros(1, 1, hidden, dtype=torch.float16),
        torch.zeros(1, 1, 1, ctx, dtype=torch.float16),
        torch.zeros(1, 1, 1, W, dtype=torch.float16),
        torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
    )
    inputs = [
        ct.TensorType(name="hidden_states",      shape=sample[0].shape, dtype=fp16),
        ct.TensorType(name="causal_mask_full",   shape=sample[1].shape, dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=sample[2].shape, dtype=fp16),
        ct.TensorType(name="per_layer_combined", shape=sample[3].shape, dtype=fp16),
        ct.TensorType(name="cos_s",              shape=sample[4].shape, dtype=fp16),
        ct.TensorType(name="sin_s",              shape=sample[5].shape, dtype=fp16),
        ct.TensorType(name="cos_f",              shape=sample[6].shape, dtype=fp16),
        ct.TensorType(name="sin_f",              shape=sample[7].shape, dtype=fp16),
        ct.TensorType(name="current_pos",        shape=(1,),            dtype=np.int32),
        ct.TensorType(name="ring_pos",           shape=(1,),            dtype=np.int32),
    ]
    outputs = [
        ct.TensorType(name="hidden_states_out", dtype=fp16),
        ct.TensorType(name="kv13_k",            dtype=fp16),
        ct.TensorType(name="kv13_v",            dtype=fp16),
        ct.TensorType(name="kv14_k",            dtype=fp16),
        ct.TensorType(name="kv14_v",            dtype=fp16),
    ]
    states = [
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(2 * ns, HKV, W, max_hd), dtype=fp16),
            name="kv_cache_sliding",
        ),
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(2 * nf, HKV, ctx, max_hd), dtype=fp16),
            name="kv_cache_full",
        ),
    ]
    _trace_and_convert_stateful(
        chunk, sample, inputs, outputs, states, out_path, nbits,
        activation_quant=activation_quant,
        calib_data_path=calib_data_path,
        activation_scope=activation_scope,
        activation_mode=activation_mode)


def convert_chunk_shared(chunk_cls, base, c_start, c_end, ctx,
                         out_path, nbits, name, with_lm_head, *,
                         use_linear=False, activation_quant=False,
                         calib_data_path=None, activation_scope="linear",
                         activation_mode="linear_symmetric"):
    """Stateless chunk (3 or 4). All layers KV-shared from kv13/kv14."""
    print("\n" + "=" * 60)
    print(f"{name} (L{c_start}-{c_end-1}) — stateless, reads kv13/14"
          + (" + lm_head" if with_lm_head else ""))
    print("=" * 60)
    cfg = base.config
    hidden = cfg.hidden_size
    pld = cfg.hidden_size_per_layer_input
    nlayers = cfg.num_hidden_layers
    W = cfg.sliding_window
    hd_s, hd_f = cfg.head_dim, cfg.global_head_dim
    HKV = cfg.num_key_value_heads

    chunk = chunk_cls(base, c_start, c_end,
                      use_linear=use_linear).eval().to(MODEL_DTYPE)
    sample = (
        torch.zeros(1, 1, hidden, dtype=torch.float16),
        torch.zeros(1, 1, 1, ctx, dtype=torch.float16),
        torch.zeros(1, 1, 1, W, dtype=torch.float16),
        torch.zeros(1, 1, nlayers * pld, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_s, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),
        torch.zeros(1, 1, 1, hd_f, dtype=torch.float16),
        torch.zeros(1, HKV, W, hd_s, dtype=torch.float16),     # kv13_k
        torch.zeros(1, HKV, W, hd_s, dtype=torch.float16),     # kv13_v
        torch.zeros(1, HKV, ctx, hd_f, dtype=torch.float16),   # kv14_k
        torch.zeros(1, HKV, ctx, hd_f, dtype=torch.float16),   # kv14_v
    )
    inputs = [
        ct.TensorType(name="hidden_states",      shape=sample[0].shape, dtype=fp16),
        ct.TensorType(name="causal_mask_full",   shape=sample[1].shape, dtype=fp16),
        ct.TensorType(name="causal_mask_sliding", shape=sample[2].shape, dtype=fp16),
        ct.TensorType(name="per_layer_combined", shape=sample[3].shape, dtype=fp16),
        ct.TensorType(name="cos_s",              shape=sample[4].shape, dtype=fp16),
        ct.TensorType(name="sin_s",              shape=sample[5].shape, dtype=fp16),
        ct.TensorType(name="cos_f",              shape=sample[6].shape, dtype=fp16),
        ct.TensorType(name="sin_f",              shape=sample[7].shape, dtype=fp16),
        ct.TensorType(name="kv13_k",             shape=sample[8].shape, dtype=fp16),
        ct.TensorType(name="kv13_v",             shape=sample[9].shape, dtype=fp16),
        ct.TensorType(name="kv14_k",             shape=sample[10].shape, dtype=fp16),
        ct.TensorType(name="kv14_v",             shape=sample[11].shape, dtype=fp16),
    ]
    if with_lm_head:
        outputs = [
            ct.TensorType(name="token_id",    dtype=np.int32),
            ct.TensorType(name="token_logit", dtype=fp16),
            ct.TensorType(name="hidden_normed", dtype=fp16),
        ]
    else:
        outputs = [ct.TensorType(name="hidden_states_out", dtype=fp16)]
    _trace_and_convert_stateful(
        chunk, sample, inputs, outputs, state_specs=[],
        out_path=out_path, quantize_nbits=nbits,
        activation_quant=activation_quant,
        calib_data_path=calib_data_path,
        activation_scope=activation_scope,
        activation_mode=activation_mode)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4-e2b",
                    help="Model name (gemma4-e2b | gemma4-e4b)")
    ap.add_argument("--output", required=True,
                    help="Output directory for stateful mlpackages")
    ap.add_argument("--hf-dir", default=None,
                    help="Override HF model path (skip auto-download)")
    ap.add_argument("--ctx", type=int, default=None,
                    help="Context length (default: registry default)")
    ap.add_argument("--nbits", type=int, default=4, choices=[0, 4, 8],
                    help="Palettization (0 = fp16, 4 = INT4, 8 = INT8)")
    ap.add_argument("--only-chunk", type=int, default=None, choices=[1, 2, 3, 4],
                    help="Smoke test: convert only one chunk and stop. "
                         "Useful for first runs on Mac Studio to validate "
                         "the conversion path before committing to all 4.")
    ap.add_argument("--linear-projections", action="store_true",
                    help="Plan-3 (cml9 PR #2577) variant: replace every "
                         "Conv2d(1×1) projection with shape-equivalent "
                         "nn.Linear (weights reshaped from (out,in,1,1) "
                         "to (out,in)). Drops the permute/squeeze wrapper "
                         "ops. ANE placement was 100% in 5-layer PoC + W4, "
                         "but Mac W4 latency was +21% — iPhone re-test "
                         "gates production migration.")
    ap.add_argument("--activation-quant", action="store_true",
                    help="Stage 1 (cml9 PR #2577 final form): apply "
                         "linear_quantize_activations (INT8 symmetric) on "
                         "top of W4 palettize. Halves intra-op memory "
                         "bandwidth on Apple ANE. Currently cml9 only "
                         "supports n=8 activations. Calibration uses "
                         "synthetic samples by default.")
    ap.add_argument("--calib-samples", type=int, default=4,
                    help="Number of synthetic calibration samples for "
                         "--activation-quant (default: 4). More samples "
                         "= broader activation range coverage but slower "
                         "calibration. Ignored when --calib-data is set.")
    ap.add_argument("--calib-data", default=None,
                    help="Path to .npz produced by gen_calib_data_real.py. "
                         "When set, uses real-prompt activation samples "
                         "for calibration (overrides synthetic). Stage 1 "
                         "retry path; the synthetic calibration produced "
                         "cos sim 0.108 vs W4 baseline (HOLD doc).")
    ap.add_argument("--activation-scope", default="linear",
                    choices=["linear", "all"],
                    help="Op-type scope for INT8 activation quant. "
                         "'linear' (default) only quantizes matmul-path "
                         "linear ops (cml9 PR #2577 contribution). 'all' "
                         "applies the global config including conv/add/"
                         "pool. 'all' compounds error across residuals.")
    ap.add_argument("--activation-mode", default="linear_symmetric",
                    choices=["linear_symmetric", "linear"],
                    help="Quantization mode. 'linear_symmetric' (default) "
                         "uses zero_point=0, scale via max(|rmin|, |rmax|). "
                         "'linear' is asymmetric (zero_point !=0). Asymmetric "
                         "is generally tighter for non-zero-centered activs.")
    ap.add_argument("--awq", action="store_true",
                    help="Stage 1 v3: apply AWQ-style per-channel "
                         "smoothing (q/k/v at input_layernorm; gate/up at "
                         "pre_feedforward_layernorm) BEFORE conversion. "
                         "Redistributes activation outliers into weights "
                         "so cml9 INT8 activation quant has tighter range "
                         "to fit. Requires --calib-data.")
    ap.add_argument("--joint-int8-lut", action="store_true",
                    help="Round 8 lever: after palettize_weights, run "
                         "linear_quantize_weights(joint_compression=True) so "
                         "the LUT entries are INT8 instead of FP16. Per "
                         "Apple opt-joint-compression.html. ROUND8_FINDINGS.md.")
    ap.add_argument("--awq-alpha", type=float, default=0.5,
                    help="AWQ scaling exponent. 0=no migration, 1=full "
                         "outlier migration into weights. Default 0.5 "
                         "(AWQ paper sweet spot).")
    args = ap.parse_args()

    if args.ctx is None:
        if args.model in MODEL_REGISTRY:
            args.ctx = MODEL_REGISTRY[args.model].default_context_length
        else:
            args.ctx = 2048

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    hf_dir = _resolve_hf_dir(args.model, args.hf_dir)
    print(f"Loading {args.model} from {hf_dir}...")
    t0 = time.time()
    base = Gemma4Model.from_pretrained(hf_dir, context_length=args.ctx)
    base.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    cfg = base.config
    boundaries = compute_chunk_boundaries(cfg)
    print(f"\nctx={args.ctx}  W={cfg.sliding_window}  "
          f"hidden={cfg.hidden_size}  pld={cfg.hidden_size_per_layer_input}")
    print(f"layers={cfg.num_hidden_layers}  "
          f"head_dim={cfg.head_dim}  global_head_dim={cfg.global_head_dim}  "
          f"num_kv_heads={cfg.num_key_value_heads}")
    print(f"KV producers: sliding=L{cfg.kv_sliding_producer}, "
          f"full=L{cfg.kv_full_producer}")
    print(f"Chunk boundaries: {boundaries}")
    print(f"Quantize: int{args.nbits}" if args.nbits else "Quantize: fp16")
    if args.linear_projections:
        print(f"Projections: nn.Linear (cml9 PR #2577 variant)")
    else:
        print(f"Projections: nn.Conv2d(1×1) wrapper (current default)")
    if args.activation_quant:
        if args.calib_data:
            print(f"Activations: INT8 linear_symmetric (scope={args.activation_scope}) "
                  f"using REAL calibration data {args.calib_data}")
        else:
            print(f"Activations: INT8 linear_symmetric (scope={args.activation_scope}) "
                  f"with {args.calib_samples} synthetic calib samples")

    do = (lambda n: args.only_chunk is None or args.only_chunk == n)
    use_linear = args.linear_projections
    act_q = args.activation_quant
    cdp = args.calib_data
    asc = args.activation_scope
    amd = args.activation_mode
    awq_path = args.calib_data if args.awq else None
    awq_alpha = args.awq_alpha
    if args.awq and not args.calib_data:
        raise SystemExit("--awq requires --calib-data PATH "
                         "(reuses the same .npz for activation stats)")

    if do(1):
        convert_chunk1(base, *boundaries[0], args.ctx,
                       str(out / "chunk_1.mlpackage"), args.nbits,
                       use_linear=use_linear, activation_quant=act_q,
                       calib_data_path=cdp, activation_scope=asc,
                       activation_mode=amd,
                       awq_calib_data_path=awq_path, awq_alpha=awq_alpha,
                       joint_int8_lut=args.joint_int8_lut)
    if do(2):
        convert_chunk2(base, *boundaries[1], args.ctx,
                       str(out / "chunk_2.mlpackage"), args.nbits,
                       use_linear=use_linear, activation_quant=act_q,
                       calib_data_path=cdp, activation_scope=asc,
                       activation_mode=amd)
    if do(3):
        convert_chunk_shared(SWAStatefulChunk3, base, *boundaries[2], args.ctx,
                             str(out / "chunk_3.mlpackage"), args.nbits,
                             name="CHUNK 3", with_lm_head=False,
                             use_linear=use_linear, activation_quant=act_q,
                             calib_data_path=cdp, activation_scope=asc,
                             activation_mode=amd)
    if do(4):
        convert_chunk_shared(SWAStatefulChunk4, base, *boundaries[3], args.ctx,
                             str(out / "chunk_4.mlpackage"), args.nbits,
                             name="CHUNK 4", with_lm_head=True,
                             use_linear=use_linear, activation_quant=act_q,
                             calib_data_path=cdp, activation_scope=asc,
                             activation_mode=amd)

    print(f"\nartifacts under {out}")
    for p in sorted(out.iterdir()):
        size = (sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
                if p.is_dir() else p.stat().st_size) / 1e6
        print(f"  {p.name}: {size:.1f} MB")


if __name__ == "__main__":
    main()
