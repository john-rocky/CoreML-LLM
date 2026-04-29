#!/usr/bin/env python3
"""Produce a sideload-ready bundle for LFM2 / LFM2.5 models.

Output layout (matches what `Sources/CoreMLLLM/CoreMLLLM.swift::load`
expects for monolithic models):

    output/<folder_name>/bundle/
      ├── model.mlmodelc            (compiled — first launch is instant)
      ├── model_config.json         (with lfm2_conv_l_pad echoed in)
      └── hf_model/                 (tokenizer files copied from HF)
          ├── tokenizer.json
          ├── tokenizer_config.json (sanitised — see below)
          └── ... (other tokenizer assets)

Why sanitise tokenizer_config.json: the upstream file ships
``"tokenizer_class": "TokenizersBackend"`` which is a transformers v5
class.  swift-transformers (and transformers ≤ 4.x on Mac) reject it.
We rewrite to ``PreTrainedTokenizerFast`` which both stacks understand;
the underlying ``tokenizer.json`` is bit-identical.

Usage:
    python conversion/build_lfm2_bundle.py --model lfm2.5-350m
    # or with an explicit HF dir / output name:
    python conversion/build_lfm2_bundle.py \\
        --model-path ./output/lfm2.5-350m/hf_model \\
        --output ./output/lfm2.5-350m/bundle \\
        --quantize none
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys


def _maybe_run_convert(model: str | None, model_path: str | None,
                        ctx: int, quantize: str, output_root: str) -> str:
    """Run convert.py if the mlpackage doesn't already exist; return its path."""
    pkg = os.path.join(output_root, "model.mlpackage")
    if os.path.exists(pkg):
        return pkg

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    from convert import main as convert_main

    argv = ["convert.py", "--context-length", str(ctx),
            "--quantize", quantize, "--output", output_root]
    if model_path:
        argv += ["--model-path", model_path]
    elif model:
        argv += ["--model", model]
    else:
        raise SystemExit("either --model or --model-path is required")

    saved = sys.argv
    try:
        sys.argv = argv
        convert_main()
    finally:
        sys.argv = saved
    return pkg


def _compile_to_mlmodelc(pkg: str, dst_dir: str) -> str:
    """Compile a .mlpackage and copy the result into dst_dir/model.mlmodelc.

    Uses ``xcrun coremlcompiler compile`` rather than going through the
    Python coremltools framework loader, because the LFM2 INT4-palettized
    mlpackage trips a Core ML framework load error (-14) that doesn't
    affect the on-device runtime — the iPhone is fine, the Mac SDK
    framework loader chokes.  ``coremlcompiler`` runs the same MIL-→
    -mlmodelc transform without trying to instantiate the model.
    """
    import subprocess

    print(f"Compiling {pkg} → {dst_dir}/model.mlmodelc")
    cmd = ["xcrun", "coremlcompiler", "compile", pkg, dst_dir]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"coremlcompiler failed (rc={proc.returncode}):\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
    # coremlcompiler writes <dst_dir>/<pkg-stem>.mlmodelc — rename to
    # canonical "model.mlmodelc" if needed.
    expected = os.path.join(dst_dir, "model.mlmodelc")
    if not os.path.isdir(expected):
        # find the produced .mlmodelc and rename it.
        for name in os.listdir(dst_dir):
            if name.endswith(".mlmodelc"):
                src_path = os.path.join(dst_dir, name)
                if src_path != expected:
                    if os.path.exists(expected):
                        shutil.rmtree(expected)
                    shutil.move(src_path, expected)
                break
    return expected


def _sanitise_tokenizer_config(src: str, dst: str) -> None:
    """Copy tokenizer_config.json with TokenizersBackend → PreTrainedTokenizerFast."""
    with open(src) as f:
        cfg = json.load(f)
    if cfg.get("tokenizer_class") == "TokenizersBackend":
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
    cfg.pop("backend", None)  # also a v5-ism
    with open(dst, "w") as f:
        json.dump(cfg, f, indent=2)


def _copy_tokenizer(hf_src: str, dst_root: str) -> None:
    dst_hf = os.path.join(dst_root, "hf_model")
    os.makedirs(dst_hf, exist_ok=True)
    keep = {
        "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
        "special_tokens_map.json", "added_tokens.json",
        "chat_template.jinja", "generation_config.json", "config.json",
    }
    for name in os.listdir(hf_src):
        if name not in keep:
            continue
        src = os.path.join(hf_src, name)
        dst = os.path.join(dst_hf, name)
        if name == "tokenizer_config.json":
            _sanitise_tokenizer_config(src, dst)
        else:
            shutil.copy2(src, dst)
    print(f"Copied tokenizer assets to {dst_hf}")


def _patch_model_config(out_root: str, hf_src: str, l_pad: int = 16) -> None:
    """Echo lfm2-specific knobs into model_config.json so the Swift loader
    doesn't have to peek at the mlpackage spec."""
    cfg_path = os.path.join(out_root, "model_config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["lfm2_conv_l_pad"] = l_pad
    cfg["model_name"] = cfg.get("model_name") or os.path.basename(hf_src)
    # tokenizer_repo points at the bundled hf_model so swift-transformers can
    # find it offline.
    cfg["tokenizer_repo"] = "hf_model"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Patched {cfg_path}: lfm2_conv_l_pad={l_pad}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lfm2.5-350m",
                    help="registry id (default: lfm2.5-350m)")
    ap.add_argument("--model-path", default=None,
                    help="optional HF model dir (skip download)")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--quantize", default="none",
                    choices=["none", "int4", "int8"],
                    help="default 'none' (fp16) — confirmed working on CPU; "
                         "int4 builds but Mac SDK can't validate (iPhone OK)")
    ap.add_argument("--source-mlpackage", default=None,
                    help="skip convert.py and use this prebuilt mlpackage")
    ap.add_argument("--output", default=None,
                    help="bundle output dir; default: output/<model>/bundle")
    ap.add_argument("--l-pad", type=int, default=16,
                    help="LFM2 conv state padded width (must match converter)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    work_root = os.path.join(repo, "output", args.model)
    bundle_root = args.output or os.path.join(work_root, "bundle")
    os.makedirs(bundle_root, exist_ok=True)

    if args.source_mlpackage:
        pkg = args.source_mlpackage
        if not os.path.exists(pkg):
            raise SystemExit(f"--source-mlpackage not found: {pkg}")
    else:
        pkg = _maybe_run_convert(
            args.model, args.model_path,
            ctx=args.ctx, quantize=args.quantize,
            output_root=work_root,
        )

    # Bundle layout:
    #   bundle/model.mlmodelc
    #   bundle/model_config.json
    #   bundle/hf_model/...
    _compile_to_mlmodelc(pkg, bundle_root)

    src_cfg = os.path.join(work_root, "model_config.json")
    shutil.copy2(src_cfg, os.path.join(bundle_root, "model_config.json"))

    hf_src = (args.model_path
              or os.path.join(work_root, "hf_model"))
    _copy_tokenizer(hf_src, bundle_root)

    _patch_model_config(bundle_root, hf_src, l_pad=args.l_pad)

    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(bundle_root)
        for f in fns
    ) / 1024 / 1024
    print(f"\nBundle ready at {bundle_root} ({size_mb:.1f} MB)")
    print("Sideload to iPhone with e.g.")
    print(f"  xcrun devicectl device copy to --device <id> \\")
    print(f"      --domain-type appDataContainer \\")
    print(f"      --domain-identifier com.example.CoreMLLLMChat \\")
    print(f"      --source {bundle_root} \\")
    print(f"      --destination Documents/Models/{args.model} \\")
    print(f"      --remove-existing-content true")


if __name__ == "__main__":
    main()
