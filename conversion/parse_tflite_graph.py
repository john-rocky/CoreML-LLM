#!/usr/bin/env python3
"""Parse TFLite operator graph to trace norm weight → operator connections.

Run with Python 3.12:
    python3.12 conversion/parse_tflite_graph.py
"""
from __future__ import annotations
import sys
import tflite


def main():
    tflite_path = "output/mtp_probe/section_10.tflite"

    with open(tflite_path, "rb") as f:
        buf = f.read()

    model = tflite.Model.GetRootAs(buf, 0)

    # Build op code table
    opcodes = {}
    for i in range(model.OperatorCodesLength()):
        oc = model.OperatorCodes(i)
        # Check builtin vs custom
        builtin = oc.DeprecatedBuiltinCode()
        custom = oc.CustomCode()
        if custom:
            opcodes[i] = custom.decode()
        else:
            builtin2 = oc.BuiltinCode()
            opcodes[i] = f"BUILTIN_{builtin2}"

    # For each subgraph, trace operators
    print(f"Subgraphs: {model.SubgraphsLength()}")
    print(f"Op codes: {len(opcodes)}")

    # Focus on the main subgraph (0)
    sg = model.Subgraphs(0)
    print(f"\nMain subgraph: {sg.TensorsLength()} tensors, {sg.OperatorsLength()} operators")

    # Build tensor name lookup
    tensor_names = {}
    tensor_shapes = {}
    for ti in range(sg.TensorsLength()):
        t = sg.Tensors(ti)
        name = t.Name().decode() if t.Name() else f"t_{ti}"
        shape = tuple(t.Shape(d) for d in range(t.ShapeLength()))
        tensor_names[ti] = name
        tensor_shapes[ti] = shape

    # Find all jax2tf_arg tensors
    arg_tensors = {}
    for ti, name in tensor_names.items():
        if "jax2tf_arg_" in name:
            arg_num = int(name.split("jax2tf_arg_")[1].split("/")[0])
            arg_tensors[arg_num] = ti

    print(f"\nNorm arg tensors: {sorted(arg_tensors.keys())}")

    # For each operator, check if it uses any arg tensor as input
    # and trace what it produces
    print("\n=== Operators using norm weights ===\n")

    for oi in range(sg.OperatorsLength()):
        op = sg.Operators(oi)
        opcode_idx = op.OpcodeIndex()
        opname = opcodes.get(opcode_idx, f"op_{opcode_idx}")

        inputs = [op.Inputs(i) for i in range(op.InputsLength())]
        outputs = [op.Outputs(i) for i in range(op.OutputsLength())]

        # Check if any input is a jax2tf_arg tensor
        for inp_ti in inputs:
            if inp_ti in [arg_tensors[k] for k in arg_tensors]:
                arg_num = [k for k, v in arg_tensors.items() if v == inp_ti][0]
                input_names = [f"{tensor_names.get(i, '?')}:{tensor_shapes.get(i, '?')}"
                              for i in inputs]
                output_names = [f"{tensor_names.get(o, '?')}:{tensor_shapes.get(o, '?')}"
                               for o in outputs]
                print(f"  Op {oi} ({opname}): arg_{arg_num}")
                # Show all inputs and outputs
                for inp in input_names:
                    # Truncate long names
                    if len(inp) > 80:
                        inp = inp[:77] + "..."
                    print(f"    IN:  {inp}")
                for out in output_names:
                    if len(out) > 80:
                        out = out[:77] + "..."
                    print(f"    OUT: {out}")
                print()
                break

    # Also trace the MLP-related operators to understand the full flow
    print("\n=== Key operator sequence for layer 0 ===\n")
    for oi in range(sg.OperatorsLength()):
        op = sg.Operators(oi)
        opcode_idx = op.OpcodeIndex()
        opname = opcodes.get(opcode_idx, f"op_{opcode_idx}")

        inputs = [op.Inputs(i) for i in range(op.InputsLength())]
        outputs = [op.Outputs(i) for i in range(op.OutputsLength())]

        # Show operators that mention layer_0 in their tensor names
        all_tensors = inputs + outputs
        for ti in all_tensors:
            name = tensor_names.get(ti, "")
            if "layer_0" in name and ("rms_norm" in name.lower() or "gating" in name
                                     or "linear" in name or "dot_general" in name
                                     or "add" in name.lower()):
                short_inputs = [f"t{i}({tensor_names.get(i, '?')[:40]})" for i in inputs]
                short_outputs = [f"t{o}({tensor_names.get(o, '?')[:40]})" for o in outputs]
                print(f"  Op {oi:3d} {opname[:30]:30s} "
                      f"in=[{','.join(short_inputs)}] "
                      f"out=[{','.join(short_outputs)}]")
                break


if __name__ == "__main__":
    main()
