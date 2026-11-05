"""Graph rewrites that make ONNX stem models run correctly and safely on ONNX Runtime's DirectML provider.

Used by ``scripts/patch_onnx_for_directml.py``. Every pass is semantics-preserving on CPU.
"""
