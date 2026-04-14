"""Transformer package with lazy symbol loading."""

from importlib import import_module

_SYMBOL_REGISTRY = {
    "IntermediateSequential": (".intmd_sequential", "IntermediateSequential"),
    "FixedPositionalEncoding": (".positional_encoding", "FixedPositionalEncoding"),
    "LearnedPositionalEncoding": (".positional_encoding", "LearnedPositionalEncoding"),
    "SETR_Naive": (".setr", "SETR_Naive"),
    "SETR_PUP": (".setr", "SETR_PUP"),
    "SETR_MLA": (".setr", "SETR_MLA"),
    "SegmentationTransformer": (".setr", "SegmentationTransformer"),
    "PatchShifting": (".spt", "PatchShifting"),
    "ShiftedPatchTokenization": (".spt", "ShiftedPatchTokenization"),
    "SelfAttention": (".transformer_core", "SelfAttention"),
    "Residual": (".transformer_core", "Residual"),
    "PreNorm": (".transformer_core", "PreNorm"),
    "PreNormDrop": (".transformer_core", "PreNormDrop"),
    "FeedForward": (".transformer_core", "FeedForward"),
    "TransformerModel": (".transformer_core", "TransformerModel"),
}

__all__ = sorted(_SYMBOL_REGISTRY.keys())


def __getattr__(name: str):
    if name not in _SYMBOL_REGISTRY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, symbol_name = _SYMBOL_REGISTRY[name]
    module = import_module(module_path, package=__name__)
    symbol = getattr(module, symbol_name)
    globals()[name] = symbol
    return symbol
