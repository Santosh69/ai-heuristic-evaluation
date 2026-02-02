from __future__ import annotations

def patch_transformers_flash_attn_check() -> None:
    """
    Fix false-positive 'flash_attn required' ImportError that occurs when
    Transformers scans remote Florence model code on non-CUDA systems.

    This patches transformers.dynamic_module_utils.get_imports so that
    'flash_attn' is NOT treated as a hard dependency on CPU / MPS.
    """

    import torch

    # IMPORTANT:
    # If CUDA is available, DO NOT patch anything.
    # CUDA users may legitimately want flash-attn.
    if torch.cuda.is_available():
        return

    import transformers.dynamic_module_utils as dmu

    original_get_imports = dmu.get_imports

    def patched_get_imports(filename: str):
        imports = original_get_imports(filename)

        # Remove flash_attn ONLY from dependency scan
        if isinstance(imports, list) and "flash_attn" in imports:
            imports = [pkg for pkg in imports if pkg != "flash_attn"]

        return imports

    dmu.get_imports = patched_get_imports
