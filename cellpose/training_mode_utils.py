from __future__ import annotations

from typing import Any, Optional

from cellpose import models


def configure_trainable_params(
    net: Any,
    *,
    use_lora: bool,
    lora_blocks: Optional[int],
    unfreeze_blocks: Optional[int],
    logger=None,
):
    """
    Apply a consistent trainability policy for local and remote training.

    Policy:
    - Always freeze all params first.
    - Always unfreeze output head (`net.out`) when present.
    - If `use_lora=True`, inject LoRA into last `lora_blocks` encoder blocks.
    - Else, unfreeze last `unfreeze_blocks` encoder blocks.
    """
    # 1) Freeze all base params.
    for p in net.parameters():
        p.requires_grad = False

    # 2) Always train output head.
    if hasattr(net, "out"):
        for p in net.out.parameters():
            p.requires_grad = True

    result = {
        "mode": "lora" if use_lora else "non_lora",
        "lora_info": None,
        "applied_unfreeze_blocks": 0,
        "total_encoder_blocks": 0,
    }

    if use_lora:
        blocks = 9 if lora_blocks is None else int(lora_blocks)
        if logger is not None:
            logger.info("GUI_INFO: Converting model to LoRA for training")
        lora_info = models.convert_to_lora(net, n_last_blocks=blocks)
        result["lora_info"] = lora_info
        return result

    # Non-LoRA path: selectively unfreeze last N encoder blocks.
    if not hasattr(net, "encoder") or not hasattr(net.encoder, "blocks"):
        return result

    total_blocks = len(net.encoder.blocks)
    result["total_encoder_blocks"] = total_blocks
    n_unfreeze = 0 if unfreeze_blocks is None else int(unfreeze_blocks)
    if n_unfreeze > total_blocks and logger is not None:
        logger.info(
            "Requested %s unfreeze blocks, but model has %s; using full encoder.",
            n_unfreeze,
            total_blocks,
        )
    n_unfreeze = min(max(0, n_unfreeze), total_blocks)
    result["applied_unfreeze_blocks"] = n_unfreeze
    if n_unfreeze > 0:
        for blk in net.encoder.blocks[-n_unfreeze:]:
            for p in blk.parameters():
                p.requires_grad = True
    return result

