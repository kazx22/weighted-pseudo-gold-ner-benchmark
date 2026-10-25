"""Strict CPU-only inference for *pretrained, off-the-shelf* NER models.

This module is not used for fine-tuning or for the Ollama MedGemma service.
"""
from __future__ import annotations

import os


def enforce_cpu():
    # This happens before importing transformers / torch in calling modules.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    os.environ["CAWV_NER_CPU_ONLY"] = "1"


def verify_cpu(ner, model_name: str):
    import torch

    pipeline_device = torch.device(ner.device)
    params = list(ner.model.parameters())
    model_devices = {p.device.type for p in params}
    if pipeline_device.type != "cpu" or model_devices != {"cpu"}:
        raise RuntimeError(
            f"OFF-THE-SHELF CPU REQUIREMENT VIOLATED: {model_name}: "
            f"pipeline={pipeline_device}, model parameters={model_devices}"
        )
    ner.model.eval()
    print(f"[OFF-THE-SHELF CPU VERIFIED] {model_name}: pipeline=cpu, weights=cpu, gradients=off during inference", flush=True)
    return ner


def cpu_pipeline(*args, **kwargs):
    enforce_cpu()
    from transformers import pipeline

    kwargs["device"] = -1
    ner = pipeline(*args, **kwargs)
    return verify_cpu(ner, str(kwargs.get("model", "off-the-shelf NER")))
