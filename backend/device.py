"""Torch device auto-detection. Applies to YOLO only — SAM3 runs on ONNXRuntime (CPU)."""
import logging

import torch

logger = logging.getLogger(__name__)


def get_device() -> "torch.device":
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    logger.info("Using device (YOLO): %s", dev)
    return dev
