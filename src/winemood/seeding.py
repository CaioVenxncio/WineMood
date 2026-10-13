"""Deterministic seeding for Python, NumPy and (optionally) TensorFlow."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(
    seed: int, *, include_tensorflow: bool = True, deterministic_tf: bool = True
) -> None:
    """Seed every RNG used by the project.

    TensorFlow is imported lazily and only when ``include_tensorflow`` is true,
    so data-only commands stay fast. Full GPU determinism additionally depends on
    the hardware and cuDNN kernels; ``deterministic_tf`` requests deterministic
    ops where TF supports them.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if not include_tensorflow:
        return
    try:
        import tensorflow as tf
    except ImportError:
        return
    tf.random.set_seed(seed)
    if deterministic_tf:
        tf.config.experimental.enable_op_determinism()
