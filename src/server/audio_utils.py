"""
Audio utility functions shared across modules.
"""

import numpy as np


def float32_to_int16(audio: np.ndarray) -> np.ndarray:
    """Convert float32 PCM [-1, 1] to int16 PCM [-32768, 32767].

    Allocates a single int16 array (no intermediate float copies).
    """
    return np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)


def int16_to_float32(audio: np.ndarray) -> np.ndarray:
    """Convert int16 PCM to float32 PCM [-1, 1]."""
    return audio.astype(np.float32) / 32768.0
