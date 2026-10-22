"""Independent implementation of the sinusoidal position formula.

For position p and frequency index j, the phase is p / 10000**(2*j/D).
WavJEPA's checkpoint layout concatenates all sine channels, then all cosine
channels. Float64 evaluation followed by the caller's float32 conversion keeps
its fixed positional tensors compatible. No AudioMAE utilities are imported.
"""

import numpy as np


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Return the D sine/cosine coordinates of each flattened position."""
    if type(embed_dim) is not int or embed_dim <= 0 or embed_dim % 2:
        raise ValueError("embed_dim must be a positive even integer")
    positions = np.asarray(pos, dtype=np.float64).reshape(-1, 1)
    channels = np.arange(0, embed_dim, 2, dtype=np.float64)
    periods = np.power(10000.0, channels / embed_dim)
    phase = positions * np.reciprocal(periods)
    return np.concatenate((np.sin(phase), np.cos(phase)), axis=1)
