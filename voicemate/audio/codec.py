"""PCM conversion, resampling and the binary audio frame format used on the WebSocket.

Browser → server frames are raw mono PCM16 at :data:`SAMPLE_RATE`. Server → browser frames
carry their sample rate in a 4-byte little-endian header because the TTS voices differ
(Piper 22.05 kHz, Kokoro 24 kHz).
"""

from __future__ import annotations

import struct

import numpy as np
import soxr

#: Sample rate of microphone audio and of every model input (Hz).
SAMPLE_RATE: int = 16_000

_PCM_SCALE = 32767.0
_HEADER = struct.Struct("<I")


def pcm16_to_float32(pcm: bytes) -> np.ndarray:
    """Convert little-endian signed 16-bit PCM bytes to float32 samples in ``[-1, 1]``.

    Args:
        pcm: Raw PCM bytes (even length).

    Returns:
        1-D float32 array.

    Raises:
        ValueError: If the byte count is odd.
    """
    if len(pcm) % 2:
        raise ValueError(f"PCM16 payload must have an even byte count, got {len(pcm)}")
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / _PCM_SCALE


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert float samples to little-endian signed 16-bit PCM, clipping to ``[-1, 1]``.

    Args:
        audio: 1-D float array.

    Returns:
        Raw PCM bytes.
    """
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (clipped * _PCM_SCALE).astype("<i2").tobytes()


def resample(audio: np.ndarray, rate_in: int, rate_out: int) -> np.ndarray:
    """Resample mono audio with a high-quality polyphase resampler.

    Args:
        audio: 1-D float array.
        rate_in: Input sample rate (Hz).
        rate_out: Output sample rate (Hz).

    Returns:
        Float32 array at ``rate_out`` (the input object itself when rates match).
    """
    if rate_in == rate_out:
        return audio
    return soxr.resample(np.asarray(audio, dtype=np.float32), rate_in, rate_out).astype(np.float32)


def pack_audio_frame(pcm: bytes, sample_rate: int) -> bytes:
    """Prefix PCM16 bytes with their sample rate for the browser player."""
    return _HEADER.pack(sample_rate) + pcm


def unpack_audio_frame(frame: bytes) -> tuple[int, bytes]:
    """Split a packed frame into ``(sample_rate, pcm)``.

    Raises:
        ValueError: If the frame is shorter than its header.
    """
    if len(frame) < _HEADER.size:
        raise ValueError("Audio frame shorter than its header")
    (sample_rate,) = _HEADER.unpack_from(frame)
    return sample_rate, frame[_HEADER.size :]
