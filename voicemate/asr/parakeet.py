"""NVIDIA Parakeet TDT 0.6B v3 on Apple MLX (25 European languages incl. Hungarian)."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

#: Default MLX checkpoint.
DEFAULT_MODEL: str = "mlx-community/parakeet-tdt-0.6b-v3"

#: Parakeet needs at least one STFT hop of audio; shorter inputs are treated as silence.
MIN_SAMPLES: int = 1600


class ParakeetRecognizer:
    """Speech recognizer backed by ``parakeet-mlx``.

    MLX computations must stay on one thread, so call :meth:`transcribe_text` from a single
    worker thread (the voice session uses a dedicated executor).

    Args:
        model: Hugging Face repository of the MLX checkpoint.
        device: ``"cpu"`` or ``"gpu"``. CPU is the default because the GPU is shared with
            the LLM (about 280 ms instead of 160 ms for a 4 s utterance).
    """

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "cpu") -> None:
        import mlx.core as mx
        from parakeet_mlx import from_pretrained

        logger.info("Loading ASR model %s on %s", model, device)
        self._mx = mx
        self._device = mx.cpu if device == "cpu" else mx.gpu
        self._model = from_pretrained(model)
        # Warm up the kernels so the first real utterance is fast.
        self.transcribe_text(np.zeros(16000, dtype=np.float32))

    def transcribe_text(self, audio: np.ndarray) -> str:
        """Transcribe 16 kHz mono float32 audio.

        Args:
            audio: Samples in ``[-1, 1]``.

        Returns:
            The transcript (empty for silence or too-short input).
        """
        from parakeet_mlx.audio import get_logmel

        if len(audio) < MIN_SAMPLES:
            return ""
        with self._mx.stream(self._device):
            samples = self._mx.array(np.asarray(audio, dtype=np.float32))
            mel = get_logmel(samples, self._model.preprocessor_config)
            return self._model.generate(mel)[0].text.strip()
