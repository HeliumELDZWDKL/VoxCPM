"""
ZipEnhancer Module - Audio Denoising Enhancer

Provides on-demand import ZipEnhancer functionality for audio denoising processing.
Related dependencies are imported only when denoising functionality is needed.
"""

import os
import tempfile
from typing import Optional
import torch
import soundfile as sf
import torchaudio

# torchaudio >= 2.11 forces torchcodec which is unavailable on Windows.
# Replace torchaudio.load/save globally with soundfile-based implementations
# so that modelscope pipeline internals also use soundfile.

def _sf_load(uri, frame_offset=0, num_frames=-1, normalize=True,
             channels_first=True, format=None, buffer_size=4096, backend=None):
    """Drop-in replacement for torchaudio.load using soundfile."""
    data, sr = sf.read(uri, start=frame_offset,
                       stop=frame_offset + num_frames if num_frames > 0 else None,
                       dtype='float32', always_2d=True)
    # data shape: (samples, channels) -> torch (channels, samples)
    tensor = torch.from_numpy(data.T)
    if not channels_first:
        tensor = tensor.T
    return tensor, sr

def _sf_save(uri, src, sample_rate, channels_first=True, format=None,
             encoding=None, bits_per_sample=None, buffer_size=4096, backend=None,
             compression=None):
    """Drop-in replacement for torchaudio.save using soundfile."""
    if channels_first:
        data = src.cpu().numpy().T  # (channels, samples) -> (samples, channels)
    else:
        data = src.cpu().numpy()
    sf.write(uri, data, sample_rate)

torchaudio.load = _sf_load
torchaudio.save = _sf_save

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


class ZipEnhancer:
    """ZipEnhancer Audio Denoising Enhancer"""

    def __init__(self, model_path: str = "iic/speech_zipenhancer_ans_multiloss_16k_base"):
        """
        Initialize ZipEnhancer
        Args:
            model_path: ModelScope model path or local path
        """
        self.model_path = model_path
        self._pipeline = pipeline(Tasks.acoustic_noise_suppression, model=self.model_path)

    def _normalize_loudness(self, wav_path: str):
        """
        Audio loudness normalization

        Args:
            wav_path: Audio file path
        """
        audio, sr = torchaudio.load(wav_path)
        loudness = torchaudio.functional.loudness(audio, sr)
        normalized_audio = torchaudio.functional.gain(audio, -20 - loudness)
        torchaudio.save(wav_path, normalized_audio, sr)

    def enhance(self, input_path: str, output_path: Optional[str] = None, normalize_loudness: bool = True) -> str:
        """
        Audio denoising enhancement
        Args:
            input_path: Input audio file path
            output_path: Output audio file path (optional, creates temp file by default)
            normalize_loudness: Whether to perform loudness normalization
        Returns:
            str: Output audio file path
        Raises:
            RuntimeError: If pipeline is not initialized or processing fails
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input audio file does not exist: {input_path}")
        # Create temporary file if no output path is specified
        if output_path is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                output_path = tmp_file.name
        try:
            # Perform denoising processing
            self._pipeline(input_path, output_path=output_path)
            # Loudness normalization
            if normalize_loudness:
                self._normalize_loudness(output_path)
            return output_path
        except Exception as e:
            # Clean up possibly created temporary files
            if output_path and os.path.exists(output_path):
                try:
                    os.unlink(output_path)
                except OSError:
                    pass
            raise RuntimeError(f"Audio denoising processing failed: {e}")
