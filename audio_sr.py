"""
AP-BWE (Amplitude and Phase Bandwidth Extension) for VoxCPM2.
Standalone audio super-resolution module adapted from GPT-SoVITS/AP-BWE.
Usage: 48kHz → downsample to 24kHz → AP-BWE upsample back to 48kHz with cleaner high-freq.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as aF
import numpy as np


# ─── STFT helpers ───────────────────────────────────────────────

def amp_pha_stft(audio, n_fft, hop_size, win_size, center=True):
    hann_window = torch.hann_window(win_size).to(audio.device)
    stft_spec = torch.stft(
        audio, n_fft, hop_length=hop_size, win_length=win_size,
        window=hann_window, center=center, pad_mode="reflect",
        normalized=False, return_complex=True,
    )
    log_amp = torch.log(torch.abs(stft_spec) + 1e-4)
    pha = torch.angle(stft_spec)
    com = torch.stack((torch.exp(log_amp) * torch.cos(pha),
                       torch.exp(log_amp) * torch.sin(pha)), dim=-1)
    return log_amp, pha, com


def amp_pha_istft(log_amp, pha, n_fft, hop_size, win_size, center=True):
    amp = torch.exp(log_amp)
    com = torch.complex(amp * torch.cos(pha), amp * torch.sin(pha))
    hann_window = torch.hann_window(win_size).to(com.device)
    audio = torch.istft(com, n_fft, hop_length=hop_size,
                        win_length=win_size, window=hann_window, center=center)
    return audio


# ─── ConvNeXt block ────────────────────────────────────────────

def _get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, layer_scale_init_value=None, adanorm_num_embeddings=None):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, dim * 3)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dim * 3, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value and layer_scale_init_value > 0
            else None
        )

    def forward(self, x, cond_embedding_id=None):
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)
        return residual + x


# ─── APNet BWE Model ──────────────────────────────────────────

class APNet_BWE_Model(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        layer_scale_init_value = 1 / h["ConvNeXt_layers"]

        self.conv_pre_mag = nn.Conv1d(h["n_fft"] // 2 + 1, h["ConvNeXt_channels"], 7, 1,
                                      padding=_get_padding(7, 1))
        self.norm_pre_mag = nn.LayerNorm(h["ConvNeXt_channels"], eps=1e-6)
        self.conv_pre_pha = nn.Conv1d(h["n_fft"] // 2 + 1, h["ConvNeXt_channels"], 7, 1,
                                      padding=_get_padding(7, 1))
        self.norm_pre_pha = nn.LayerNorm(h["ConvNeXt_channels"], eps=1e-6)

        self.convnext_mag = nn.ModuleList([
            ConvNeXtBlock(dim=h["ConvNeXt_channels"], layer_scale_init_value=layer_scale_init_value)
            for _ in range(h["ConvNeXt_layers"])
        ])
        self.convnext_pha = nn.ModuleList([
            ConvNeXtBlock(dim=h["ConvNeXt_channels"], layer_scale_init_value=layer_scale_init_value)
            for _ in range(h["ConvNeXt_layers"])
        ])

        self.norm_post_mag = nn.LayerNorm(h["ConvNeXt_channels"], eps=1e-6)
        self.norm_post_pha = nn.LayerNorm(h["ConvNeXt_channels"], eps=1e-6)
        self.apply(self._init_weights)
        self.linear_post_mag = nn.Linear(h["ConvNeXt_channels"], h["n_fft"] // 2 + 1)
        self.linear_post_pha_r = nn.Linear(h["ConvNeXt_channels"], h["n_fft"] // 2 + 1)
        self.linear_post_pha_i = nn.Linear(h["ConvNeXt_channels"], h["n_fft"] // 2 + 1)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, mag_nb, pha_nb):
        x_mag = self.conv_pre_mag(mag_nb)
        x_pha = self.conv_pre_pha(pha_nb)
        x_mag = self.norm_pre_mag(x_mag.transpose(1, 2)).transpose(1, 2)
        x_pha = self.norm_pre_pha(x_pha.transpose(1, 2)).transpose(1, 2)

        for conv_block_mag, conv_block_pha in zip(self.convnext_mag, self.convnext_pha):
            x_mag = x_mag + x_pha
            x_pha = x_pha + x_mag
            x_mag = conv_block_mag(x_mag)
            x_pha = conv_block_pha(x_pha)

        x_mag = self.norm_post_mag(x_mag.transpose(1, 2))
        mag_wb = mag_nb + self.linear_post_mag(x_mag).transpose(1, 2)

        x_pha = self.norm_post_pha(x_pha.transpose(1, 2))
        x_pha_r = self.linear_post_pha_r(x_pha)
        x_pha_i = self.linear_post_pha_i(x_pha)
        pha_wb = torch.atan2(x_pha_i, x_pha_r).transpose(1, 2)

        com_wb = torch.stack((
            torch.exp(mag_wb) * torch.cos(pha_wb),
            torch.exp(mag_wb) * torch.sin(pha_wb)
        ), dim=-1)

        return mag_wb, pha_wb, com_wb


# ─── High-level wrapper ───────────────────────────────────────

class AudioSuperResolution:
    """Audio super-resolution via AP-BWE (24kHz→48kHz bandwidth extension)."""

    def __init__(self, device="cuda", checkpoint_dir=None):
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(os.path.dirname(__file__), "ap_bwe_checkpoints", "24kto48k")

        checkpoint_file = os.path.join(checkpoint_dir, "g_24kto48k")
        config_file = os.path.join(checkpoint_dir, "config.json")

        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(
                f"AP-BWE checkpoint not found: {checkpoint_file}\n"
                "Download from: https://drive.google.com/drive/folders/1IIYTf2zbJWzelu4IftKD6ooHloJ8mnZF"
            )

        with open(config_file) as f:
            self.h = json.load(f)

        model = APNet_BWE_Model(self.h).to(device)
        state_dict = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict["generator"])
        model.eval()
        self.model = model
        self.device = device

    @torch.inference_mode()
    def __call__(self, audio_np: np.ndarray, orig_sr: int = 48000) -> tuple:
        """
        Apply audio super-resolution.

        Args:
            audio_np: 1D numpy array (float32), typically 48kHz from VoxCPM2
            orig_sr: original sample rate

        Returns:
            (enhanced_audio_np, output_sr) where output_sr is always 48000
        """
        audio = torch.from_numpy(audio_np).float().unsqueeze(0).to(self.device)

        # Resample to the model's expected input rate (24kHz for narrowband)
        lr_sr = self.h["lr_sampling_rate"]  # 24000
        hr_sr = self.h["hr_sampling_rate"]  # 48000
        audio_lr = aF.resample(audio, orig_freq=orig_sr, new_freq=lr_sr)
        # Upsample back to hr_sr for STFT (model works at hr_sr resolution)
        audio_input = aF.resample(audio_lr, orig_freq=lr_sr, new_freq=hr_sr)

        amp_nb, pha_nb, com_nb = amp_pha_stft(
            audio_input, self.h["n_fft"], self.h["hop_size"], self.h["win_size"]
        )
        amp_wb_g, pha_wb_g, com_wb_g = self.model(amp_nb, pha_nb)
        audio_hr = amp_pha_istft(
            amp_wb_g, pha_wb_g, self.h["n_fft"], self.h["hop_size"], self.h["win_size"]
        )

        result = audio_hr.squeeze().cpu().numpy()
        # Normalize to prevent clipping
        max_val = np.abs(result).max()
        if max_val > 1.0:
            result = result / max_val

        return result, hr_sr
