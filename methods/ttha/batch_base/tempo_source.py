"""Vendored TempoPFN numerical augmentation functions (DEV-TEMPO-AUG-SOURCE-ALIGNMENT, task doc §4).

Source:   TempoPFN, https://github.com/automl/TempoPFN, commit 5969ec6ddbded5a8d2976d42504c1ae436a2f1b3 (2025-11-10),
          local read-only clone ../TempoPFN. Licensed under the Apache License, Version 2.0; a copy of the license is in
          TEMPOPFN_LICENSE next to this file. Copyright of the copied code remains with the TempoPFN authors.
Copied VERBATIM (bodies byte-identical to the source lines; generated from the clone, not retyped):
  src/synthetic_generation/augmentations/offline_per_sample_iid_augmentations.py
      UnivariateOfflineAugmentor._apply_regime_change                   lines 505-556
      UnivariateOfflineAugmentor._apply_shock_recovery                  lines 558-589
      UnivariateOfflineAugmentor._apply_calendar_injections             lines 591-662
      UnivariateOfflineAugmentor._apply_seasonality_amplitude_modulation lines 664-684
      UnivariateOfflineAugmentor._apply_resample_artifacts              lines 686-732
  src/data/augmentations.py
      CensorAugmenter                                                   lines 319-371
      RandomConvAugmenter                                               lines 1013-1213
Modifications (the only ones; listed as required by Apache-2.0 section 4(b)):
  M1  The five UnivariateOfflineAugmentor methods are hosted on `SourceAugmentor`, whose only state is `rng` (the numpy
      Generator the original methods read as self.rng). The original UnivariateOfflineAugmentor.__init__ (which reseeds the global
      numpy / torch RNGs and builds unrelated augmenters) and apply() are NOT copied; composition lives in tempo_aug.py.
  M2  `Frequency` / `parse_frequency` used by _apply_calendar_injections are an HOURLY-ONLY subset of src/data/frequency.py written
      here (below). For "h"/"H" they return what the original returns (Frequency.H, pandas alias "h"; checked by the package
      smoke against the original module); any other frequency raises instead of the original's silent daily fallback.
  M3  No function body, default, probability or distribution is changed. Randomness: SourceAugmentor methods draw from
      self.rng; CensorAugmenter / RandomConvAugmenter draw from the GLOBAL torch and legacy numpy RNGs exactly as in the
      source -- tempo_aug.Streams swaps those global states in and out around each call (RNG isolation outside the copy).
Nothing else from TempoPFN (generators, mixup, models, CLIs, weights) is copied or imported.
"""
from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class Frequency(Enum):
    """M2: hourly-only subset of TempoPFN src/data/frequency.py Frequency (members referenced by _apply_calendar_injections)."""

    A = "A"
    Q = "Q"
    M = "M"
    W = "W"
    D = "D"
    H = "h"
    S = "s"
    T1 = "1min"
    T5 = "5min"
    T10 = "10min"
    T15 = "15min"
    T30 = "30min"

    def to_pandas_freq(self, for_date_range: bool = True) -> str:
        if self is not Frequency.H:
            raise ValueError("vendored subset supports hourly data only")
        return "h"  # original: FREQUENCY_MAPPING[Frequency.H] = ("h", "", 1/24) -> prefix "" + base "h"


def parse_frequency(freq_str: str) -> Frequency:
    """M2: hourly-only subset of TempoPFN src/data/frequency.py parse_frequency."""
    if str(freq_str) in ("h", "H"):
        return Frequency.H
    raise ValueError("vendored subset supports hourly data only: %r" % (freq_str,))


class SourceAugmentor:
    """M1: host of the five verbatim UnivariateOfflineAugmentor methods; `rng` is a numpy Generator (np.random.default_rng)."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def _apply_regime_change(self, series: torch.Tensor, p_apply: float) -> torch.Tensor:
        """
        Apply piecewise affine transforms with 1-3 change-points per series.
        series shape: [batch, length, 1]
        """
        if series.numel() == 0:
            return series
        batch_size, length, _ = series.shape
        result = series.clone()

        # Iterate per-series to allow different change-points
        for b in range(batch_size):
            if self.rng.random() >= p_apply:
                continue
            # sample number of change points and ensure minimum segment length
            num_cp = int(self.rng.integers(1, 4))
            min_seg = max(8, length // 32)
            if length <= (num_cp + 1) * min_seg:
                num_cp = max(1, length // (2 * min_seg) - 1)
            if num_cp <= 0:
                num_cp = 1
            # pick change-point indices
            valid_positions = np.arange(min_seg, length - min_seg)
            if valid_positions.size == 0:
                continue
            cp = np.sort(self.rng.choice(valid_positions, size=num_cp, replace=False))
            boundaries = np.concatenate([[0], cp, [length]])

            # compute per-segment scale/shift
            series_b = result[b, :, 0]
            seg_scales = []
            seg_shifts = []
            overall_std = torch.std(series_b).item()
            if not np.isfinite(overall_std) or overall_std == 0:
                overall_std = 1.0
            for _ in range(len(boundaries) - 1):
                scale = float(self.rng.uniform(0.8, 1.25))
                shift = float(self.rng.normal(0.0, 0.15 * overall_std))
                seg_scales.append(scale)
                seg_shifts.append(shift)

            # apply per segment
            for i in range(len(boundaries) - 1):
                s, e = int(boundaries[i]), int(boundaries[i + 1])
                if e <= s:
                    continue
                segment = series_b[s:e]
                # preserve segment mean roughly while scaling deviations
                seg_mean = torch.mean(segment)
                transformed = (segment - seg_mean) * seg_scales[i] + seg_mean + seg_shifts[i]
                result[b, s:e, 0] = transformed
        return result

    def _apply_shock_recovery(self, series: torch.Tensor, p_apply: float) -> torch.Tensor:
        """
        Add an impulse at a random time and exponentially decay to baseline.
        series shape: [batch, length, 1]
        """
        if series.numel() == 0:
            return series
        batch_size, length, _ = series.shape
        device = series.device
        result = series.clone()

        time_idx = torch.arange(length, device=device).float()

        for b in range(batch_size):
            if self.rng.random() >= p_apply:
                continue
            # choose shock time away from edges
            t0 = int(self.rng.integers(low=max(1, length // 16), high=max(2, length - length // 16)))
            # magnitude relative to series std
            s_b = result[b, :, 0]
            std_b = torch.std(s_b).item()
            if not np.isfinite(std_b) or std_b == 0:
                std_b = 1.0
            mag = float(self.rng.uniform(0.5, 2.0) * std_b)
            if self.rng.random() < 0.5:
                mag = -mag
            # decay constant
            half_life = float(self.rng.uniform(0.03, 0.25) * length)
            decay = torch.exp(-(time_idx - t0).clamp(min=0) / max(1.0, half_life))
            effect = mag * decay
            result[b, :, 0] = s_b + effect
        return result

    def _apply_calendar_injections(
        self,
        series: torch.Tensor,
        starts: list[pd.Timestamp] | None,
        frequencies: list[str] | None,
        p_apply: float,
    ) -> torch.Tensor:
        if series.numel() == 0:
            return series
        if starts is None or frequencies is None:
            return series
        batch_size, length, _ = series.shape
        result = series.clone()

        for b in range(batch_size):
            if b >= len(starts) or b >= len(frequencies):
                continue
            if self.rng.random() >= p_apply:
                continue
            start_ts = starts[b]
            try:
                freq_enum = parse_frequency(str(frequencies[b]))
                freq_alias = freq_enum.to_pandas_freq(for_date_range=True)
            except Exception:
                freq_alias = "D"
            try:
                index = pd.date_range(start=start_ts, periods=length, freq=freq_alias)
            except Exception:
                index = pd.date_range(start=start_ts, periods=length, freq="D")

            factors = np.ones(length, dtype=np.float32)
            # Weekend dips (for daily/hourly-like)
            try:
                freq_enum_check = parse_frequency(str(frequencies[b]))
            except Exception:
                freq_enum_check = Frequency.D
            if freq_enum_check in [
                Frequency.H,
                Frequency.D,
                Frequency.S,
                Frequency.T1,
                Frequency.T5,
                Frequency.T10,
                Frequency.T15,
                Frequency.T30,
            ]:
                dow = index.dayofweek
                if (dow >= 5).any():
                    dip = float(self.rng.uniform(0.7, 0.95))
                    factors[dow >= 5] *= dip

            # Month-end bumps
            if hasattr(index, "is_month_end"):
                me = np.asarray(index.is_month_end, dtype=bool)
                if me.any():
                    bump = float(self.rng.uniform(1.05, 1.3))
                    factors[me] *= bump

            # Holiday-like one-off effects (1-2 random impulses)
            n_imp = int(self.rng.integers(1, 3))
            imp_positions = self.rng.integers(0, length, size=n_imp)
            for pos in np.atleast_1d(imp_positions):
                if 0 <= pos < length:
                    impulse = float(self.rng.uniform(0.8, 1.4))
                    factors[pos] *= impulse

            # Apply multiplicatively around mean to avoid drift
            s = result[b, :, 0].cpu().numpy()
            mean_val = float(np.mean(s))
            s_new = (s - mean_val) * factors + mean_val
            result[b, :, 0] = torch.from_numpy(s_new).to(result.device)
        return result

    def _apply_seasonality_amplitude_modulation(self, series: torch.Tensor, p_apply: float) -> torch.Tensor:
        if series.numel() == 0:
            return series
        batch_size, length, _ = series.shape
        result = series.clone()

        for b in range(batch_size):
            if self.rng.random() >= p_apply:
                continue
            min_w = max(8, length // 16)
            max_w = max(min_w + 1, length // 2)
            win = int(self.rng.integers(min_w, max_w + 1))
            start = int(self.rng.integers(0, max(1, length - win)))
            end = start + win
            seg = result[b, start:end, 0]
            if seg.numel() == 0:
                continue
            seg_mean = torch.mean(seg)
            amp = float(self.rng.uniform(0.5, 1.8))
            result[b, start:end, 0] = (seg - seg_mean) * amp + seg_mean
        return result

    def _apply_resample_artifacts(
        self,
        series: torch.Tensor,
        p_apply: float,
    ) -> torch.Tensor:
        """
        Downsample then upsample with interpolation to introduce artifacts.
        """
        if series.numel() == 0:
            return series
        batch_size, length, _ = series.shape
        result = series.clone()

        for b in range(batch_size):
            if self.rng.random() >= p_apply:
                continue

            s_np = result[b, :, 0].cpu().numpy()
            max_factor = max(2, min(8, length // 32))
            if max_factor <= 1:
                continue
            factor = int(self.rng.integers(2, max_factor + 1))
            offset = int(self.rng.integers(0, factor))
            ds_idx = np.arange(offset, length, factor)
            if ds_idx.size < 3:
                continue
            ds_vals = s_np[ds_idx]
            base_idx = np.arange(length)
            mode = self.rng.choice(["linear", "hold", "linear_smooth"], p=[0.5, 0.2, 0.3])
            if mode == "linear":
                us = np.interp(base_idx, ds_idx, ds_vals)
            elif mode == "hold":
                us = np.empty(length, dtype=s_np.dtype)
                last = ds_vals[0]
                j = 0
                for i in range(length):
                    while j + 1 < ds_idx.size and i >= ds_idx[j + 1]:
                        j += 1
                        last = ds_vals[j]
                    us[i] = last
            else:
                us = np.interp(base_idx, ds_idx, ds_vals)
                k = max(3, length // 128)
                kernel = np.ones(k) / k
                us = np.convolve(us, kernel, mode="same")
            result[b, :, 0] = torch.from_numpy(us).to(result.device)
        return result


class CensorAugmenter:
    """
    Applies censor augmentation by clipping values from above, below, or both.
    """

    def __init__(self):
        """Initializes the CensorAugmenter."""
        pass

    def transform(self, time_series_batch: torch.Tensor) -> torch.Tensor:
        """
        Applies a vectorized censor augmentation to a batch of time series.
        """
        batch_size, seq_len, num_channels = time_series_batch.shape
        assert num_channels == 1
        time_series_batch = time_series_batch.squeeze(-1)
        with torch.no_grad():
            batch_size, seq_len = time_series_batch.shape
            device = time_series_batch.device

            # Step 1: Choose an op mode for each series
            op_mode = torch.randint(0, 3, (batch_size, 1), device=device)

            # Step 2: Calculate potential thresholds for all series
            q1 = torch.rand(batch_size, device=device)
            q2 = torch.rand(batch_size, device=device)
            q_low = torch.minimum(q1, q2)
            q_high = torch.maximum(q1, q2)

            sorted_series = torch.sort(time_series_batch, dim=1).values
            indices_low = (q_low * (seq_len - 1)).long()
            indices_high = (q_high * (seq_len - 1)).long()

            c_low = torch.gather(sorted_series, 1, indices_low.unsqueeze(1))
            c_high = torch.gather(sorted_series, 1, indices_high.unsqueeze(1))

            # Step 3: Compute results for all possible clipping operations
            clip_above = torch.minimum(time_series_batch, c_high)
            clip_below = torch.maximum(time_series_batch, c_low)

            # Step 4: Select the final result based on the op_mode
            result = torch.where(
                op_mode == 1,
                clip_above,
                torch.where(op_mode == 2, clip_below, time_series_batch),
            )
            augmented_batch = torch.where(
                op_mode == 0,
                time_series_batch,
                result,
            )

        return augmented_batch.unsqueeze(-1)


class RandomConvAugmenter:
    """
    Applies a stack of 1-to-N random 1D convolutions to a time series batch.

    This augmenter is inspired by the principles of ROCKET and RandConv,
    randomizing nearly every aspect of the convolution process to create a
    highly diverse set of transformations. This version includes multiple
    kernel generation strategies, random padding modes, and optional non-linearities.
    """

    def __init__(
        self,
        p_transform: float = 0.5,
        kernel_size_range: tuple[int, int] = (3, 31),
        dilation_range: tuple[int, int] = (1, 8),
        layer_range: tuple[int, int] = (1, 3),
        sigma_range: tuple[float, float] = (0.5, 5.0),
        bias_range: tuple[float, float] = (-0.5, 0.5),
    ):
        """
        Initializes the augmenter.

        Args:
            p_transform (float): Probability of applying the augmentation to a series.
            kernel_size_range (Tuple[int, int]): [min, max] range for kernel sizes.
                                                 Must be odd numbers.
            dilation_range (Tuple[int, int]): [min, max] range for dilation factors.
            layer_range (Tuple[int, int]): [min, max] range for the number of
                                           stacked convolution layers.
            sigma_range (Tuple[float, float]): [min, max] range for the sigma of
                                               Gaussian kernels.
            bias_range (Tuple[float, float]): [min, max] range for the bias term.
        """
        assert kernel_size_range[0] % 2 == 1 and kernel_size_range[1] % 2 == 1, "Kernel sizes must be odd."

        self.p_transform = p_transform
        self.kernel_size_range = kernel_size_range
        self.dilation_range = dilation_range
        self.layer_range = layer_range
        self.sigma_range = sigma_range
        self.bias_range = bias_range
        self.padding_modes = ["reflect", "replicate", "circular"]

    def _rescale_signal(self, processed_signal: torch.Tensor, original_signal: torch.Tensor) -> torch.Tensor:
        """Rescales the processed signal to match the min/max range of the original."""
        original_min = torch.amin(original_signal, dim=-1, keepdim=True)
        original_max = torch.amax(original_signal, dim=-1, keepdim=True)
        processed_min = torch.amin(processed_signal, dim=-1, keepdim=True)
        processed_max = torch.amax(processed_signal, dim=-1, keepdim=True)

        original_range = original_max - original_min
        processed_range = processed_max - processed_min
        epsilon = 1e-8

        is_flat = processed_range < epsilon

        rescaled_signal = (
            (processed_signal - processed_min) / (processed_range + epsilon)
        ) * original_range + original_min

        original_mean = torch.mean(original_signal, dim=-1, keepdim=True)
        flat_rescaled = original_mean.expand_as(original_signal)

        return torch.where(is_flat, flat_rescaled, rescaled_signal)

    def _apply_random_conv_stack(self, series: torch.Tensor) -> torch.Tensor:
        """
        Applies a randomly configured stack of convolutions to a single time series.

        Args:
            series (torch.Tensor): A single time series of shape (1, num_channels, seq_len).

        Returns:
            torch.Tensor: The augmented time series.
        """
        num_channels = series.shape[1]
        device = series.device

        num_layers = torch.randint(self.layer_range[0], self.layer_range[1] + 1, (1,)).item()

        processed_series = series
        for i in range(num_layers):
            # 1. Sample kernel size
            k_min, k_max = self.kernel_size_range
            kernel_size = torch.randint(k_min // 2, k_max // 2 + 1, (1,)).item() * 2 + 1

            # 2. Sample dilation
            d_min, d_max = self.dilation_range
            dilation = torch.randint(d_min, d_max + 1, (1,)).item()

            # 3. Sample bias
            b_min, b_max = self.bias_range
            bias_val = (b_min + (b_max - b_min) * torch.rand(1)).item()

            # 4. Sample padding mode
            padding_mode = np.random.choice(self.padding_modes)

            conv_layer = nn.Conv1d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding="same",  # Let PyTorch handle padding calculation
                padding_mode=padding_mode,
                groups=num_channels,
                bias=True,
                device=device,
            )

            # 5. Sample kernel weights from a wider variety of types
            weight_type = torch.randint(0, 4, (1,)).item()
            if weight_type == 0:  # Gaussian kernel
                s_min, s_max = self.sigma_range
                sigma = (s_min + (s_max - s_min) * torch.rand(1)).item()
                ax = torch.arange(
                    -(kernel_size // 2),
                    kernel_size // 2 + 1,
                    device=device,
                    dtype=torch.float32,
                )
                kernel = torch.exp(-0.5 * (ax / sigma) ** 2)
            elif weight_type == 1:  # Standard normal kernel
                kernel = torch.randn(kernel_size, device=device)
            elif weight_type == 2:  # Polynomial kernel
                coeffs = torch.randn(3, device=device)  # a, b, c for ax^2+bx+c
                x_vals = torch.linspace(-1, 1, kernel_size, device=device)
                kernel = coeffs[0] * x_vals**2 + coeffs[1] * x_vals + coeffs[2]
            else:  # Noisy Sobel kernel
                # Ensure kernel is large enough for a Sobel filter
                actual_kernel_size = 3 if kernel_size < 3 else kernel_size
                sobel_base = torch.tensor([-1, 0, 1], dtype=torch.float32, device=device)
                noise = torch.randn(3, device=device) * 0.1
                noisy_sobel = sobel_base + noise
                # Pad if the random kernel size is larger than 3
                pad_total = actual_kernel_size - 3
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                kernel = F.pad(noisy_sobel, (pad_left, pad_right), "constant", 0)

            # 6. Probabilistic normalization
            if torch.rand(1).item() < 0.8:  # 80% chance to normalize
                kernel /= torch.sum(torch.abs(kernel)) + 1e-8

            kernel = kernel.view(1, 1, -1).repeat(num_channels, 1, 1)

            conv_layer.weight.data = kernel
            conv_layer.bias.data.fill_(bias_val)
            conv_layer.weight.requires_grad = False
            conv_layer.bias.requires_grad = False

            # Apply convolution
            processed_series = conv_layer(processed_series)

            # 7. Optional non-linearity (not on the last layer)
            if i < num_layers - 1:
                activation_type = torch.randint(0, 3, (1,)).item()
                if activation_type == 1:
                    processed_series = F.relu(processed_series)
                elif activation_type == 2:
                    processed_series = torch.tanh(processed_series)
                # if 0, do nothing (linear)

        return processed_series

    def transform(self, time_series_batch: torch.Tensor) -> torch.Tensor:
        """Applies a random augmentation to a subset of the batch."""
        with torch.no_grad():
            if self.p_transform == 0:
                return time_series_batch

            batch_size, seq_len, num_channels = time_series_batch.shape
            device = time_series_batch.device

            augment_mask = torch.rand(batch_size, device=device) < self.p_transform
            indices_to_augment = torch.where(augment_mask)[0]
            num_to_augment = indices_to_augment.numel()

            if num_to_augment == 0:
                return time_series_batch

            subset_to_augment = time_series_batch[indices_to_augment]

            subset_permuted = subset_to_augment.permute(0, 2, 1)

            augmented_subset_list = []
            for i in range(num_to_augment):
                original_series = subset_permuted[i : i + 1]
                augmented_series = self._apply_random_conv_stack(original_series)

                rescaled_series = self._rescale_signal(augmented_series.squeeze(0), original_series.squeeze(0))
                augmented_subset_list.append(rescaled_series.unsqueeze(0))

            if augmented_subset_list:
                augmented_subset = torch.cat(augmented_subset_list, dim=0)
                augmented_subset_final = augmented_subset.permute(0, 2, 1)

                augmented_batch = time_series_batch.clone()
                augmented_batch[indices_to_augment] = augmented_subset_final
                return augmented_batch
            else:
                return time_series_batch
