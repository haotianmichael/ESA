"""
Siamese positive-pair dataset for signal-domain contrastive training.

A positive pair is::

    x_1 = { 'signal': query_real_signal_window,   'attention_mask': mask_T }
    x_2 = { 'signal': ref_expected_signal_window, 'attention_mask': mask_T }

where the reference window ``x_2`` is produced by
``PoreModel.sequence_to_signal(reference[coord:coord+win_bp])`` and ``coord`` is
the ground-truth genomic start of the query read. Both sides go through the same
``SignalEncoder``; InfoNCE then pulls the two domains (noisy recorded vs. clean
expected) into one space.

Windows are fixed length (``input_signal_len`` samples), so the down-sampled
attention mask is well defined and aligns with the encoder output T; shorter
signals are zero-padded and the mask marks the valid region.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset


# --------------------------------------------------------------------------- #
# Shared preprocessing (also used at inference time)
# --------------------------------------------------------------------------- #
def znormalize(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size == 0:
        return signal
    mean = float(signal.mean())
    std = float(signal.std())
    if std < 1e-6:
        std = 1e-6
    return (signal - mean) / std


def fix_length(signal: np.ndarray, target_len: int) -> Tuple[np.ndarray, int]:
    """Truncate or zero-pad ``signal`` to ``target_len``; return (array, valid_len)."""
    signal = np.asarray(signal, dtype=np.float32)
    n = int(signal.shape[0])
    if n >= target_len:
        return signal[:target_len].copy(), target_len
    out = np.zeros(target_len, dtype=np.float32)
    out[:n] = signal
    return out, n


def make_mask_T(valid_len: int, input_signal_len: int, downsample_factor: int) -> np.ndarray:
    """Build a T-resolution mask for a window with ``valid_len`` valid samples."""
    T = input_signal_len // downsample_factor
    valid_T = min(T, max(1, valid_len // downsample_factor))
    mask = np.zeros(T, dtype=np.int64)
    mask[:valid_T] = 1
    return mask


def preprocess_window(
    signal: np.ndarray, input_signal_len: int, downsample_factor: int
) -> Tuple[np.ndarray, np.ndarray]:
    """z-normalize -> fix length -> (signal[input_signal_len], mask[T])."""
    sig, valid = fix_length(znormalize(signal), input_signal_len)
    mask = make_mask_T(valid, input_signal_len, downsample_factor)
    return sig, mask


# --------------------------------------------------------------------------- #
class SignalPairDataset(IterableDataset):
    def __init__(
        self,
        query_signals: List[np.ndarray],
        query_coords: List[int],
        reference_seq: str,
        pore_model,
        unit_length: int,
        input_signal_len: int = 2000,
        downsample_factor: int = 5,
        samples_per_kmer: int = 9,
    ):
        super().__init__()
        assert len(query_signals) == len(query_coords)
        self.query_signals = query_signals
        self.query_coords = query_coords
        self.reference_seq = reference_seq
        self.pore_model = pore_model
        self.input_signal_len = input_signal_len
        self.downsample_factor = downsample_factor
        # The training reference span MUST equal the index window span
        # (``unit_length``) — otherwise the encoder learns to match a reference
        # vector length that does not exist in the FAISS index, and the true
        # window is pushed out of the top-k (recall can drop below random).
        self.win_bp = unit_length

    def _pair(self, idx: int) -> Tuple[Dict, Dict]:
        coord = int(self.query_coords[idx])

        q_sig, q_mask = preprocess_window(
            self.query_signals[idx], self.input_signal_len, self.downsample_factor
        )

        ref_bases = self.reference_seq[coord : coord + self.win_bp]
        ref_signal = self.pore_model.sequence_to_signal(ref_bases)
        r_sig, r_mask = preprocess_window(
            ref_signal, self.input_signal_len, self.downsample_factor
        )

        x_1 = {"signal": q_sig, "attention_mask": q_mask}
        x_2 = {"signal": r_sig, "attention_mask": r_mask}
        return x_1, x_2

    def __iter__(self):
        n = len(self.query_signals)
        while True:
            idx = int(torch.randint(0, n, (1,)).item())
            yield self._pair(idx)


def signal_collate(batch) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Stack a list of (x_1, x_2) pairs into batched tensor dicts."""
    x_1_list, x_2_list = list(zip(*batch))

    def stack(dicts):
        signals = torch.from_numpy(np.stack([d["signal"] for d in dicts])).float()
        masks = torch.from_numpy(np.stack([d["attention_mask"] for d in dicts])).long()
        return {"signal": signals, "attention_mask": masks}

    return stack(x_1_list), stack(x_2_list)
