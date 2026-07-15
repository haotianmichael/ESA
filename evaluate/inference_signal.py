"""
Signal-domain inference model, the analogue of ``inference_models.EvalModel``.

``SignalEvalModel.encode(list_of_signals) -> np.ndarray[N, D]`` runs the same
preprocessing as training (z-normalize + fixed-length window + T-resolution
mask), pushes windows through the shared ``SignalEncoder`` + pooler, and returns
L2-normalized embeddings. Note the normalization is ``dim=-1`` (per-vector),
unlike the batch-flattening ``dim=0`` in the legacy text ``EvalModel``.

Because it exposes ``encode`` and ``get_sentence_embedding_dimension``, it drops
straight into ``FaissStore`` (via ``SignalFaissStore``) without touching it.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch

from dna2vec.signal_dataset import preprocess_window


class SignalEvalModel:
    def __init__(
        self,
        encoder,
        pooling,
        device,
        input_signal_len: int = 2000,
        downsample_factor: int = 5,
        embedding_dim: int = 384,
        batch_size: int = 256,
    ):
        self.encoder = encoder.to(device)
        self.pooling = pooling.to(device)
        self.device = device
        self.input_signal_len = input_signal_len
        self.downsample_factor = downsample_factor
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.encoder.eval()

    def get_sentence_embedding_dimension(self) -> int:
        return self.embedding_dim

    def _prep_batch(self, signals: List[np.ndarray]):
        sigs, masks = [], []
        for s in signals:
            sig, mask = preprocess_window(s, self.input_signal_len, self.downsample_factor)
            sigs.append(sig)
            masks.append(mask)
        signal = torch.from_numpy(np.stack(sigs)).float().to(self.device)
        attention_mask = torch.from_numpy(np.stack(masks)).long().to(self.device)
        return signal, attention_mask

    def encode(self, signals: List[np.ndarray]) -> np.ndarray:
        outputs = []
        with torch.no_grad():
            for start in range(0, len(signals), self.batch_size):
                chunk = signals[start : start + self.batch_size]
                signal, attention_mask = self._prep_batch(chunk)
                last_hidden = self.encoder(signal=signal, attention_mask=attention_mask)
                y = self.pooling(last_hidden, attention_mask=attention_mask)
                y = torch.nn.functional.normalize(y, dim=-1)
                outputs.append(y.detach().cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32)
