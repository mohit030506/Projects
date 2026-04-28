import os
from pathlib import Path
import torch
import librosa
import numpy as np

RAVDESS_TO_FER = {
    1: 6,
    3: 3,
    4: 4,
    5: 0,
    6: 2,
    7: 1,
    8: 5,
}

class RAVDESSDataset(torch.utils.data.Dataset):
    """
    Returns (waveform_tensor, label_int) where:
      * waveform_tensor: shape (1, N) – single‑channel audio
      * label_int: integer 0‑6 matching FER‑2013 order
    """
    def __init__(self, root_dir='data/ravdess', sample_rate=16000,
                 max_len_seconds=4,
                 transform=None):
        self.root = Path(root_dir)
        self.sample_rate = sample_rate
        self.max_len = max_len_seconds * sample_rate
        self.transform = transform

        self.samples = []
        for wav_path in self.root.rglob('*.wav'):
            emo_code = int(wav_path.name.split('-')[0])
            if emo_code not in RAVDESS_TO_FER:
                continue
            label = RAVDESS_TO_FER[emo_code]
            self.samples.append((wav_path, label))

        print(f'🔊  Loaded {len(self.samples)} audio samples from {root_dir}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path, label = self.samples[idx]

        waveform, sr = librosa.load(wav_path, sr=self.sample_rate, mono=True)
        waveform = torch.from_numpy(waveform).unsqueeze(0).float()

        if waveform.shape[1] > self.max_len:
            waveform = waveform[:, :self.max_len]
        else:
            pad_len = self.max_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label
