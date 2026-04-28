import torch.nn as nn
import torchaudio

class MFCCMLP(nn.Module):
    def __init__(self, n_mfcc=40, hidden=128, n_classes=7):
        super().__init__()
        self.mfcc = torchaudio.transforms.MFCC(sample_rate=16000,
                                              n_mfcc=n_mfcc,
                                              melkwargs={'n_fft':512,
                                                         'hop_length':256,
                                                         'n_mels':64})
        self._max_len = 4 * 16000
        self.classifier = nn.Sequential(
            nn.Linear(n_mfcc *  (self._max_len // 256 + 1), hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, waveform):
        mfcc = self.mfcc(waveform.squeeze(1))
        mfcc = mfcc.reshape(mfcc.size(0), -1)
        return self.classifier(mfcc)
