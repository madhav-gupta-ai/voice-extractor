"""Noise-type classifiers and shared constants."""

from pathlib import Path

import torch
from torch import nn

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
NOISE_DIR = ASSETS_DIR / "noise"
DEFAULT_MODEL = ASSETS_DIR / "denoiser_cnn.pt"

SEED = 42

# UrbanSound8K classes, indexed by classID.
NOISE_CLASSES = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
]

# Each sample is a 40x5 matrix: 40 coefficients for each of five feature types
# (MFCC, mel spectrogram, chroma-STFT, chroma-CQT, chroma-CENS).
FEATURE_SHAPE = (40, 5)


class CNNClassifier(nn.Module):
    """Baseline: two convolution blocks followed by dense layers."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True),  # (40, 5) -> (20, 3)
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True),  # (20, 3) -> (10, 2)
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(128 * 10 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, len(NOISE_CLASSES)),
        )

    def forward(self, x):
        return self.net(x)


class UNetClassifier(nn.Module):
    """Single-level convolutional encoder-decoder ahead of a linear head."""

    def __init__(self):
        super().__init__()
        self.pad = nn.ZeroPad2d((0, 1, 0, 0))  # width 5 -> 6 so pooling divides evenly
        self.encode = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (40, 6) -> (20, 3)
        )
        self.decode = nn.Sequential(
            nn.Upsample(scale_factor=2),  # (20, 3) -> (40, 6)
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 40 * 6, len(NOISE_CLASSES)),
        )

    def forward(self, x):
        return self.head(self.decode(self.encode(self.pad(x))))


ARCHITECTURES = {"cnn": CNNClassifier, "unet": UNetClassifier}


def save_classifier(model, arch, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": arch, "state_dict": model.state_dict()}, path)


def load_classifier(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = ARCHITECTURES[checkpoint["arch"]]()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
