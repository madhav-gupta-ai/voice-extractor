"""Environmental-noise removal: identify the noise types, then gate them out."""

import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf
import torch

from .models import FEATURE_SHAPE, NOISE_CLASSES, NOISE_DIR, load_classifier

# Reference noise recordings used as reduction profiles for each class.
# gun_shot ships no profile of its own; the nearest transient profiles are used.
NOISE_PROFILES = {
    "air_conditioner": ["ac1.wav", "ac2.wav"],
    "car_horn": ["horn1.wav", "horn2.wav"],
    "children_playing": ["children1.wav", "children2.wav"],
    "dog_bark": ["bark1.wav", "bark2.wav"],
    "drilling": ["drill1.wav", "drill2.wav"],
    "engine_idling": ["engine1.wav", "engine2.wav"],
    "gun_shot": ["engine1.wav", "drill2.wav"],
    "jackhammer": ["jack1.wav", "jack2.wav"],
    "siren": ["siren1.wav", "siren2.wav"],
    "street_music": ["street1.wav", "street2.wav"],
}


def extract_features(y, sr):
    """Stack five 40-coefficient feature vectors into the classifier's 40x5 input."""
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, fmax=8000).T, axis=0)
    stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=40).T, axis=0)
    cqt = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=40, bins_per_octave=40).T, axis=0)
    cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr, n_chroma=40, bins_per_octave=40).T, axis=0)
    return np.vstack((mfcc, mel, stft, cqt, cens)).reshape(FEATURE_SHAPE)


def identify_noises(y, sr, model, top_k=2):
    features = extract_features(y, sr).astype("float32")
    batch = torch.from_numpy(features).reshape(1, 1, *FEATURE_SHAPE)
    with torch.no_grad():
        probabilities = torch.softmax(model(batch)[0], dim=0)
    top = torch.topk(probabilities, top_k)
    return [NOISE_CLASSES[i] for i in top.indices.tolist()]


def denoise_file(input_path, output_path, model_path):
    """Write a copy of ``input_path`` with its detected environmental noises reduced."""
    model = load_classifier(model_path)
    y, sr = librosa.load(str(input_path))
    detected = identify_noises(y, sr, model)
    print(f"detected noise types: {', '.join(detected)}")
    for name in detected:
        for profile in NOISE_PROFILES[name]:
            noise, _ = librosa.load(str(NOISE_DIR / profile), sr=sr)
            y = nr.reduce_noise(y=y, y_noise=noise, sr=sr)
    sf.write(str(output_path), y, sr)
    return output_path
