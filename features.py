import numpy as np
import librosa
import pywt
from scipy.stats import kurtosis, skew

from config import DWT_LEVEL, FEATURE_COUNT, N_MFCC, SEGMENT_DURATION, TARGET_SR, WAVELET


def load_audio(path, target_sr=TARGET_SR):
    signal, sr = librosa.load(path, sr=target_sr, mono=True)
    return signal.astype(np.float32), sr


def segment_signal(signal, sr, segment_duration=SEGMENT_DURATION):
    segment_length = int(segment_duration * sr)
    if segment_length <= 0:
        raise ValueError("segment_duration must produce a positive segment length")

    return [
        signal[start:start + segment_length]
        for start in range(0, len(signal) - segment_length + 1, segment_length)
    ]


def extract_mfcc_features(segment, sr, n_mfcc=N_MFCC):
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    return np.concatenate([mfcc_mean, mfcc_std]).astype(np.float32)


def extract_dwt_features(segment, wavelet=WAVELET, level=DWT_LEVEL):
    coeffs = pywt.wavedec(segment, wavelet=wavelet, level=level)
    features = []

    for coeff in coeffs:
        coeff = np.asarray(coeff, dtype=np.float32)
        energy = float(np.sum(coeff ** 2))
        if energy <= 0:
            entropy = 0.0
        else:
            probability = (coeff ** 2) / energy
            probability = probability[probability > 0]
            entropy = float(-np.sum(probability * np.log(probability)))

        values = [
            energy,
            entropy,
            float(np.mean(coeff)),
            float(np.std(coeff)),
            float(skew(coeff)),
            float(kurtosis(coeff)),
        ]
        features.extend(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0))

    return np.array(features, dtype=np.float32)


def extract_dwt_mfcc_features(segment, sr):
    dwt_features = extract_dwt_features(segment)
    mfcc_features = extract_mfcc_features(segment, sr)
    features = np.concatenate([dwt_features, mfcc_features]).astype(np.float32)
    if features.shape[0] != FEATURE_COUNT:
        raise ValueError(f"Expected {FEATURE_COUNT} features, got {features.shape[0]}.")
    return features


def extract_audio_feature_matrix(path):
    signal, sr = load_audio(path)
    segments = segment_signal(signal, sr)
    if not segments:
        raise ValueError("Audio is shorter than the minimum 4-second segment length.")

    features = [extract_dwt_mfcc_features(segment, sr) for segment in segments]
    if not features:
        raise ValueError("No valid audio segment could be processed.")

    return np.vstack(features), {
        "sample_rate": sr,
        "segment_count": len(features),
        "processed_duration_sec": len(features) * SEGMENT_DURATION,
    }
