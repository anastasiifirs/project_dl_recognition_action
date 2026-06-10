import numpy as np

from event_video_recognition.repetitions import count_peaks, smooth_signal


def test_count_peaks_counts_periodic_motion():
    x = np.linspace(0, 8 * np.pi, 80)
    signal = smooth_signal((np.sin(x) + 1.0).tolist(), window=3)

    count, confidence = count_peaks(signal, min_distance=8, prominence_std=0.2)

    assert 3 <= count <= 5
    assert confidence > 0


def test_count_peaks_returns_zero_for_flat_signal():
    count, confidence = count_peaks(np.ones(20), min_distance=4, prominence_std=0.2)

    assert count == 0
    assert confidence == 0
