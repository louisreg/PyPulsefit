import numpy as np
import pytest
from pypulsefit.patch_clamp import Sweep


import numpy as np
import pandas as pd
import pytest
from pypulsefit.patch_clamp import Sweep


def noise_std_mad(x: np.ndarray) -> float:
    """Robust noise estimator using the Median Absolute Deviation (MAD)."""
    mad = np.median(np.abs(x - np.median(x)))
    return mad / 0.6745


@pytest.fixture
def mock_sweep():
    """Create a mock Sweep object with two clear peaks."""
    fs = 10000  # 10 kHz sampling rate
    t = np.arange(0, 1, 1 / fs)  # 1 second
    signal = np.sin(2 * np.pi * 5 * t)
    signal[500] += 2.0
    signal[2500] += 1.5

    # Use DataFrame instead of ndarray
    df = pd.DataFrame({
        "time[s]": t,
        "trace2": signal
    })

    sweep = Sweep(
        name="test",
        data=df,
        recording_type=None,
        metadata={}
    )

    return sweep


def test_get_peaks_detects_known_peaks(mock_sweep):
    """Ensure get_peaks() correctly identifies inserted peaks."""
    peaks, thr = mock_sweep.get_peaks(threshold=1.5, min_interval_ms=5.0)
    assert isinstance(peaks, np.ndarray)
    # Peaks should be near sample 500 and 2500
    assert np.any(np.isclose(peaks, 500, atol=5))
    assert np.any(np.isclose(peaks, 2500, atol=5))
    assert len(peaks) <= 3


def test_get_peaks_auto_threshold(mock_sweep):
    """Ensure automatic threshold estimation works properly."""
    peaks, thr = mock_sweep.get_peaks(min_interval_ms=5.0)
    assert isinstance(thr, float)
    assert len(peaks) > 0


