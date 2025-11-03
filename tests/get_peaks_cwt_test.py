import numpy as np
import pytest
from pypulsefit.patch_clamp import Sweep
import pandas as pd

@pytest.fixture
def mock_sweep():
    """Create a mock Sweep with synthetic spike data for testing."""
    fs = 10000  # 10 kHz
    duration = 1.0
    time = np.arange(0, duration, 1/fs)

    # Create low-noise signal + synthetic spikes
    signal = 0.02 * np.random.randn(len(time))
    signal[500] += 1.0
    signal[2500] += 0.8
    signal[4500] += 0.9

    # Build a DataFrame with correct column names
    df = pd.DataFrame({
        "time[s]": time,
        "trace2": signal
    })

    sweep = Sweep(name="test", data=df, recording_type=None)
    return sweep


def test_get_peaks_cwt_detects_spikes(mock_sweep):
    """Ensure get_peaks_cwt() detects the main spikes correctly."""
    peaks, response = mock_sweep.get_peaks_cwt(min_interval_ms=2.0, threshold_ratio=0.4)

    assert isinstance(peaks, np.ndarray)
    assert isinstance(response, np.ndarray)
    assert len(peaks) > 0

    # Should detect around the known spike locations
    expected_positions = [500, 2500, 4500]
    for pos in expected_positions:
        assert np.any(np.isclose(peaks, pos, atol=5)), f"Missed spike near {pos}"


def test_get_peaks_cwt_respects_min_interval(mock_sweep):
    """Check that minimum interval (refractory period) between peaks is enforced."""
    peaks, _ = mock_sweep.get_peaks_cwt(min_interval_ms=10.0)
    fs = mock_sweep.fs
    min_interval_samples = int(fs * 10e-3)

    if len(peaks) > 1:
        diffs = np.diff(peaks)
        assert np.all(diffs >= min_interval_samples), "Detected peaks too close together"


def test_get_peaks_cwt_returns_response_shape(mock_sweep):
    """Ensure response has same length as input signal."""
    peaks, response = mock_sweep.get_peaks_cwt()
    assert response.shape[0] == mock_sweep.voltage.values.shape[0]
