import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import find_peaks, find_peaks_cwt
from typing import Tuple
from pypulsefit.patch_clamp import Dataset, RecordingType


def cwt_detect_peaks(
    signal: np.ndarray,
    fs: float,
    min_width_ms: float = 0.3,
    max_width_ms: float = 3.0,
    wavelet: str = "mexh",
    detector: str = "sum",  # "max", "sum", "multiscale_product"
    scale_step: float = 1.0,
    noise_portion: float = 0.1,
    peak_prominence: float | None = None,
    peak_distance_ms: float = 1.0,
) -> dict:
    """
    Detect peaks using PyWavelets CWT with improved heuristics.

    Parameters
    ----------
    signal : np.ndarray
        1D signal (voltage).
    fs : float
        Sampling freq (Hz).
    min_width_ms, max_width_ms : float
        Expected spike width range (ms). Use physiologically plausible values.
    detector : str
        "max", "sum" or "multiscale_product".
    noise_portion : float
        Portion of signal used to estimate noise (start of trace).
    peak_prominence : float|None
        If provided, passed to find_peaks.
    peak_distance_ms : float
        Minimum distance between peaks (ms).

    Returns
    -------
    dict with keys: cwt_response, peaks_idx, peaks_time, scales, cwtmatr, threshold, noise_est
    """
    # compute scales from expected widths (approx)
    # approximate relation: width_in_samples ~ scale  (depends on wavelet)
    # we choose scales proportional to sample widths
    min_scale = max(1, int(round(min_width_ms * fs / 1000.0)))
    max_scale = max(min_scale + 1, int(round(max_width_ms * fs / 1000.0)))
    scales = np.arange(min_scale, max_scale + 1, scale_step)

    # compute CWT matrix (scales x time)
    cwtmatr, freqs = pywt.cwt(signal, scales, wavelet)  # shape: (len(scales), len(signal))

    # Normalize each scale row (robust): divide by std or MAD to equalize scales
    # Use robust std: 1) convert row to abs, compute mad/std
    normed = np.empty_like(cwtmatr)
    for i in range(cwtmatr.shape[0]):
        row = cwtmatr[i]
        # robust scale estimate
        mad = np.median(np.abs(row - np.median(row)))
        scale_std = mad / 0.6745 if mad > 0 else np.std(row) + 1e-12
        normed[i] = row / (scale_std + 1e-12)

    # Build detector response across scales
    if detector == "max":
        cwt_response = np.max(np.abs(normed), axis=0)
    elif detector == "sum":
        cwt_response = np.sum(np.abs(normed), axis=0)
    elif detector == "multiscale_product":
        # product of absolute responses; use +eps and log to avoid underflow
        eps = 1e-12
        cwt_response = np.prod(np.abs(normed) + eps, axis=0)
        # optional log to compress dynamic range
        cwt_response = np.log(cwt_response + 1e-12)
    else:
        raise ValueError("detector must be 'max', 'sum' or 'multiscale_product'")

    # Estimate noise on detector -> robust MAD on portion considered quiet
    n_quiet = max(1, int(len(cwt_response) * noise_portion))
    quiet_segment = cwt_response[:n_quiet]
    mad = np.median(np.abs(quiet_segment - np.median(quiet_segment)))
    noise_est = mad / 0.6745 if mad > 0 else np.std(quiet_segment)

    # threshold: factor * noise_est (try 3..6)
    threshold = 4.0 * noise_est

    threshold = np.max(cwt_response)*0.35

    # find peaks with prominence and min distance (in samples)
    distance = int(round(peak_distance_ms * fs / 1000.0))
    peaks_kwargs = {"height": threshold, "distance": distance}
    if peak_prominence is not None:
        peaks_kwargs["prominence"] = peak_prominence

    peaks, props = find_peaks(cwt_response, **peaks_kwargs)

    return {
        "cwt_response": cwt_response,
        "peaks_idx": peaks,
        "peaks_time_s": peaks / fs,
        "scales": scales,
        "cwtmatr": cwtmatr,
        "threshold": threshold,
        "noise_est": noise_est,
        "detector": detector,
        "props": props,
    }


# ---------------- Example usage, compare detectors ----------------
# assume sweep is your Sweep object, with sweep.voltage (np.ndarray or pd.Series) and sweep.fs
# signal = sweep.voltage.values - sweep.baseline_v  # recommended preproc
# fs = sweep.fs

def compare_cwt_detectors(signal: np.ndarray, fs: float, **kwargs):
    detectors = ["max"]
    results = {}
    for d in detectors:
        res = cwt_detect_peaks(signal, fs, detector=d, **kwargs)
        results[d] = res

    # plot
    t = np.arange(len(signal)) / fs
    fig, axes = plt.subplots(len(detectors)+1, 1, figsize=(12, 3*(len(detectors)+1)), sharex=True)
    axes[0].plot(t, signal, label="signal")
    axes[0].set_title("Signal (preprocessed)")
    for i, d in enumerate(detectors, start=1):
        r = results[d]
        axes[i].plot(t, r["cwt_response"], label=f"CWT response ({d})")
        axes[i].axhline(r["threshold"], color="k", ls="--", label=f"thr={r['threshold']:.3f}")
        axes[i].plot(r["peaks_idx"]/fs, r["cwt_response"][r["peaks_idx"]], "ro", label="peaks")
        axes[i].legend()
        axes[i].set_ylabel("CWT resp")
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

    for d in detectors:
        print(f"{d}: detected {len(results[d]['peaks_idx'])} peaks (threshold={results[d]['threshold']:.3f})")

    return results


# --- Load data ---
data = Dataset.from_asc(
    "../ANALYSE Anna/GH4C1/2024/06062024/cell12 PA induits.asc",
    recording_type=RecordingType.INTRACELLULAR,
)
sweep = data.get_sweep_by_index(0)

# --- Filter signal ---
lpf = 200
hpf = None
order_lpf = 5
order_hpf = 2
sweep.filter_v(lpf=lpf, hpf=hpf, order_lpf=order_lpf, order_hpf=order_hpf)
fig,ax = plt.subplots()
sweep.plot(ax)
# --- Extract signal and remove baseline ---
signal = sweep.voltage.values - sweep.baseline_v
signal = np.clip(signal, a_min=0, a_max=None)
#noise_level = np.std(signal) * 0.002  # 1% du bruit total
#noise = np.random.normal(0, noise_level, size=signal.shape)
#signal = np.where(signal > 0, signal, noise)


# ---------------- Quick defaults you can try ----------------


results = compare_cwt_detectors(signal, sweep.fs,
                                 min_width_ms=20, max_width_ms=400,
                                 noise_portion=0.1,
                                 peak_prominence=None,
                                 peak_distance_ms=300.0)
