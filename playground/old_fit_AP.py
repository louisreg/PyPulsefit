from pypulsefit.utils import list_asc_files, get_AP_summary
from pypulsefit.patch_clamp import Dataset, RecordingType
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, lfilter, detrend
import pywt
from scipy.optimize import curve_fit

# --- 1️⃣ DoG model ---
def spike_model_DoG(t, V0, A1, sigma1, A2, sigma2, delta):
    """
    Difference-of-Gaussians model for a spike
    - V0 : baseline
    - A1, sigma1 : amplitude and width of depolarizing (positive) component
    - A2, sigma2, delta : amplitude, width, and time offset of hyperpolarizing (negative) component
    """
    #V0= 0
    t = np.asarray(t)
    if v is not None:
        t_peak = t[np.argmax(v)]
    else:
        t_peak = t[len(t)//2]  # fallback
    y = V0 \
        + A1 * np.exp(-(t - t_peak)**2 / (2*sigma1**2)) \
        - A2 * np.exp(-((t - t_peak - delta)**2) / (2*sigma2**2))
    return y

# --- 2️⃣ Fit function ---
def fit_spike_DoG(t, v):
    """
    Fit spike waveform to a Difference-of-Gaussians model.
    """
    # Initial guess
    V0 = np.mean(v)
    A1 = np.max(v) 
    sigma1 = 1   # ms, positive (rising) component
    A2 = A1 / 2
    sigma2 = 1.0   # ms, negative (falling) component
    delta = 15.0     # ms, offset of negative component

    p0 = [V0, A1, sigma1, A2, sigma2, delta]

    # Bounds
    bounds = (
        [-0.002, 0.99*A1, 0.5, 0, 0.01, 5],  # lower
        [+0.002, 1.1*A1, 50, 2*A2, 20, 25]  # upper
    )

    try:
        popt, _ = curve_fit(spike_model_DoG, t, v, p0=p0, bounds=bounds, maxfev=20000)
        fit_curve = spike_model_DoG(t, *popt)
        params = {
            "V0": popt[0],
            "A1": popt[1],
            "sigma1": popt[2],
            "A2": popt[3],
            "sigma2": popt[4],
            "delta": popt[5],
        }
        return params, fit_curve
    except Exception as e:
        print(f"⚠️ DoG fit failed: {e}")
        return None, None

def extract_window(voltage, peak_idx, fs, pre_ms=5.0, post_ms=15.0):
    """
    Extract a window around peak_idx from voltage (1D array).
    Returns (start_idx, end_idx, window_array).
    Handles edges by clipping.
    """
    voltage = np.asarray(voltage)
    n = len(voltage)
    pre_samp = int(round(pre_ms * fs / 1000.0))
    post_samp = int(round(post_ms * fs / 1000.0))
    start = max(0, peak_idx - pre_samp)
    end = min(n - 1, peak_idx + post_samp)
    window = voltage[start:end + 1].copy()
    return start, end, window


def remove_baseline(window, fs, method='mean', poly_order=2,
                    highpass_cutoff=1.0, hp_order=3,
                    wavelet='db4', wavelet_level=3):
    """
    Removes the baseline from a signal window using different methods.
    Returns (corrected_signal, estimated_baseline).
    """
    w = np.asarray(window).astype(float)
    n = len(w)

    if method == 'mean':
        baseline = np.mean(w)
        corrected = w - baseline
        return corrected, np.full_like(w, baseline)

    elif method == 'detrend':
        baseline_est = w - detrend(w)
        corrected = detrend(w)
        return corrected, baseline_est

    elif method == 'poly':
        x = np.linspace(0, 1, n)
        coeffs = np.polyfit(x, w, poly_order)
        baseline = np.polyval(coeffs, x)
        corrected = w - baseline
        return corrected, baseline

    elif method == 'highpass':
        nyq = 0.5 * fs
        cutoff_norm = highpass_cutoff / nyq
        b, a = butter(hp_order, cutoff_norm, btype='high')
        try:
            filtered = filtfilt(b, a, w)
        except ValueError:
            filtered = lfilter(b, a, w)
        baseline = w - filtered
        corrected = filtered
        return corrected, baseline

    elif method == 'wavelet':
        coeffs = pywt.wavedec(w, wavelet, level=wavelet_level)
        approx = coeffs.copy()
        approx[1:] = [np.zeros_like(c) for c in approx[1:]]
        baseline = pywt.waverec(approx, wavelet)
        baseline = baseline[:n]
        corrected = w - baseline
        return corrected, baseline

    else:
        raise ValueError(f"Unknown baseline removal method: '{method}'.")


def extract_and_correct(voltage, peak_idx, fs,
                        pre_ms=3.0, post_ms=20.0,
                        baseline_method='poly',
                        poly_order=2):
    """
    Extracts a spike around a known peak index, then removes the baseline.

    Parameters
    ----------
    voltage : np.ndarray
        Full voltage trace.
    peak_idx : int
        Index of the peak (already detected).
    fs : float
        Sampling frequency in Hz.
    pre_ms, post_ms : float
        Time window before and after the peak, in milliseconds.
    baseline_method : str
        One of ['mean', 'detrend', 'poly', 'highpass', 'wavelet'].
    poly_order : int
        Order for polynomial detrending (only used if method='poly').

    Returns
    -------
    dict with keys:
        'time' : np.ndarray (windowed time vector)
        'original' : np.ndarray (raw signal)
        'corrected' : np.ndarray (baseline corrected signal)
        'baseline' : np.ndarray (estimated baseline)
        'start_idx', 'end_idx' : int (absolute indices in the trace)
    """

    n = len(voltage)
    pre_samp = int(pre_ms * fs / 1000)
    post_samp = int(post_ms * fs / 1000)

    start_idx = max(0, peak_idx - pre_samp)
    end_idx = min(n, peak_idx + post_samp)
    window = voltage[start_idx:end_idx]
    t = np.arange(len(window)) / fs * 1000  # time in ms

    corrected, baseline = remove_baseline(window, fs,
                                          method=baseline_method,
                                          poly_order=poly_order)

    return {
        "time": t,
        "original": window,
        "corrected": corrected,
        "baseline": baseline,
        "start_idx": start_idx,
        "end_idx": end_idx,
    }


def plot_spike_correction(result, fs, title=None, show=True, savepath=None, peak_rel_idx=None,
                          fit_func=None, fit_params=None):
    """
    Plot comparison between raw, baseline, corrected spike signals, and optional fit.

    Parameters
    ----------
    result : dict
        Output from `extract_and_correct()` containing:
        ['time', 'original', 'baseline', 'corrected', 'start_idx', 'end_idx']
    fs : float
        Sampling frequency (Hz)
    title : str or None
        Title for the plot
    show : bool
        If True, calls plt.show()
    savepath : str or None
        If provided, saves the figure to this path
    peak_rel_idx : int or None
        Index of the peak within the extracted window (optional)
    fit_func : callable or None
        Function used to generate the fit (e.g., spike_model_DoG)
    fit_params : dict or list or None
        Parameters of the fit function
    """

    # Extract fields safely
    t = result.get("time")
    window = result.get("original")
    baseline = result.get("baseline")
    corrected = result.get("corrected")

    if t is None or window is None:
        raise ValueError("Missing required keys in `result`: expected ['time', 'original', ...]")

    if peak_rel_idx is not None:
        peak_t = t[peak_rel_idx]
    else:
        peak_t = t[len(t)//2]

    # --- Create figure ---
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                             gridspec_kw={'hspace': 0.25})

    # --- (1) Raw + baseline ---
    ax = axes[0]
    ax.plot(t, window, color='black', lw=1.2, label='Original')
    ax.plot(t, baseline, color='tab:orange', lw=1.0, label='Baseline')
    ax.axvline(peak_t, color='r', ls='--', lw=1, label='Peak')
    ax.set_ylabel("Voltage (V)")
    ax.set_title(title or "Spike extraction and baseline correction", fontsize=12)
    ax.legend(loc='best', frameon=False)
    ax.grid(alpha=0.3)

    # --- (2) Corrected signal ---
    ax = axes[1]
    ax.plot(t, corrected, color='tab:blue', lw=1.2, label='Corrected (baseline removed)')
    ax.axvline(peak_t, color='r', ls='--', lw=1)

    # --- Plot fit if provided ---
    if fit_func is not None and fit_params is not None:
        try:
            # If fit_params is a dict, expand as kwargs
            if isinstance(fit_params, dict):
                fit_curve = fit_func(t, **fit_params)
            else:
                fit_curve = fit_func(t, *fit_params)

            ax.plot(t, fit_curve, color='tab:red', lw=2, label='Fit')
            # Display parameters on the plot
            param_text = "\n".join([f"{k}={v:.3f}" for k, v in (fit_params.items() if isinstance(fit_params, dict) else enumerate(fit_params))])
            ax.text(0.98, 0.95, param_text, transform=ax.transAxes,
                    ha='right', va='top', fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        except Exception as e:
            print(f"⚠️ Could not plot fit: {e}")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (V)")
    ax.legend(loc='best', frameon=False)
    ax.grid(alpha=0.3)

    # --- Save / Show ---
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"✅ Saved spike comparison figure to: {savepath}")

    if show:
        plt.show()
    else:
        plt.close(fig)



# --- Example usage ---
asc_p = "../ANALYSE Anna/GH4C1/2024/03072024/cell7 PA spont rxtx.ASC"
asc_p = "../ANALYSE Anna/GH4C1/2024/03072024/"

asc_files = list_asc_files(asc_p)
asc_files = [asc_files[1]]

lpf = 200
hpf = None
order_lpf = 5
order_hpf = 2

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

for asc in asc_files:

    dataset = Dataset.from_asc(
        asc,
        recording_type=RecordingType.INTRACELLULAR,
    )

    sweeps = dataset.list_sweeps()
    sweeps = [dataset.list_sweeps()[0]]
    for sweep_name in sweeps:
        sweep = dataset[sweep_name]
        sweep.filter_v(lpf=lpf, hpf=hpf, order_lpf=order_lpf, order_hpf=order_hpf)
        csv_file = get_AP_summary(asc, sweep)
        df = pd.read_csv(csv_file, index_col=False)

        print("CSV found:", csv_file)

        #fig, ax = plt.subplots(figsize=(8, 4))
        #sweep.plot(ax)

        #ax.plot(df["time_s"], df["voltage_V"], "ro", markersize=5, label="Detected APs")

        #res = extract_and_correct(sweep.voltage, idx, sweep.fs, pre_ms=3.0, post_ms=20.0,
        #                  baseline_method='highpass', highpass_cutoff=2.0)

        for idx in df['sample_index']:
            #start_idx, end_idx, spike = extract_adaptive_spike(sweep.voltage, idx, baseline=sweep.baseline_v, threshold_frac=0.1)
            res = extract_and_correct(sweep.voltage-sweep.baseline_v, idx, sweep.fs,
                                    pre_ms=500.0, post_ms=750.0,
                                    baseline_method='poly',poly_order=1)
            

            t = res['time']             # in ms
            v = res['corrected']        # baseline-corrected spike

            param, fit_curve = fit_spike_DoG(t, v)


            ## --- 4️⃣ Plot ---
            #plt.figure(figsize=(8, 5))
            #plt.plot(t, v, label="Extracted spike", color="black", lw=1.3)
            #plt.plot(t, fit_curve, label="Double-exp fit", color="tab:red", lw=2)
            #plt.xlabel("Time (ms)")
            #plt.ylabel("Voltage (V)")
            #plt.title(f"Spike fit — τr={params['tau_r']:.2f} ms, τd={params['tau_d']:.2f} ms")
            #plt.legend(frameon=False)
            #plt.grid(alpha=0.3)
            #plt.show()
            
            plot_spike_correction(res, fs=sweep.fs, title="Spike with DoG fit",
                      peak_rel_idx=res['time'].argmax(),
                      fit_func=spike_model_DoG, fit_params=param)
            
        
            #ax.plot(sweep.time[start:end], sweep.voltage[start:end], "r", label="Detected APs")

        plt.show()





