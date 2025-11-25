import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pypulsefit.patch_clamp import RecordingType, Sweep
from scipy.signal import savgol_filter
from copy import deepcopy
from scipy.interpolate import interp1d

class AP:
    """
    Represents a single Action Potential (AP) extracted from a sweep.

    Attributes
    ----------
    sweep : Sweep
        The sweep object containing the full trace.
    idx : int
        Index (sample) of the detected AP peak.
    pre_AP_ms : float
        Time before the AP peak to include (in milliseconds).
    post_AP_ms : float
        Time after the AP peak to include (in milliseconds).
    df : pd.DataFrame
        DataFrame containing 'time' and 'voltage' for the extracted AP window.
    """

    def __init__(self, sweep: Sweep, idx: int, pre_AP_ms: float = 5.0, post_AP_ms: float = 10.0) -> None:
        self.idx = int(idx)
        self.pre_AP_ms = float(pre_AP_ms)
        self.post_AP_ms = float(post_AP_ms)
        self.recording_type = sweep.recording_type

        # --- Extract time window around the initial AP index ---
        self._fs = sweep.fs
        n_pre = int(self.fs * pre_AP_ms / 1000.0)
        n_post = int(self.fs * post_AP_ms / 1000.0)
        start = max(0, self.idx - n_pre)
        end = min(len(sweep.time) - 1, self.idx + n_post)

        voltage_window = sweep.voltage.iloc[start:end].reset_index(drop=True)
        time_window = sweep.time.iloc[start:end].reset_index(drop=True)

        # --- Define search window for local peak around initial idx ---
        window_ms = 10  # search ±5 ms
        window_pts = int(self.fs * window_ms / 1000.0)

        # Ensure indices stay within voltage_window
        peak_start = max(0, n_pre - window_pts)
        peak_end = min(len(voltage_window)-1, n_pre + window_pts)

        # --- Recompute the peak index within the limited window ---
        local_peak_idx = voltage_window.iloc[peak_start:peak_end+1].idxmax()
        self.idx = start + local_peak_idx  # update absolute index
        self._df = pd.DataFrame({
            "time": time_window,
            "voltage": voltage_window*1000  # convert to mV if needed
        })

        # --- Center time axis around the true AP peak (t = 0) ---
        self._df["time"] = self._df["time"] - sweep.time.iloc[self.idx]

        # Optional: automatic name
        self.name = f"AP_{self.idx}"

    # ----------------------------
    #        PROPERTIES
    # ----------------------------
    @property
    def time(self) -> pd.Series:
        """Time vector (s), centered on AP peak."""
        return self._df["time"]

    @property
    def voltage(self) -> pd.Series:
        """Voltage trace (V) for the extracted AP."""
        return self._df["voltage"]

    @property
    def fs(self) -> float:
        """Sampling frequency (Hz)."""
        return self._fs

    @property
    def duration_ms(self) -> float:
        """Return AP window duration (ms)."""
        return (self.time.iloc[-1] - self.time.iloc[0]) * 1000
    

    @property
    def max(self) -> float:
        """Return AP max value""" 
        return(np.max(self.voltage))
    
    @property
    def min(self) -> float:
        """Return AP min value""" 
        voltage = self.voltage.values
        max_idx = np.argmax(voltage)          # index of the peak
        min_idx = np.argmin(voltage[max_idx:]) + max_idx  # min after peak
        return voltage[min_idx]
    
    @property
    def peak2peak(self) -> float:
        """Return AP peak-to-peak value"""
        return(self.max - self.min)
    

    @property
    def t_max(self) -> float:
        """Return time of AP peak relative to AP alignment (0 ms at peak)"""
        return 0.0

    @property
    def t_min(self) -> float:
        """Return time of the minimum (trough) after the AP peak"""
        voltage = self.voltage.values
        time = self.time.values
        max_idx = np.argmax(voltage)               # index of the peak
        min_idx = np.argmin(voltage[max_idx:]) + max_idx  # index of minimum after peak
        # Check if the minimum is at the last point
        if min_idx >= len(voltage) - 1:
            return self.t_adp_ahp
        return time[min_idx]


    @property
    def area(self) -> float:
        """Return area under the AP curve relative to baseline (trapezoidal integration)"""
        baseline = self.voltage.iloc[0]  # or use separate baseline
        return np.trapz(self.voltage - baseline, self.time)

    

    def remove_baseline(self, method: str = "mean", window_ms: float = 20):
        """
        Remove the slow baseline (DC component) from the action potential trace.

        Parameters
        ----------
        method : str
            Method used to estimate the baseline:
            - "mean": subtracts the mean value of the trace
            - "poly": subtracts a polynomial (linear trend)
            - "savgol": subtracts a smoothed version using a Savitzky–Golay filter
        window_ms : float
            Window size in milliseconds for smoothing (used only if method='savgol').

        Returns
        -------
        pd.Series
            The baseline-corrected voltage trace.
        """
        fs = self.fs
        voltage = self._df["voltage"].values
        time = self._df["time"].values #in ms

        if method == "mean":
            # Subtract global mean
            baseline = np.mean(voltage)
            baseline = np.full_like(voltage, baseline)
        elif method == "linear":
            # Fit and remove linear trend
            coeffs = np.polyfit(time, voltage, 1)
            baseline = np.polyval(coeffs, time)
        elif method == "savgol":
            # Smooth with Savitzky–Golay filter to approximate the baseline
            win = int((window_ms / 1000) * fs)
            if win % 2 == 0:
                win += 1  # window length must be odd
            baseline = savgol_filter(voltage, window_length=win, polyorder=2)

        else:
            raise ValueError(f"Unknown baseline method: {method}")
        

        # Subtract baseline
        corrected = voltage - baseline

        n_points = 10
        if len(corrected) < 2*n_points:
            raise ValueError("Not enough points to compute linear trend.")

        # Compute mean of first n_points and last n_points
        mean_start = np.mean(corrected[:n_points])
        mean_end = np.mean(corrected[-n_points:])

        # Fit linear baseline: y = m*x + b
        m = (mean_end - mean_start) / (time[-1] - time[0])
        b = mean_start - m*time[0]
        baseline2 = m*time + b

        corrected = corrected - baseline2 

        self._df["voltage"] = corrected
        return baseline + baseline2

    @property
    def t_adp_ahp(self) -> float:
        """
        Detect the time of the zero-crossing (baseline crossing) between t_max and t_min.
        This marks the transition from ADP to AHP.

        Returns
        -------
        float or None
            Time of the transition in seconds, or None if not found.
        """

        t = self.time.values
        v = self.voltage.values

        # Define indices for the interval
        idx_start = np.argmin(np.abs(t - self.t_max))
        idx_end = len(self.voltage.values) - 1

        # Slice the signal
        segment = v[idx_start:idx_end+1]
        segment_t = t[idx_start:idx_end+1]

        # Detect zero-crossing (sign change relative to baseline = 0)
        zero_crossings = np.where(np.diff(np.sign(segment)))[0]

        if zero_crossings.size == 0:
            return None

        # Take the first zero crossing as the ADP->AHP transition
        transition_idx = zero_crossings[0]
        transition_time = segment_t[transition_idx]

        return transition_time

    def t_start(self, fraction: float = 0.1) -> float:
        """
        Detect the start of the AP spike as the point closest to the peak
        where the voltage reaches a fraction of the peak value.

        Parameters
        ----------
        fraction : float
            Fraction of the peak voltage to define the start (default 0.1 = 10%).

        Returns
        -------
        t_start : float
            Time of the estimated spike start (seconds).
        """
        voltage = self.voltage.values
        time = self.time.values
        peak_idx = np.argmax(voltage)
        peak_voltage = voltage[peak_idx]
        threshold = fraction * peak_voltage

        # scan backward from the peak to find first point below threshold
        for idx in range(peak_idx, -1, -1):
            if voltage[idx] <= threshold:
                return time[idx]

        # fallback if not found
        return time[0]

    def t_end(self, fraction: float = 0.1) -> float:
        """
        Detect the end of the AP spike as the point closest to the trough (t_min)
        where the voltage reaches a fraction of the minimum value.

        Parameters
        ----------
        fraction : float
            Fraction of the peak-to-trough voltage to define the end (default 0.1 = 10%).

        Returns
        -------
        t_end : float
            Time of the estimated spike end (seconds).
        """
        voltage = self.voltage.values
        time = self.time.values

        # Find the trough (min voltage) after the peak
        peak_idx = np.argmax(voltage)
        min_idx = np.argmin(voltage[peak_idx:]) + peak_idx
        min_voltage = voltage[min_idx]
        threshold = fraction * min_voltage  # fraction of negative peak

        # scan forward from min_idx to find first point above threshold
        for idx in range(min_idx, len(voltage)):
            if voltage[idx] >= threshold:
                return time[idx]

        # fallback if not found
        return self.t_adp_ahp

    def t_AP(self, fraction: float = 0.1) -> float:
        """Total spike duration in seconds"""
        return self.t_end(fraction) - self.t_start(fraction)
    

    @property
    def t_half_pre(self) -> float:
        """Return the time at 50% of peak before t_max using linear interpolation."""
        if self.t_start is None or self.t_max is None:
            return None
        
        v_half = self.v_half  # 0.5 * self.max
        
        # Slice the segment between t_start and t_max
        mask = (self.time >= self.t_start()) & (self.time <= self.t_max)
        t_segment = self.time.values[mask]
        v_segment = self.voltage.values[mask]

        if len(t_segment) < 2:
            return None

        # Interpolate time for v_half
        f = interp1d(v_segment, t_segment, kind='linear', bounds_error=False, fill_value='extrapolate')
        return float(f(v_half))

    @property
    def t_half_post(self) -> float:
        """Return the time at 50% of peak after t_max using linear interpolation."""
        if self.t_max is None or self.t_adp_ahp is None:
            return None
        
        v_half = self.v_half  # 0.5 * self.max
        
        # Slice the segment between t_max and t_adp_ahp
        mask = (self.time >= self.t_max) & (self.time <= self.t_adp_ahp)
        t_segment = self.time.values[mask]
        v_segment = self.voltage.values[mask]

        if len(t_segment) < 2:
            return None

        # Interpolate time for v_half
        f = interp1d(v_segment, t_segment, kind='linear', bounds_error=False, fill_value='extrapolate')
        return float(f(v_half))
    
    @property
    def v_half(self) -> float:
        """Voltage at t_half_pre using linear interpolation"""
        return float(0.5*self.max)

    @property
    def v_half_post(self) -> float:
        """Voltage at t_half_post using linear interpolation"""
        if self.t_half_post is None:
            return None
        f = interp1d(self.time.values, self.voltage.values, kind="linear", bounds_error=False, fill_value="extrapolate")
        return float(f(self.t_half_post))

    @property
    def t_half(self) -> float:
        """Return the duration between t_half_pre and t_half_post."""
        if self.t_half_pre is None or self.t_half_post is None:
            return None
        return self.t_half_post - self.t_half_pre
    

    def get_metrics(self, fraction: float = 0.1) -> dict:
        """Compute all AP metrics in one place"""
        return {
            "max": self.max,
            "min": self.min,
            "peak2peak": self.peak2peak,
            "area": self.area,
            "t_max": self.t_max,
            "t_min": self.t_min,
            "t_adp_ahp": self.t_adp_ahp,
            "t_start": self.t_start(fraction),
            "t_end": self.t_end(fraction),
            "t_AP": self.t_AP(fraction),
            "t_half_pre": self.t_half_pre,
            "t_half_post": self.t_half_post,
            "t_half": self.t_half,
            "v_half": self.v_half,
        }

    # ----------------------------
    #         PLOT
    # ----------------------------
    def plot(self, ax: plt.Axes, show_peak: bool = True) -> None:
        """
        Plot the AP trace.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on (creates one if None)
        show_peak : bool
            Whether to mark the AP peak (t=0)

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plotted AP
        """
        ax.plot(self.time, self.voltage, lw=1.5, color='k', label=self.name)
        if show_peak:
            ax.axvline(0, color='r', ls='--', lw=1, label='AP peak')

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Voltage (mV)" if self.recording_type == RecordingType.INTRACELLULAR else "Signal (V)")
        ax.grid(alpha=0.3)


    # ----------------------------
    #         UTILITY
    # ----------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the AP data as a DataFrame."""
        return self._df.copy()

    def __repr__(self) -> str:
        return (f"<AP name={self.name} idx={self.idx} "
                f"window=[-{self.pre_AP_ms}ms, +{self.post_AP_ms}ms] "
                f"fs={self.fs:.1f}Hz>")


def average_APs(ap_list):
    """
    Compute the average and standard deviation AP from a list of AP objects.

    Parameters
    ----------
    ap_list : list of AP
        List of AP instances, must be aligned and of the same length.

    Returns
    -------
    avg_ap : AP
        AP object representing the mean AP.
    std_ap : AP
        AP object representing the standard deviation AP.
    """
    if not ap_list:
        raise ValueError("ap_list is empty")
    
    # Determine global min and max time across all APs
    t_min = min(ap.time.values[0] for ap in ap_list)
    t_max = max(ap.time.values[-1] for ap in ap_list)

    # Number of points: take the max number of points across APs
    n_max = max(len(ap.voltage) for ap in ap_list)

    # Create reference time array
    ref_time = np.linspace(t_min, t_max, n_max)

    # Interpolate all AP voltages to the common time base
    voltages = []
    for ap in ap_list:
        if len(ap.time) < 50:
            continue  # skip very short AP
        t = ap.time.values
        v = ap.voltage.values
        
        # Keep only unique times
        t_unique, idx_unique = np.unique(t, return_index=True)
        v_unique = v[idx_unique]
        
        v_interp = np.interp(ref_time, t_unique, v_unique)
        voltages.append(v_interp)
    voltages = np.array(voltages)  # shape: (n_APs, n_max)

    # Compute mean and standard deviation
    mean_voltage = np.mean(voltages, axis=0)
    std_voltage = np.std(voltages, axis=0)

    # Create average AP object
    avg_ap = deepcopy(ap_list[0])
    avg_ap._df = pd.DataFrame({"time": ref_time, "voltage": mean_voltage})
    avg_ap.idx = np.argmax(mean_voltage)
    avg_ap.name = "Average_AP"

    # Create std AP object
    std_ap = deepcopy(ap_list[0])
    std_ap._df = pd.DataFrame({"time": ref_time, "voltage": std_voltage})
    std_ap.idx = np.argmax(std_voltage)
    std_ap.name = "STD_AP"

    return avg_ap, std_ap