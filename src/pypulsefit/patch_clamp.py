from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional
import pandas as pd
import matplotlib.pyplot as plt
from pypulsefit.loader import to_dfs
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, peak_widths
from scipy.signal import detrend as scipy_detrend
import pywt


class RecordingType(str, Enum):
    """Enumeration for types of electrophysiological recordings."""
    INTRACELLULAR = "intracellular"
    EXTRACELLULAR = "extracellular"


@dataclass
class Sweep:
    """
    Represents a single patch-clamp sweep.

    Attributes
    ----------
    name : str
        Name of the sweep (e.g., 'Sweep 1_1_1').
    data : pd.DataFrame
        DataFrame containing the sweep's numeric data (time, trace1, trace2, etc.).
    recording_type : RecordingType
        Type of recording (intracellular or extracellular).
    metadata : Dict[str, str], optional
        Optional dictionary to store additional information about the sweep.
    """
    name: str
    data: pd.DataFrame
    recording_type: RecordingType
    metadata: Dict[str, str] = field(default_factory=dict)

    _current: Optional[pd.Series] = field(init=False, repr=False)
    _voltage: Optional[pd.Series] = field(init=False, repr=False)

    def __post_init__(self):
        """Standardize column names to lowercase with underscores."""
        self.data.columns = [c.strip().lower().replace(" ", "_") for c in self.data.columns]

        # Initialize private attributes
        self._current = self.data.filter(like="trace1").squeeze("columns")
        self._voltage = self.data.filter(like="trace2").squeeze("columns")

    @property
    def time(self) -> pd.Series:
        """Return the time series."""
        return self.data.get("time[s]", self.data.get("time_s", self.data.iloc[:, 0]))

    @property
    def current(self) -> Optional[pd.Series]:
        """Return the current trace (trace1). Read-only."""
        return self._current

    def _set_current(self, series: pd.Series):
        """Private setter for current trace."""
        self._current = series

    @property
    def voltage(self) -> Optional[pd.Series]:
        """Return the voltage trace (trace2). Read-only."""
        return self._voltage

    def _set_voltage(self, series: pd.Series):
        """Private setter for voltage trace."""
        self._voltage = series
    
    @property
    def fs(self) -> float:
        """Return sampling frequency."""
        return(1/(self.time.iloc[1] - self.time.iloc[0]))
    
    @property
    def baseline_v(self) ->float:
        """Return voltage baseline."""
        return(np.median(self.voltage.values))

    @property
    def baseline_i(self) ->float:
        """Return voltage baseline."""
        return(np.median(self.current.values))


    def plot(self, ax: plt.Axes) -> None:
        """
        Plot the voltage trace over time.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plotted sweep.
        """
        ax.plot(self.time, self.voltage, label=self.name)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Voltage [V]" if self.recording_type == RecordingType.INTRACELLULAR else "Signal [V]")

    def filter_v(self, lpf: float|None, hpf: float|None,  order_lpf: int = 3, order_hpf: int = 3):
        """
        """
        if lpf is not None:
            b, a = butter(order_lpf, lpf / (0.5 * self.fs), btype='low', analog=False)
            signal = filtfilt(b, a, self.voltage)
            self._set_voltage(pd.Series(signal, index=self.data.index))


        if hpf is not None:
            b, a = butter(order_hpf, hpf / (0.5 * self.fs), btype='high', analog=False)
            signal = filtfilt(b, a, self.voltage) + self.baseline_v
            self._set_voltage(pd.Series(signal, index=self.data.index))

    def get_peaks(
        self,
        threshold: float | None = None,
        min_interval_ms: float = 50.0,
        min_duration_ms: float | None = None,
        max_duration_ms: float | None = None,
        min_abs_peak: float = 0.0001,
    ):
        """
        Detect peaks in the voltage trace based on amplitude and optional duration constraints.

        Parameters
        ----------
        threshold : float | None
            Minimum amplitude to count as a peak. If None, automatically computed.
        min_interval_ms : float
            Minimum interval between peaks (ms).
        min_duration_ms : float | None
            Minimum duration of a peak (ms).
        max_duration_ms : float | None
            Maximum duration of a peak (ms).
        min_abs_peak : float
            Absolute minimum peak value (V).

        Returns
        -------
        peaks : np.ndarray
            Indices of detected peaks.
        threshold : float
            Threshold used for detection.
        """
        signal = self.voltage.values - self.baseline_v
        noise_floor = np.median(np.abs(signal - np.median(signal)))
        max_data = np.max(signal)
        if threshold is None:
            threshold = max(2 * noise_floor, 0.4 * max_data)

        min_distance = int(min_interval_ms * 1e-3 * self.fs)

        # Convert duration to samples
        width_kwargs = {}
        if min_duration_ms is not None or max_duration_ms is not None:
            widths_samples = {}
            if min_duration_ms is not None:
                widths_samples["min"] = int(min_duration_ms * 1e-3 * self.fs)
            if max_duration_ms is not None:
                widths_samples["max"] = int(max_duration_ms * 1e-3 * self.fs)
            width_kwargs["width"] = widths_samples.get("min", None)  # min width for find_peaks

        # Detect peaks
        peaks, _ = find_peaks(signal, height=threshold, distance=min_distance)

        # Filter by duration using peak_widths
        if min_duration_ms is not None or max_duration_ms is not None:
            widths = peak_widths(signal, peaks, rel_height=0.5)[0]  # widths in samples
            mask = np.ones_like(peaks, dtype=bool)
            if min_duration_ms is not None:
                mask &= widths >= widths_samples["min"]
            if max_duration_ms is not None:
                mask &= widths <= widths_samples["max"]
            peaks = peaks[mask]

        # --- ➤ filter peaks by absolute amplitude ---
        abs_mask = np.abs(signal[peaks]) >= min_abs_peak
        peaks = peaks[abs_mask]

        return peaks, threshold+self.baseline_v
    

    def detrend_signal(self, method: str = "linear", poly_order: int = 2, window_ms: float | None = None) -> np.ndarray:
        """
        Remove baseline drift from the voltage trace.

        Parameters
        ----------
        method : str
            Method to remove trend:
            - "linear": subtract linear trend (scipy.signal.detrend)
            - "poly": fit and subtract a polynomial of order `poly_order`
            - "moving_average": subtract moving average with window `window_ms`
        poly_order : int
            Order of polynomial for polynomial detrending (if method='poly')
        window_ms : float | None
            Window in ms for moving average detrending (if method='moving_average')

        Returns
        -------
        detrended : np.ndarray
            Detrended voltage trace.
        """
        signal = self.voltage.values.copy()

        if method == "linear":
            detrended = scipy_detrend(signal, type="linear")
        elif method == "poly":
            t = np.arange(len(signal))
            coeffs = np.polyfit(t, signal, poly_order)
            trend = np.polyval(coeffs, t)
            detrended = signal - trend
        elif method == "moving_average":
            if window_ms is None:
                raise ValueError("window_ms must be provided for moving_average detrending")
            window_samples = int(window_ms * 1e-3 * self.fs)
            if window_samples < 1:
                raise ValueError("window_ms too small for sampling rate")
            # Compute moving average
            trend = np.convolve(signal, np.ones(window_samples)/window_samples, mode='same')
            detrended = signal - trend
        else:
            raise ValueError(f"Unknown detrend method: {method}")

        return detrended+self.baseline_v

    

    def get_peaks_cwt(self,
        min_interval_ms: float = 50.0,
        min_duration_ms: float | None = None,
        max_duration_ms: float | None = None,
        threshold_ratio: float = 0.5,
    ):
        """
        Detect peaks using Continuous Wavelet Transform (CWT) with duration filtering.

        Parameters
        ----------
        min_interval_ms : float
            Minimum interval between peaks (ms).
        min_duration_ms : float | None
            Minimum duration of a peak (ms).
        max_duration_ms : float | None
            Maximum duration of a peak (ms).
        threshold_ratio : float
            Fraction of maximum CWT response for detection.

        Returns
        -------
        peaks : np.ndarray
            Indices of detected peaks above baseline + 10%.
        cwt_response : np.ndarray
            Maximum CWT response across scales.
        """
        signal = self.voltage.values - self.baseline_v
        #signal = self.detrend_signal(method="moving_average", window_ms=250)- self.baseline_v
        #signal = np.maximum(signal, 0)

        scales = np.arange(20, 50)
        cwtmatr, _ = pywt.cwt(signal, scales, 'mexh')
        cwt_response = np.max(np.abs(cwtmatr), axis=0)
        #cwt_response[cwt_response>0.01] = 0

        threshold = np.max(cwt_response) * threshold_ratio
        min_distance = int(self.fs * min_interval_ms * 1e-3)
        peaks, _ = find_peaks(cwt_response, height=threshold, distance=min_distance)

        # Filter by amplitude (baseline + 10%)
        amplitude_threshold = self.baseline_v + 0.1 * np.max(signal)
        #peaks = np.array([p for p in peaks if self.voltage.values[p] > amplitude_threshold])

        # Filter by duration using peak_widths
        if min_duration_ms is not None or max_duration_ms is not None:
            widths = peak_widths(cwt_response, peaks, rel_height=0.5)[0]  # widths in samples
            mask = np.ones_like(peaks, dtype=bool)
            if min_duration_ms is not None:
                mask &= widths >= int(min_duration_ms * 1e-3 * self.fs)
            if max_duration_ms is not None:
                mask &= widths <= int(max_duration_ms * 1e-3 * self.fs)
            peaks = peaks[mask]

        return peaks, cwt_response, threshold


@dataclass
class Dataset:
    """
    Represents a patch-clamp recording file containing one or multiple sweeps.

    Attributes
    ----------
    filepath : str
        Path to the .asc file.
    recording_type : RecordingType
        Type of recording (intracellular or extracellular).
    sweeps : Dict[str, Sweep]
        Dictionary mapping sweep names to Sweep objects.
    """
    filepath: str
    recording_type: RecordingType
    sweeps: Dict[str, Sweep] = field(default_factory=dict)

    @classmethod
    def from_asc(cls, filepath: str, recording_type: RecordingType) -> Dataset:
        """
        Load a dataset from a .asc file.

        Parameters
        ----------
        filepath : str
            Path to the .asc file.
        recording_type : RecordingType
            Type of recording (intracellular or extracellular).

        Returns
        -------
        Dataset
            A Dataset object containing all sweeps from the file.
        """
        datasets = to_dfs(filepath)
        sweeps = {
            name: Sweep(name=name, data=df, recording_type=recording_type)
            for name, df in datasets.items()
        }
        return cls(filepath=filepath, recording_type=recording_type, sweeps=sweeps)
    
    def list_sweeps(self) -> list[str]:
        """
        Return a list of all available sweep names in the dataset.

        Returns
        -------
        list[str]
            Names of all sweeps contained in this dataset.
        """
        return list(self.sweeps.keys())

    def __getitem__(self, key: str) -> Sweep:
        """
        Access a Sweep by name.

        Parameters
        ----------
        key : str
            Name of the sweep.

        Returns
        -------
        Sweep
            The corresponding Sweep object.
        """
        return self.sweeps[key]

    def get_sweep_by_index(self, index: int) -> Sweep:
        """
        Retrieve a sweep by its index in the dataset.

        Parameters
        ----------
        index : int
            Zero-based index of the sweep.

        Returns
        -------
        Sweep
            The sweep at the specified index.

        Raises
        ------
        IndexError
            If the index is out of range.
        """
        sweep_names = self.list_sweeps()
        if index < 0 or index >= len(sweep_names):
            raise IndexError(f"Sweep index {index} out of range (0-{len(sweep_names)-1})")
        return self.sweeps[sweep_names[index]]