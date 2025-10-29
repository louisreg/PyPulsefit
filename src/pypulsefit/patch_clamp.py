from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional
import pandas as pd
import matplotlib.pyplot as plt
from pypulsefit.loader import to_dfs
import numpy as np
from scipy.signal import butter, filtfilt

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