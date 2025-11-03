from pypulsefit.utils import list_asc_files
import matplotlib.pyplot as plt
from pypulsefit.patch_clamp import Dataset, RecordingType
import numpy as np
"""
03072024
04072024
06062024
16072024
07062024
14032024
15032024
16072024
18042024
19042024
29032024
"""
# --- CONFIGURATION ---
path = "../ANALYSE Anna/GH4C1/2024/16072024/"
o_path = ""

asc_files = list_asc_files(path)
lpf = 200
hpf = None
order_lpf = 5
order_hpf = 2

min_interval_ms = 100
min_duration_ms = None
max_duration_ms = None

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

# --- MAIN LOOP ---
for asc in asc_files:

    dataset = Dataset.from_asc(
        asc,
        recording_type=RecordingType.INTRACELLULAR,
    )

    for sweep_name in dataset.list_sweeps():
        sweep = dataset[sweep_name]
        fig, ax = plt.subplots(1, figsize=(8, 4))

        # --- Plot non-filtered signal (gray, transparent) ---
        ax.plot(
            sweep.time, sweep.voltage,
            color="gray", alpha=0.5, lw=1.0, label="Raw signal"
        )

        # --- Filter signal ---
        sweep.filter_v(lpf=lpf, hpf=hpf, order_lpf=order_lpf, order_hpf=order_hpf)

        # --- Detect APs ---
        peaks_fp, threshold = sweep.get_peaks(
            min_interval_ms=min_interval_ms,
            min_duration_ms=min_duration_ms,
            max_duration_ms=max_duration_ms
        )

        # --- Plot filtered signal (highlighted) ---
        ax.plot(
            sweep.time, sweep.voltage,
            color="black", lw=1.6, label="Filtered signal"
        )

        # --- Mark detected APs ---
        if len(peaks_fp):
            off = np.max(sweep.voltage[peaks_fp])
            if off >= 0:
                offs = [off * 1.05] * len(peaks_fp)
                off_label = offs[0] * 1.03
            else:
                offs = [off * 0.95] * len(peaks_fp)
                off_label = offs[0] * 0.97

            ax.plot(sweep.time.iloc[peaks_fp], offs, "ro", markersize=5, label="Detected APs")

            # ➤ Add labels "# n" near peaks
            for i, peak_idx in enumerate(peaks_fp):
                t = sweep.time.iloc[peak_idx]
                ax.text(
                    t, off_label,
                    f"# {i + 1}",
                    rotation=45,
                    ha='left', va='bottom',
                    fontsize=9, color='red', weight='bold'
                )

        # --- Threshold and baseline lines (lighter) ---
        ax.axhline(y=threshold, color="gray", ls="--", lw=1, label=f"Threshold ({threshold:.3f} V)")
        ax.axhline(y=sweep.baseline_v, color="k", ls=":", lw=1, label=f"Baseline ({sweep.baseline_v:.3f} V)")

        # --- Beautify plot ---
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Voltage [V]")
        ax.legend(frameon=False)
        ax.margins(x=0)
        ax.tick_params(width=1.2, length=5, direction="out")

        plt.tight_layout()
        plt.show()