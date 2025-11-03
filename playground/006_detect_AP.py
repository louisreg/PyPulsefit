from pypulsefit.patch_clamp import Dataset, RecordingType
import matplotlib.pyplot as plt

# --- Load data ---
data = Dataset.from_asc(
    "../ANALYSE Anna/GH4C1/2024/06062024/cell11 PA spont RxTx.asc",
    recording_type=RecordingType.INTRACELLULAR,
)
sweep = data.get_sweep_by_index(0)

# --- Filter signal ---
lpf = 200
hpf = None
order_lpf = 5
order_hpf = 2
sweep.filter_v(lpf=lpf, hpf=hpf, order_lpf=order_lpf, order_hpf=order_hpf)
peaks_fp,threshold = sweep.get_peaks()

# --- Plot results ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(sweep.time, sweep.voltage, label="Filtered signal", lw=1.2)
ax.plot(sweep.time.iloc[peaks_fp], sweep.voltage[peaks_fp], "ro", label="find_peaks")
ax.axhline(y=threshold, color="gray", ls="--", label=f"Threshold ({threshold:.3f} V)")
ax.legend()
ax.set_xlabel("Time [s]")
ax.set_ylabel("Voltage [V]")
plt.show()

