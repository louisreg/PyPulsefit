import numpy as np
from pypulsefit.patch_clamp import Dataset, RecordingType
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


data = Dataset.from_asc("../ANALYSE Anna/GH4C1/2024/16072024/cell8 petri PDL PA induits.asc",recording_type=RecordingType.INTRACELLULAR )
sweep = data.get_sweep_by_index(0)


fig, ax = plt.subplots()

lpf = 200
hpf = None
order_lpf = 5
order_hpf = 2 
# High-pass filter example (cutoff 1 Hz)
sweep.filter_v(lpf=lpf, hpf=hpf,  order_lpf=order_lpf, order_hpf=order_hpf)

data = sweep.voltage.values - sweep.baseline_v
ax.plot(sweep.time, sweep.voltage, label='data', alpha=1)



height = np.max(data)*0.65
peaks, _ = find_peaks(data,height, distance=500)
ax.plot(sweep.time.iloc[peaks], sweep.voltage.iloc[peaks], "ro", label="Peaks")
ax.legend()
plt.show()
