import numpy as np
from pypulsefit.patch_clamp import Dataset, RecordingType
import matplotlib.pyplot as plt


data = Dataset.from_asc("../ANALYSE Anna/GH4C1/2024/16072024/cell8 petri PDL PA induits.asc",recording_type=RecordingType.INTRACELLULAR )
sweep = data.get_sweep_by_index(0)


fig, ax = plt.subplots()

sweep.plot(ax)



lpf = 300
hpf = None
order_lpf = 5
order_hpf = 2 
# High-pass filter example (cutoff 1 Hz)
sweep.filter_v(lpf=lpf, hpf=hpf,  order_lpf=order_lpf, order_hpf=order_hpf)
ax.plot(sweep.time, sweep.voltage, label='Filtered', alpha=1)


ax.legend()
plt.show()
