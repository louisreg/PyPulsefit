import numpy as np
from pypulsefit.patch_clamp import Dataset, RecordingType
import matplotlib.pyplot as plt

data = Dataset.from_asc("../ANALYSE Anna/GH4C1/2024/03072024/cell8.asc",recording_type=RecordingType.INTRACELLULAR )
sweep = data.get_sweep_by_index(0)



print("Baseline (median):", sweep.baseline_v)


fig, ax = plt.subplots()

sweep.plot(ax)

ax.hlines(sweep.baseline_v, sweep.time.iloc[0], sweep.time.iloc[-1], 
            colors='red', linestyles='--', label='Baseline (median)')

ax.legend()
plt.show()
