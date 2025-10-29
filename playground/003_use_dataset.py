import pypulsefit.patch_clamp as pc
import matplotlib.pyplot as plt

data = pc.Dataset.from_asc("../ANALYSE Anna/GH4C1/2024/18042024/cell23 PA induits.asc",recording_type=pc.RecordingType.INTRACELLULAR )
sweep = data.get_sweep_by_index(3)

fig, ax = plt.subplots()

sweep.plot(ax)

plt.show()

