from pypulsefit.utils import list_asc_files, get_AP_summary
from pypulsefit.patch_clamp import Dataset, RecordingType
from pypulsefit.action_potential import AP
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Example usage ---
asc_p = "../ANALYSE Anna/GH4C1/2024/03072024/cell7 PA spont rxtx.ASC"
asc_p = "../ANALYSE Anna/GH4C1/2024/06062024/"

asc_files = list_asc_files(asc_p)
asc_files = [asc_files[8]]

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

fig, axs = plt.subplots(2,1)
for asc in asc_files:

    dataset = Dataset.from_asc(
        asc,
        recording_type=RecordingType.INTRACELLULAR,
    )

    sweeps = dataset.list_sweeps()
    for sweep_name in sweeps:
        print(sweep_name)
        sweep = dataset[sweep_name]
        sweep.filter_v(lpf=lpf, hpf=hpf, order_lpf=order_lpf, order_hpf=order_hpf)
        csv_file = get_AP_summary(asc, sweep)
        df = pd.read_csv(csv_file, index_col=False)

        df = df[df["type"] != "burst"]

        for AP_idx in df["sample_index"]:
            pre_AP_ms = 50
            post_AP_ms = 100

            ap = AP(sweep, AP_idx, pre_AP_ms, post_AP_ms)
            ap.plot(ax=axs[0])

            baseline = ap.remove_baseline(method = "linear")
            axs[0].plot(ap.time, baseline, '--', c = "grey", alpha = 0.4)
            ap.plot(ax=axs[1])

axs[0].set_title("Raw")
axs[1].set_title("Baseline Removed")

        #print(idx)

fig.tight_layout()
plt.show()






