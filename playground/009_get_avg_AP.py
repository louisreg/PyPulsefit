from pypulsefit.utils import list_asc_files, get_AP_summary
from pypulsefit.patch_clamp import Dataset, RecordingType
from pypulsefit.action_potential import AP, average_APs
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List

import matplotlib.pyplot as plt
import numpy as np

def plot_basic_metrics(ap, ax):
    """
    Plot AP trace and mark t_max, t_min, t_start, t_end, ADP->AHP, and t_half duration.

    Parameters
    ----------
    ap : AP
        Action potential object with `time` and `voltage` properties.
    ax : matplotlib.axes.Axes
        Axes to plot on.
    """
    metrics = ap.get_metrics()
    t_max = metrics['t_max']
    t_min = metrics['t_min']
    t_adp_ahp = metrics.get('t_adp_ahp', None)
    t_start = metrics.get('t_start', None)
    t_end = metrics.get('t_end', None)
    t_half_pre = metrics.get('t_half_pre', None)
    t_half_post = metrics.get('t_half_post', None)
    t_half = metrics.get('t_half', None)
    v_half = metrics.get('v_half', None)

    t = ap.time.values
    v = ap.voltage.values

    # --- Plot AP trace ---
    ax.plot(t, v, color='black', lw=1.5, label='AP trace')

    # --- Mark key points ---
    idx_max = np.argmin(np.abs(t - t_max))
    ax.plot(t_max, v[idx_max], 'ro', label='t_max (peak)')

    idx_min = np.argmin(np.abs(t - t_min))
    ax.plot(t_min, v[idx_min], 'bo', label='t_min (trough)')

    if t_adp_ahp is not None:
        idx_adp = np.argmin(np.abs(t - t_adp_ahp))
        ax.plot(t_adp_ahp, v[idx_adp], 'mo', label='ADP->AHP')

    if t_start is not None:
        idx_start = np.argmin(np.abs(t - t_start))
        ax.plot(t_start, v[idx_start], 'co', label='t_start')

    if t_end is not None:
        idx_end = np.argmin(np.abs(t - t_end))
        ax.plot(t_end, v[idx_end], 'yo', label='t_end')

    # --- Plot t_half vertical line ---
    if t_half_pre is not None and t_half_post is not None:
        v_half_line = [v_half, v_half]
        ax.plot([t_half_pre, t_half_post], v_half_line, 'g-', lw=2, label='t_half duration', alpha = 0.6)
        # Optional: mark t_half points
        ax.plot(t_half_pre, v_half, 'gs', markersize=6)
        ax.plot(t_half_post, v_half, 'gs', markersize=6)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.legend(loc='best', frameon=False)
    ax.grid(alpha=0.3)


# --- Example usage ---
asc_p = "../ANALYSE Anna/GH4C1/2024/03072024/cell7 PA spont rxtx.ASC"
asc_p = "../ANALYSE Anna/GH4C1/2024/06062024/"

asc_files = list_asc_files(asc_p)
asc_files = [asc_files[9]]

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

# --- Collect APs for averaging ---
AP_l = []
fig, ax = plt.subplots(figsize=(12,6))

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
        if "type" in df.columns:
            df = df[df["type"] != "burst"]

        for AP_idx in df["sample_index"]:
            pre_AP_ms = 100
            post_AP_ms = 100
            ap = AP(sweep, AP_idx, pre_AP_ms, post_AP_ms)
            ap.remove_baseline(method="linear")
            ap.remove_baseline(method="mean")
            AP_l.append(ap)

            # --- Plot discrete AP ---
            t = ap.time.values
            v = ap.voltage.values
            ax.plot(t, v, 'o', markersize=2, color='gray', alpha=0.5)

# --- Compute and plot average AP ---
if AP_l:
    avg_ap, std_ap = average_APs(ap_list=AP_l)
    t = avg_ap.time.values
    v = avg_ap.voltage.values
    std_v = std_ap.voltage.values  # standard deviation

    # Plot average AP line
    #ax.plot(t, v, color='red', lw=2, label='Average AP')

    # Plot ±1 STD as shaded area
    ax.fill_between(t, v - std_v, v + std_v, color='orange', alpha=0.3, label='±1 STD')

    # Optional: mark metrics on average AP
    plot_basic_metrics(avg_ap, ax)
# --- Clean legend ---
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='best', frameon=False)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Voltage (mV)")
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()



TODO: Plot pour chaque fichier l'ensemble du sweep raw + detrend with metric + average

    --> créer un df par fichier et rajouter intra/extra, pre_AP_ms, post_AP_ms, sweep_name, + all metrics, condition (voir avec anna)
        --> merger avec le swwep d'avant of course pour calculer ISI and stuff 