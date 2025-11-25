import numpy as np
from pathlib import Path
from typing import List
import os
import glob


def noise_std_mad(signal: np.ndarray) -> float:
    """
    Robust noise standard deviation estimate using MAD.
    Works well if signal contains outliers/spikes.
    """
    med = np.median(signal)
    mad = np.median(np.abs(signal - med))
    return float(mad / 0.6745)

def list_asc_files(folder: str, recursive: bool = False) -> List[str]:
    """
    List all .asc or .ASC files in a given folder.

    Parameters
    ----------
    folder : str
        Path to the folder to search in.
    recursive : bool, optional
        If True, search recursively in subdirectories. Default is False.

    Returns
    -------
    List[str]
        List of absolute paths to .asc/.ASC files found.

    Examples
    --------
    >>> list_asc_files("/data/recordings")
    ['/data/recordings/cell1.asc', '/data/recordings/cell2.ASC']

    >>> list_asc_files("/data", recursive=True)
    ['/data/day1/cell1.asc', '/data/day2/cell3.ASC']
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    pattern = "**/*.[aA][sS][cC]" if recursive else "*.[aA][sS][cC]"
    return sorted(str(p.resolve()) for p in folder_path.glob(pattern))


def get_AP_summary(asc_path, sweep, output_dir=None):
    """
    Returns the path of the AP summary CSV file associated with a given ASC file and specific sweep.

    Arguments:
    - asc_path : str, path to the .ASC file
    - sweep : Sweep object or str, the sweep to retrieve
    - output_dir : str or None, folder where the CSV is stored. If None, defaults to 'raw_plot' next to the ASC file.

    Returns:
    - str : full path to the associated CSV if found
    - None : if no corresponding CSV is found
    """
    if not os.path.isfile(asc_path):
        print(f"❌ The ASC file '{asc_path}' does not exist.")
        return None

    if output_dir is None:
        asc_dir = os.path.dirname(asc_path)
        output_dir = os.path.join(asc_dir, "raw_plot")

    if not os.path.isdir(output_dir):
        print(f"❌ CSV folder '{output_dir}' does not exist.")
        return None

    # Handle sweep name
    sweep_name = sweep.name if hasattr(sweep, "name") else str(sweep)
    sweep_safe = sweep_name.replace(" ", "_").replace("/", "_")

    asc_name = os.path.basename(asc_path)
    asc_base, _ = os.path.splitext(asc_name)
    asc_base_safe = asc_base.replace(" ", "_")

    # Match CSV that includes both ASC base name and sweep name
    pattern = os.path.join(output_dir, f"{asc_base_safe}__{sweep_safe}__peaks.csv")
    csv_files = glob.glob(pattern)

    if not csv_files:
        print(f"⚠️ No CSV found for ASC '{asc_path}' and sweep '{sweep_name}' in '{output_dir}'")
        return None

    if len(csv_files) > 1:
        print(f"⚠️ Multiple CSVs found for ASC '{asc_path}' and sweep '{sweep_name}', returning the first one: {csv_files[0]}")

    return csv_files[0]