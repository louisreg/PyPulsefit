import numpy as np
from pathlib import Path
from typing import List


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




