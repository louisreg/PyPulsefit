import pandas as pd
from io import StringIO
from typing import Dict

def to_dfs(filepath: str) -> Dict[str, pd.DataFrame]:
    """
    Parse a .asc file containing one or more datasets separated by 'Sweep' headers.

    Each dataset block in the file typically looks like:
        Sweep 1_4_4
        6663 points
        "time[s]",   "trace1 [A]",   "trace2 [V]"
        0.00000E+000  -2.81250E-011   5.30938E-002
        3.46000E-004  -2.81250E-011   5.31250E-002
        ...

    The function automatically detects each "Sweep" section, reads its data,
    and returns a dictionary where each key is the sweep name (e.g., "Sweep 1_4_4")
    and each value is a pandas DataFrame containing the numeric data.

    Parameters
    ----------
    filepath : str
        Path to the .asc file to parse.

    Returns
    -------
    Dict[str, pd.DataFrame]
        A dictionary mapping each sweep name to its corresponding DataFrame.
    """
    datasets: Dict[str, pd.DataFrame] = {}

    # Read the file line by line
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Detect the start of a new dataset
        if line.startswith("Sweep"):
            sweep_name = line
            header_line = lines[i + 2].strip()

            # Extract column names
            columns = [h.strip().replace('"', '') for h in header_line.split(",")]

            # Find where the next "Sweep" starts or the file ends
            data_start = i + 3
            data_end = data_start
            while data_end < len(lines) and not lines[data_end].strip().startswith("Sweep"):
                data_end += 1

            # Join all numeric lines into one text block
            data_block = "".join(lines[data_start:data_end]).strip()

            # Read the numeric block into a DataFrame
            if data_block:  # avoid empty sections
                df = pd.read_csv(StringIO(data_block), sep=r"\s+", names=columns)
                datasets[sweep_name] = df

            # Jump to the next dataset
            i = data_end
        else:
            i += 1

    return datasets
