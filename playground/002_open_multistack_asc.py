import pandas as pd
import matplotlib.pyplot as plt
import re

def read_multi_asc(filepath):
    """
    Read a .asc file containing multiple datasets separated by 'Sweep' headers.
    Returns a dictionary of {sweep_name: DataFrame}.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    datasets = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Detect the start of a new dataset
        if line.startswith("Sweep"):
            sweep_name = line.strip()
            n_points = int(re.findall(r'\d+', lines[i+1])[0])  # extract number of points
            header_line = lines[i+2].strip()
            
            # Extract column names
            columns = [h.strip().replace('"', '') for h in header_line.split(",")]

            # Find where the next "Sweep" starts or the file ends
            data_start = i + 3
            data_end = data_start
            while data_end < len(lines) and not lines[data_end].startswith("Sweep"):
                data_end += 1
            
            # Read the numeric block using pandas
            from io import StringIO
            data_block = "".join(lines[data_start:data_end])
            df = pd.read_csv(StringIO(data_block), sep=r"\s+", names=columns)
            
            # Store in dict
            datasets[sweep_name] = df
            
            # Move to the next dataset
            i = data_end
        else:
            i += 1
    
    return datasets

data = read_multi_asc("../ANALYSE Anna/GH4C1/2024/18042024/cell23 PA induits.asc")

# List available sweeps
print(data.keys())

# Access one DataFrame
df = data["Sweep 1_4_7"]
print(df.head())

# Plot one sweep
plt.plot(df["time[s]"], df["trace2 [V]"])
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.title("Sweep 1_4_7")
plt.show()