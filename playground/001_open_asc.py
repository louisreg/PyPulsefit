import pandas as pd
import matplotlib.pyplot as plt

# Read the .asc file
df = pd.read_csv(
    "../ANALYSE Anna/GH4C1/2024/03072024/cell7 PA spont rxtx.asc",     
    skiprows=3,           # skip the first two lines ("Sweep ..." and "xxxx points")
    sep=r"\s+",           # split columns by one or more spaces
    names=["time", "current", "voltage"],  # column names
)
df = df.apply(pd.to_numeric, errors='coerce')
# Show the first few rows
print(df.head())

#plot it
fig, axs = plt.subplots(2,1)
axs[0].plot(df["time"],df["voltage"])
axs[1].plot(df["time"],df["current"])

fig.tight_layout()
plt.show()