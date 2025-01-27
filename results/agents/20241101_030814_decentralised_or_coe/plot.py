import os
import pandas as pd
import matplotlib as plt

import os
import pandas as pd
import matplotlib.pyplot as plt

# Define labels for the runs we want to plot
labels = {
    "run_0": "Baseline",
    "run_1": "High Centralization",
    "run_2": "Moderate Centralization",
    "run_3": "Low Centralization",
    "run_4": "Very Low Centralization",
    "run_5": "Random Centralization",
}

# Load final results
final_results = {}
for folder in labels.keys():
    final_results[folder] = pd.read_csv(os.path.join(folder, "results.csv"))

# Plot all columns on the figure
for run, label in labels.items():
    df = final_results[run]
    state_columns = [col for col in df.columns if col not in ["step", "total_agents"]]

    # Create individual plots
    for col in state_columns:
        # Create new figure for each metric
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        # Plot the data
        df.plot(x="step", y=col, ax=ax, marker="o", label=label)

        # Customize the plot
        ax.set_title(f"{col} over Time", fontsize=14, pad=20)
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel(col, fontsize=12)
        ax.grid(True)
        ax.legend()

        # Add some padding around the plot
        plt.tight_layout()
        plt.savefig(f"agent_{run}_{col}.jpg")
        plt.close()
