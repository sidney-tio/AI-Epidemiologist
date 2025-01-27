import os
import pandas as pd
import matplotlib.pyplot as plt

# Define labels for the runs to be plotted
labels = {
    "run_1": "Baseline",
    "run_2": "High Resilience",
    "run_3": "High Social Support",
    "run_4": "High Resilience and Social Support",
    "run_5": "Varying Initial Stress Levels",
}

# Load final results
final_results = {}
for folder in labels.keys():
    final_results[folder] = pd.read_csv(os.path.join(folder, "results.csv"))

# Plot all relevant columns
for run, label in labels.items():
    df = final_results[run]
    state_columns = [col for col in df.columns if col not in ["step", "total_agents"]]

    for col in state_columns:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        df.plot(x="step", y=col, ax=ax, marker="o", label=label)

        ax.set_title(f"{col} over Time", fontsize=14, pad=20)
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel(col, fontsize=12)
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(f"agent_{run}_{col}.jpg")
        plt.close()
