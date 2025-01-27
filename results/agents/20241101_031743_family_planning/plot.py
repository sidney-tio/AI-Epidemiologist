import os
import pandas as pd
import matplotlib.pyplot as plt

# Load final results
folders = [f for f in os.listdir("./") if os.path.isdir(f)]
final_results = {}
for folder in folders:
    results_path = os.path.join(folder, "results.csv")
    if os.path.exists(results_path):
        final_results[folder] = pd.read_csv(results_path)


# Define labels for the runs to be plotted
labels = {
    "run_0": "Baseline",
    "run_1": "High Social Support",
    "run_2": "Low Social Support",
    "run_3": "Medium Social Support",
    "run_4": "Very High Social Support",
    "run_5": "Mixed Social Support",
}

# Plot all columns on the figure
for run, label in labels.items():
    if run not in final_results:
        continue
    df = final_results[run]
    state_columns = [col for col in df.columns if col not in ["step", "total_agents"]]

    # Create individual plots
    for col in state_columns:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the data
        df.plot(x="step", y=col, ax=ax, marker="o", label=label)

        # Customize the plot
        ax.set_title(f"{col} over Time", fontsize=14, pad=20)
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel(col, fontsize=12)
        ax.grid(True)
        ax.legend()

        # Save the plot
        plt.tight_layout()
        plt.savefig(f"{run}_{col}.jpg")
        plt.close()
