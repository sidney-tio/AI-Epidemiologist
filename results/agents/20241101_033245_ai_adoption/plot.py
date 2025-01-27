import os
import pandas as pd
import matplotlib.pyplot as plt

# Define labels for the runs to be plotted
labels = {
    "run_0": "Baseline",
    "run_1": "Increased Social Connectivity",
    "run_2": "Digital Literacy Campaign",
    "run_3": "Targeted Support",
    "run_4": "Mixed Approach",
    "run_5": "Control Group",
}

# Load final results
final_results = {}
for folder in labels.keys():
    file_path = os.path.join(folder, "results.csv")
    if os.path.exists(file_path):
        final_results[folder] = pd.read_csv(file_path)
    else:
        print(f"Warning: {file_path} does not exist and will be skipped.")


# PLOT ALL COLUMNS ON THE FIGURE
for i, run in enumerate(final_results.keys()):
    df = final_results[run]
    state_columns = [col for col in df.columns if col not in ["step", "total_agents"]]
    figures = []

    # Create individual plots
    for col in state_columns:
        # Create new figure for each metric
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        # Plot the data
        df.plot(x="step", y=col, ax=ax, marker="o")

        # Customize the plot
        ax.set_title(f"{col} over Time", fontsize=14, pad=20)
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel(col, fontsize=12)
        ax.grid(True)

        # Add some padding around the plot
        plt.tight_layout()
        plt.savefig(f"agent_{folder}_{col}.jpg")
        plt.close()
