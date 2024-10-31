import os
import pandas as pd
import matplotlib as plt

# LOAD FINAL RESULTS:
folders = os.listdir("./")
final_results = {}
for folder in folders:
    final_results[folder] = pd.read_csv(os.path.join(folder,'results.csv'))


#PLOT ALL COLUMNS ON THE FIGURE
for i, run in enumerate(final_results.key()):
    df = final_results[run]
    state_columns = [col for col in df.columns
                    if col not in ['step','total_agents']]
    figures = []

    # Create individual plots
    for col in state_columns:
        # Create new figure for each metric
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        # Plot the data
        df.plot(x='step', y=col, ax=ax, marker='o')

        # Customize the plot
        ax.set_title(f'{col} over Time', fontsize=14, pad=20)
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel(col, fontsize=12)
        ax.grid(True)

        # Add some padding around the plot
        plt.tight_layout()
        plt.savefig(f"agent_{folder}_{col}.jpg")
        plt.close()