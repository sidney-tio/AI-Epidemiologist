import os
import json
import pandas as pd
import matplotlib as plt

try:
    import seaborn as sns

    seaborn_available = True
except ImportError:
    seaborn_available = False

# LOAD FINAL RESULTS:
folders = os.listdir("./")
final_results = {}
model_results = {}
agent_results = {}
for folder in folders:
    if folder.startswith("run") and os.path.isdir(folder):
        final_info_path = os.path.join(folder, "final_info.json")
        model_results_path = os.path.join(folder, "model_results.csv")
        agent_results_path = os.path.join(folder, "agent_results.csv")

        if os.path.exists(final_info_path):
            with open(final_info_path, "r") as f:
                final_results[folder] = json.load(f)

        if os.path.exists(model_results_path):
            model_results[folder] = pd.read_csv(model_results_path)

        if os.path.exists(agent_results_path):
            agent_results[folder] = pd.read_csv(agent_results_path)

# CREATE LEGEND -- PLEASE FILL IN YOUR RUN NAMES HERE
# Keep the names short, as these will be in the legend.
labels = {
    "run_0": "Baseline",
    "run_1": "Mental Health Attribute",
    "run_2": "Support Program",
    "run_3": "Targeted Support",
    "run_4": "Community Support",
    "run_5": "Dynamic Support",
}

# Use the run key as the default label if not specified
runs = list(final_results.keys())
for run in runs:
    if run not in labels:
        labels[run] = run

# Model results plot
first_folder = list(model_results.keys())[0]
data_columns = [col for col in model_results[first_folder].columns[1:]]
n_cols = 2
n_rows = (len(data_columns) + n_cols - 1) // n_cols

fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
axes1 = axes1.flatten()

for idx, col in enumerate(data_columns):
    for folder in model_results.keys():
        if col in model_results[folder].columns:
            if seaborn_available:
                sns.lineplot(
                    data=model_results[folder],
                    x=model_results[folder].index,
                    y=col,
                    label=labels[folder],
                    ax=axes1[idx],
                )
            else:
                axes1[idx].plot(
                    model_results[folder].index,
                    model_results[folder][col],
                    label=labels[folder],
                )
    axes1[idx].set_title(col)
    axes1[idx].set_xlabel("Timestep")

# Remove empty subplots
for idx in range(len(data_columns), len(axes1)):
    fig1.delaxes(axes1[idx])

plt.tight_layout()
plt.save_fig(f"model_results.png")
plt.close()

# Agent Results Plot
first_folder = list(agent_results.keys())[0]
data_columns = [
    col for col in agent_results[first_folder].columns if col not in ["Step", "AgentID"]
]
n_cols = 2
n_rows = (len(data_columns) + n_cols - 1) // n_cols

fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
axes2 = axes2.flatten()

for idx, col in enumerate(data_columns):
    for folder in agent_results.keys():
        if col in agent_results[folder].columns:
            # Calculate mean across agents for each timestep
            mean_data = agent_results[folder].groupby("Step")[col].mean()
            if seaborn_available:
                sns.lineplot(
                    data=mean_data,
                    x=mean_data.index,
                    y=mean_data.values,
                    label=folder,
                    ax=axes2[idx],
                )
            else:
                axes2[idx].plot(mean_data.index, mean_data.values, label=folder)
    axes2[idx].set_title(col)
    axes2[idx].set_xlabel("Step")

# Remove empty subplots
for idx in range(len(data_columns), len(axes2)):
    fig2.delaxes(axes2[idx])

plt.tight_layout()
plt.save_fig(f"agent_results.png")
plt.close()
