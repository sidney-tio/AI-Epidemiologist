import os
import json
import pandas as pd
import matplotlib as plt
import seaborn as sns

# LOAD FINAL RESULTS:
folders = os.listdir("./")
final_results = {}
model_results = {}
agent_results = {}
for folder in folders:
    if folder.startswith("run") and os.path.isdir(folder):
        with open(os.path.join(folder, "final_info.json"), "r") as f:
            final_results[folder] = json.load(f)
        model_results[folder] = pd.read_csv(os.path.join(folder, "model_results.csv"))
        agent_results[folder] = pd.read_csv(os.path.join(folder, "agent_results.csv"))

# CREATE LEGEND -- PLEASE FILL IN YOUR RUN NAMES HERE
# Keep the names short, as these will be in the legend.
labels = {
    "run_0": "Baseline",
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

fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
axes1 = axes1.flatten()

for idx, col in enumerate(data_columns):
    for folder in model_results.keys():
        if col in model_results[folder].columns:
            sns.lineplot(data=model_results[folder], x=model_results[folder].index,
                        y=col, label=labels[folder], ax=axes1[idx])
    axes1[idx].set_title(col)
    axes1[idx].set_xlabel('Timestep')

# Remove empty subplots
for idx in range(len(data_columns), len(axes1)):
    fig1.delaxes(axes1[idx])

plt.tight_layout()
plt.save_fig(f"model_results.png")
plt.close()

# Agent Results Plot
first_folder = list(agent_results.keys())[0]
data_columns = [col for col in agent_results[first_folder].columns if col not in ['Step', 'AgentID']]
n_cols = 2
n_rows = (len(data_columns) + n_cols - 1) // n_cols

fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
axes2 = axes2.flatten()

for idx, col in enumerate(data_columns):
    for folder in agent_results.keys():
        if col in agent_results[folder].columns:
            # Calculate mean across agents for each timestep
            mean_data = agent_results[folder].groupby('Step')[col].mean()
            sns.lineplot(data=mean_data, x=mean_data.index, y=mean_data.values,
                        label=folder, ax=axes2[idx])
    axes2[idx].set_title(col)
    axes2[idx].set_xlabel('Step')

# Remove empty subplots
for idx in range(len(data_columns), len(axes2)):
    fig2.delaxes(axes2[idx])

plt.tight_layout()
plt.save_fig(f"agent_results.png")
plt.close()
