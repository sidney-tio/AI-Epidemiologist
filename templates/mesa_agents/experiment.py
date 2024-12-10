import mesa
import os
import numpy as np
import pandas as pd
import json
import argparse

def compute_gini(model):
    agent_wealths = [agent.wealth for agent in model.agents]
    x = sorted(agent_wealths)
    n = model.num_agents
    B = sum(xi * (n - i) for i, xi in enumerate(x)) / (n * sum(x))
    return 1 + (1 / n) - 2 * B


class ABModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, n, width, height, seed=None):
        super().__init__(seed=seed)
        self.num_agents = n
        self.grid = mesa.space.MultiGrid(width, height, True)
        self.running = True

        # Create agents
        for _ in range(self.num_agents):
            a = Agent(self)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        self.datacollector = mesa.DataCollector(
            model_reporters={"Gini": compute_gini},
            agent_reporters={"Wealth": "wealth", "Steps_not_given": "steps_not_given"},
        )

    def step(self):
        self.datacollector.collect(self)
        self.agents.shuffle_do("move")
        self.agents.shuffle_do("give_money")


class Agent(mesa.Agent):
    """An agent with that interacts with the environment."""

    def __init__(self, model):
        super().__init__(model)
        self.wealth = 1
        self.steps_not_given = 0

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def give_money(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        cellmates.pop(cellmates.index(self))
        if len(cellmates) > 0 and self.wealth > 0:
            other = self.random.choice(cellmates)
            other.wealth += 1
            self.wealth -= 1
            self.steps_not_given = 0
        else:
            self.steps_not_given += 1


if __name__== "__main__":
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
    args = parser.parse_args()
    N_AGENTS = 100
    WIDTH = 20
    HEIGHT = 20
    STEPS = 100
    out_dir = args.out_dir
    os.makedirs(out_dir,exist_ok=True)
    model = ABModel(n=N_AGENTS, width=WIDTH,height=HEIGHT)

    for _ in range(STEPS):
        model.step()

    model_df = model.datacollector.get_model_vars_dataframe()
    model_df.to_csv(os.path.join(out_dir, "model_results.csv"))
    agent_df = model.datacollector.get_agent_vars_dataframe()
    agent_df.to_csv(os.path.join(out_dir, "agent_results.csv"))

    state_columns = [col for col in model_df.columns]
    agent_columns = [col for col in agent_df.columns]
    means = {col: float(model_df[col].mean()) for col in state_columns}
    agent_means = {col: float(agent_df[col].mean()) for col in agent_columns}

    file_path = os.path.join("./", "final_info.json")
    run_data = {out_dir: {'means':{'model_means': means, 'agent_means': agent_means}}}
    with open(os.path.join(out_dir, "final_info.json"), "w") as f:
        json.dump(run_data,f)

    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r') as f:
            existing_data = json.load(f)
        existing_data.update(run_data)
        run_data = existing_data

    with open(file_path, 'w') as f:
        json.dump(run_data, f)

