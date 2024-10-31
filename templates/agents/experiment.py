import os
import argparse
import json
from enum import Enum, auto
from typing import Dict, List, Set, Any, Optional
import networkx as nx
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from collections import defaultdict
from abc import ABC, abstractmethod

parser = argparse.ArgumentParser(description="Run experiment")
parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
args = parser.parse_args()

# Instead of inheritance, use separate Enums for each simulation type
class AgentState(str, Enum):
    """Base/default agent state"""
    NEUTRAL = "NEUTRAL"

class ExampleState(str, Enum):
    """Example simulation states"""
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"

class LonelinessState(str, Enum):
    """States for loneliness simulation"""
    CONNECTED = "CONNECTED"
    LONELY = "LONELY"
    RECOVERING = "RECOVERING"

@dataclass
class AgentTraits:
    """Base class for agent traits"""
    pass

class Agent:
    def __init__(self,
                 agent_id: int,
                 initial_state: Enum,
                 traits: Optional[AgentTraits] = None):
        self.id = agent_id
        self.state = initial_state
        self.traits = traits
        self.connections: Set[int] = set()
        self.attributes: Dict[str, Any] = {}
        self.history: List[Dict] = []
        self.time_in_state = 0

    def update_state(self, new_state: Enum):
        self.history.append({
            'time': len(self.history),
            'old_state': self.state,
            'new_state': new_state
        })
        self.state = new_state
        self.time_in_state = 0

    def add_connection(self, other_id: int):
        self.connections.add(other_id)

    def remove_connection(self, other_id: int):
        self.connections.discard(other_id)

    def get_connection_states(self, simulation) -> Dict[Enum, int]:
        state_counts = defaultdict(int)
        for conn_id in self.connections:
            if conn_id in simulation.agents:
                state_counts[simulation.agents[conn_id].state] += 1
        return dict(state_counts)

class BaseSimulation:
    def __init__(self,
                 num_agents: int,
                 connection_probability: float = 0.1):
        self.num_agents = num_agents
        self.connection_prob = connection_probability
        self.agents: Dict[int, Agent] = {}
        self.network = nx.Graph()
        self.step_count = 0
        self.statistics: List[Dict] = []

    def initialize_network(self):
        self.network = nx.erdos_renyi_graph(self.num_agents, self.connection_prob)
        for edge in self.network.edges():
            self.agents[edge[0]].add_connection(edge[1])
            self.agents[edge[1]].add_connection(edge[0])

    @abstractmethod
    def initialize_agents(self):
        pass

    @abstractmethod
    def update_agent(self, agent: Agent):
        pass

    def collect_statistics(self) -> dict:
        stats = {
            'step': self.step_count,
            'total_agents': len(self.agents)
        }

        state_counts = defaultdict(int)
        for agent in self.agents.values():
            state_counts[agent.state] += 1
        stats.update({f'count_{state.value}': count
                     for state, count in state_counts.items()})

        return stats

    def step(self):
        for agent in self.agents.values():
            self.update_agent(agent)
            agent.time_in_state += 1

        self.statistics.append(self.collect_statistics())
        self.step_count += 1

    def run(self, num_steps: int) -> pd.DataFrame:
        for _ in range(num_steps):
            self.step()
        return pd.DataFrame(self.statistics)

# Example implementation
class ExampleSimulation(BaseSimulation):
    def initialize_agents(self):
        for i in range(self.num_agents):
            self.agents[i] = Agent(
                agent_id=i,
                initial_state=np.random.choice(list(ExampleState)),
                traits=AgentTraits()
            )
        self.initialize_network()

    def update_agent(self, agent: Agent):
        neighbor_states = agent.get_connection_states(self)
        active_neighbors = neighbor_states.get(ExampleState.ACTIVE, 0)

        if active_neighbors > len(agent.connections) / 2:
            agent.update_state(ExampleState.ACTIVE)
        else:
            agent.update_state(ExampleState.INACTIVE)

# Example usage
if __name__ == "__main__":
    out_dir = args.out_dir
    sim = ExampleSimulation(num_agents=10, connection_probability=0.1)
    sim.initialize_agents()
    results = sim.run(num_steps=50)
    print("\nFinal Statistics:")
    print(results.iloc[-1])

    state_columns = [col for col in results.columns
                    if col not in ['step','total_agents']]
    means = {col: float(results[col].mean()) for col in state_columns}

    with open(os.path.join(out_dir, "final_info.json"), "w") as f:
        json.dump({'run':{'means': means}}, f)

    results.to_csv(os.path.join(out_dir, "results.csv"), index = False)