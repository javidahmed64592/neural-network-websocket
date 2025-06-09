from pathlib import Path

import numpy as np

from nn_websocket.models.config import Config
from nn_websocket.protobuf.frame_data_types import (
    FitnessData,
    FrameRequestData,
    ObservationData,
)

rng = np.random.default_rng()

CONFIG_FILEPATH = Path("config") / "websocket_config.json"


def get_config() -> Config:
    """Load the websocket configuration from a JSON file."""
    return Config.load_config(CONFIG_FILEPATH)


def get_random_observation(num_inputs: int) -> FrameRequestData:
    """Generate a random observation for a single agent."""
    inputs = rng.uniform(low=-1, high=1, size=num_inputs).astype(np.float32).tolist()
    observation = ObservationData(inputs=inputs)
    return FrameRequestData(observation=observation)


def get_random_population_fitness(num_agents: int) -> FrameRequestData:
    """Generate random fitness values for the population."""
    fitness_values = rng.uniform(low=0, high=1, size=num_agents).astype(np.float32).tolist()
    fitness_data = FitnessData(values=fitness_values)
    return FrameRequestData(fitness=fitness_data)
