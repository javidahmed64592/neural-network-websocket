"""Utility functions for neural network websocket clients.

Provides helpers for loading configuration and generating random frame data for testing.
"""

from pathlib import Path

import numpy as np

from nn_websocket.models.config import Config
from nn_websocket.protobuf.frame_data_types import (
    FitnessData,
    FrameRequestData,
    ObservationData,
    TrainRequestData,
)

rng = np.random.default_rng()

CONFIG_FILEPATH = Path("config") / "websocket_config.json"


def get_config() -> Config:
    """Load the websocket configuration from a JSON file.

    :return Config:
        The loaded configuration object.
    """
    return Config.load_config(CONFIG_FILEPATH)


def get_random_observation_frame(num_inputs: int) -> FrameRequestData:
    """Generate a random observation for a single agent.

    :param int num_inputs:
        Number of inputs for the observation.
    :return FrameRequestData:
        Frame containing the random observation.
    """
    inputs = rng.uniform(low=-1, high=1, size=num_inputs).astype(np.float32).tolist()
    observation = ObservationData(inputs=inputs)
    return FrameRequestData(observation=observation)


def get_random_fitness_frame(num_agents: int) -> FrameRequestData:
    """Generate random fitness values for the population.

    :param int num_agents:
        Number of agents in the population.
    :return FrameRequestData:
        Frame containing the random fitness values.
    """
    fitness_values = rng.uniform(low=0, high=1, size=num_agents).astype(np.float32).tolist()
    fitness_data = FitnessData(values=fitness_values)
    return FrameRequestData(fitness=fitness_data)


def get_random_train_request_frame(batch_size: int, num_inputs: int) -> FrameRequestData:
    """Generate training data with observations and target fitness values.

    :param int batch_size:
        Number of training samples in the batch.
    :param int num_inputs:
        Number of inputs for each observation.
    :return FrameRequestData:
        Frame containing the training request data.
    """
    observations = []
    fitness_values = []

    for _ in range(batch_size):
        observations.append(get_random_observation_frame(num_inputs).observation)
        fitness_values.append(get_random_fitness_frame(1).fitness)

    train_request = TrainRequestData(observation=observations, fitness=fitness_values)  # type: ignore[arg-type]
    return FrameRequestData(train_request=train_request)
