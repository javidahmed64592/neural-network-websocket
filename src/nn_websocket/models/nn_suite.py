from __future__ import annotations

from typing import cast

import numpy as np
from neural_network.neural_network import NeuralNetwork
from numpy.typing import NDArray

from nn_websocket.ga.nn_ga import NeuralNetworkGA
from nn_websocket.protobuf.proto_types import (
    ActionData,
    ConfigurationData,
    FitnessData,
    ObservationData,
)


class NeuralNetworkSuite:
    """A suite of neural networks for handling multiple configurations."""

    def __init__(self, nn_ga: NeuralNetworkGA) -> None:
        """Initialize the suite with a configuration."""
        self.nn_ga = nn_ga

    @property
    def networks(self) -> list[NeuralNetwork]:
        """Get the list of neural networks."""
        return [member._nn for member in self.nn_ga.nn_members]

    @classmethod
    def from_config_data(cls, config_data: ConfigurationData) -> NeuralNetworkSuite:
        """
        Create a NeuralNetworkSuite from the provided configuration data.

        Parameters:
            config_data (ConfigurationData): Configuration data for the neural networks

        Returns:
            NeuralNetworkSuite: An instance of NeuralNetworkSuite
        """
        nn_config_data = config_data.neural_network
        ga_config_data = config_data.genetic_algorithm

        nn_ga = NeuralNetworkGA.from_config_data(nn_config_data, ga_config_data)
        return cls(nn_ga)

    @classmethod
    def from_bytes(cls, config_data_bytes: bytes) -> NeuralNetworkSuite:
        """
        Create a NeuralNetworkSuite from a bytes representation of the configuration data.

        Parameters:
            config_data_bytes (bytes): Bytes representation of the configuration data

        Returns:
            NeuralNetworkSuite: An instance of NeuralNetworkSuite
        """
        config_data = ConfigurationData.from_bytes(config_data_bytes)
        return cls.from_config_data(config_data)

    def crossover_networks(self, fitness_data: FitnessData) -> None:
        """Perform crossover on the neural networks based on population fitness."""
        self.nn_ga.evolve(fitness_data)

    @staticmethod
    def feedforward_through_network(nn: NeuralNetwork, observation: NDArray) -> list[float]:
        """Feedforward through the neural network and return the action data."""
        return cast(list[float], nn.feedforward(observation))

    def feedforward_through_networks(self, observation_data: ObservationData) -> ActionData:
        """Feedforward through all networks and return a list of action data."""
        observations = np.reshape(
            observation_data.inputs,
            (len(self.networks), -1),
        )

        actions = np.array(
            [
                NeuralNetworkSuite.feedforward_through_network(network, observations[i])
                for i, network in enumerate(self.networks)
            ],
        )
        return ActionData(outputs=actions.flatten().tolist())

    def feedforward_through_networks_from_bytes(self, observation_data_bytes: bytes) -> ActionData:
        """Feedforward through all networks from bytes representation of observation data."""
        observation_data = ObservationData.from_bytes(observation_data_bytes)
        return self.feedforward_through_networks(observation_data)
