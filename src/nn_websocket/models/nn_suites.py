"""Neural network suite classes for managing populations and fitness-based training."""

from __future__ import annotations

import numpy as np
from neural_network.neural_network import NeuralNetwork

from nn_websocket.ga.nn_ga import NeuralNetworkGA
from nn_websocket.ga.nn_member import NeuralNetworkMember
from nn_websocket.protobuf.frame_data_types import ActionData, FitnessData, ObservationData, TrainRequestData
from nn_websocket.protobuf.neural_network_types import (
    FitnessApproachConfigData,
    NeuroevolutionConfigData,
)


class NeuroevolutionSuite:
    """A suite of neural networks for handling multiple configurations."""

    def __init__(self, nn_ga: NeuralNetworkGA) -> None:
        """Initialize the suite with a configuration.

        :param NeuralNetworkGA nn_ga:
            The genetic algorithm instance for the suite.
        """
        self.nn_ga = nn_ga

    @property
    def networks(self) -> list[NeuralNetwork]:
        """Get the list of neural networks.

        :return list[NeuralNetwork]:
            List of neural network instances.
        """
        return [member._nn for member in self.nn_ga.nn_members]

    @classmethod
    def from_config_data(cls, config_data: NeuroevolutionConfigData) -> NeuroevolutionSuite:
        """Create a NeuroevolutionSuite from the provided configuration data.

        :param NeuroevolutionConfigData config_data:
            Configuration data for the neural networks.
        :return NeuroevolutionSuite:
            An instance of NeuroevolutionSuite.
        """
        nn_config_data = config_data.neural_network
        ga_config_data = config_data.genetic_algorithm
        nn_ga = NeuralNetworkGA.from_config_data(nn_config_data, ga_config_data)
        return cls(nn_ga)

    @classmethod
    def from_bytes(cls, config_data_bytes: bytes) -> NeuroevolutionSuite:
        """Create a NeuroevolutionSuite from a bytes representation of the configuration data.

        :param bytes config_data_bytes:
            Bytes representation of the configuration data.
        :return NeuroevolutionSuite:
            An instance of NeuroevolutionSuite.
        """
        config_data = NeuroevolutionConfigData.from_bytes(config_data_bytes)
        return cls.from_config_data(config_data)

    def feedforward_through_networks(self, observation_data: ObservationData) -> ActionData:
        """Feedforward through all networks and return a list of action data.

        :param ObservationData observation_data:
            The observation data to process.
        :return ActionData:
            The resulting action data from the neural networks.
        """
        observations = np.reshape(
            observation_data.inputs,
            (len(self.networks), -1),
        )

        actions = np.array(
            [network.feedforward(observations[i]) for i, network in enumerate(self.networks)],
        )
        return ActionData(outputs=actions.flatten().tolist())

    def crossover_networks(self, fitness_data: FitnessData) -> None:
        """Perform crossover on the neural networks based on population fitness.

        :param FitnessData fitness_data:
            The fitness data to guide the crossover process.
        """
        self.nn_ga.evolve(fitness_data)


class FitnessSuite:
    """A suite for handling fitness calculations."""

    def __init__(self, nn_member: NeuralNetworkMember) -> None:
        """Initialize the fitness suite with a neural network member.

        :param NeuralNetworkMember nn_member:
            The neural network member for the suite.
        """
        self.nn_member = nn_member

    @classmethod
    def from_config_data(cls, config_data: FitnessApproachConfigData) -> FitnessSuite:
        """Create a FitnessSuite from the provided configuration data.

        :param FitnessApproachConfigData config_data:
            Configuration data for the fitness approach.
        :return FitnessSuite:
            An instance of FitnessSuite.
        """
        nn_member = NeuralNetworkMember.from_config_data(config_data.neural_network)
        return cls(nn_member)

    @classmethod
    def from_bytes(cls, config_data_bytes: bytes) -> FitnessSuite:
        """Create a FitnessSuite from a bytes representation of the configuration data.

        :param bytes config_data_bytes:
            Bytes representation of the configuration data.
        :return FitnessSuite:
            An instance of FitnessSuite.
        """
        config_data = FitnessApproachConfigData.from_bytes(config_data_bytes)
        return cls.from_config_data(config_data)

    def feedforward(self, observation_data: ObservationData) -> ActionData:
        """Feedforward through the neural network and return action data.

        :param ObservationData observation_data:
            The observation data to process.
        :return ActionData:
            The resulting action data from the neural network.
        """
        actions = self.nn_member._nn.feedforward(observation_data.inputs)
        return ActionData(outputs=actions)

    def train(self, train_request_data: TrainRequestData) -> None:
        """Train the neural network member.

        :param TrainRequestData train_request_data:
            The training request data.
        """
        self.nn_member._nn.run_fitness_training(
            [observation.inputs for observation in train_request_data.observation],
            [value for fitness in train_request_data.fitness for value in fitness.values],
        )
