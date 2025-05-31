from __future__ import annotations

from genetic_algorithm.ga import GeneticAlgorithm

from nn_websocket.ga.nn_member import NeuralNetworkMember
from nn_websocket.protobuf.proto_types import GeneticAlgorithmConfigData, NeuralNetworkConfigData


class NeuralNetworkGA(GeneticAlgorithm):
    """
    Genetic algorithm for training neural networks.
    """

    def __init__(
        self,
        members: list[NeuralNetworkMember],
        mutation_rate: float,
    ) -> None:
        """
        Initialise NeuralNetworkGA with a mutation rate.

        Parameters:
            members (list[NeuralNetworkMember]): Population of NeuralNetworkMembers
            mutation_rate (float): Population mutation rate
        """
        super().__init__(members, mutation_rate)

    @classmethod
    def from_config_data(
        cls,
        config_data: NeuralNetworkConfigData,
        ga_config_data: GeneticAlgorithmConfigData,
    ) -> NeuralNetworkGA:
        """
        Create a NeuralNetworkGA from the provided configuration data.

        Parameters:
            config_data (NeuralNetworkConfigData): Configuration data for the neural network
            ga_config_data (GeneticAlgorithmConfigData): Configuration data for the genetic algorithm

        Returns:
            neural_network_ga (NeuralNetworkGA): Neural Network Genetic Algorithm
        """
        return cls(
            [NeuralNetworkMember.from_config_data(config_data) for _ in range(ga_config_data.population_size)],
            ga_config_data.mutation_rate,
        )

    @property
    def nn_members(self) -> list[NeuralNetworkMember]:
        """
        Get the list of neural network members.

        Returns:
            list[NeuralNetworkMember]: List of neural network members
        """
        return self._population._members

    @property
    def population_size(self) -> int:
        """
        Get the size of the population.

        Returns:
            int: Size of the population
        """
        return self._population.size
