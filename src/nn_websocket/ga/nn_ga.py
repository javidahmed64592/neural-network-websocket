"""Genetic algorithm for evolving neural network populations."""

from __future__ import annotations

from typing import cast

from genetic_algorithm.ga import GeneticAlgorithm

from nn_websocket.ga.nn_member import NeuralNetworkMember
from nn_websocket.protobuf.frame_data_types import FitnessType
from nn_websocket.protobuf.nn_websocket_data_types import GeneticAlgorithmConfigType, NeuralNetworkConfigType


class NeuralNetworkGA(GeneticAlgorithm):
    """Genetic algorithm for training neural networks."""

    def __init__(
        self,
        members: list[NeuralNetworkMember],
        mutation_rate: float,
    ) -> None:
        """Initialise NeuralNetworkGA with a mutation rate.

        :param list[NeuralNetworkMember] members:
            Population of NeuralNetworkMembers.
        :param float mutation_rate:
            Population mutation rate.
        """
        super().__init__(members, mutation_rate)

    @classmethod
    def from_config_data(
        cls,
        nn_config_data: NeuralNetworkConfigType,
        ga_config_data: GeneticAlgorithmConfigType,
    ) -> NeuralNetworkGA:
        """Create a NeuralNetworkGA from the provided configuration data.

        :param NeuralNetworkConfigType nn_config_data:
            ConfigData data for the neural network.
        :param GeneticAlgorithmConfigType ga_config_data:
            ConfigData data for the genetic algorithm.
        :return NeuralNetworkGA:
            Neural Network Genetic Algorithm.
        """
        return cls(
            [NeuralNetworkMember.from_config_data(nn_config_data) for _ in range(ga_config_data.population_size)],
            ga_config_data.mutation_rate,
        )

    @property
    def nn_members(self) -> list[NeuralNetworkMember]:
        """Get the list of neural network members.

        :return list[NeuralNetworkMember]:
            List of neural network members.
        """
        return cast(list[NeuralNetworkMember], self._population._members)

    @property
    def population_size(self) -> int:
        """Get the size of the population.

        :return int:
            Size of the population.
        """
        return int(self._population.size)

    def set_population_fitness(self, fitness_scores: list[float]) -> None:
        """Set the fitness scores for the population.

        :param list[float] fitness_scores:
            List of fitness scores for each member in the population.
        """
        for member, score in zip(self.nn_members, fitness_scores, strict=False):
            member.fitness = score

    def evolve(self, fitness_data: FitnessType) -> None:
        """Evolve the population based on the provided fitness data.

        :param FitnessType fitness_data:
            Population fitness data containing fitness scores.
        """
        self.set_population_fitness(fitness_data.values)
        self._population.evaluate()
        self._evolve()
