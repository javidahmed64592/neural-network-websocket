from __future__ import annotations

import numpy as np
from genetic_algorithm.ga import Member
from neural_network.layer import HiddenLayer, InputLayer, Layer, OutputLayer
from neural_network.math.activation_functions import ActivationFunction
from neural_network.math.matrix import Matrix
from neural_network.neural_network import NeuralNetwork

from nn_websocket.protobuf.neural_network_types import NeuralNetworkConfigData

rng = np.random.default_rng()


class NeuralNetworkMember(Member):
    """This class creates a Member for the genetic algorithm."""

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        hidden_layer_sizes: list[int],
        weights_range: tuple[float, float],
        bias_range: tuple[float, float],
        input_activation: type[ActivationFunction],
        hidden_activation: type[ActivationFunction],
        output_activation: type[ActivationFunction],
    ) -> None:
        """
        Initialise NeuralNetworkMember with a starting position, a width and a height.

        Parameters:
            hidden_layer_sizes (list[int]): Neural network hidden layer sizes
            weights_range (tuple[float, float]): Range for random weights
            bias_range (tuple[float, float]): Range for random biases
        """
        super().__init__()

        self._num_inputs = num_inputs
        self._num_outputs = num_outputs
        self._hidden_layer_sizes = hidden_layer_sizes
        self._weights_range = weights_range
        self._bias_range = bias_range
        self._input_activation = input_activation
        self._hidden_activation = hidden_activation
        self._output_activation = output_activation

        self._input_layer = InputLayer(size=self._num_inputs, activation=self._input_activation)
        self._hidden_layers = [
            HiddenLayer(
                size=size,
                activation=self._hidden_activation,
                weights_range=self._weights_range,
                bias_range=self._bias_range,
            )
            for size in self._hidden_layer_sizes
        ]
        self._output_layer = OutputLayer(
            size=self._num_outputs,
            activation=self._output_activation,
            weights_range=self._weights_range,
            bias_range=self._bias_range,
        )
        self._nn = NeuralNetwork.from_layers(layers=self.nn_layers)
        self._score: float = 0

    @classmethod
    def from_config_data(cls, config_data: NeuralNetworkConfigData) -> NeuralNetworkMember:
        """
        Create a NeuralNetworkMember from the provided configuration data.

        Parameters:
            config_data (NeuralNetworkConfigData): Configuration data for the neural network

        Returns:
            NeuralNetworkMember: A new instance of NeuralNetworkMember
        """
        return cls(
            num_inputs=config_data.num_inputs,
            num_outputs=config_data.num_outputs,
            hidden_layer_sizes=config_data.hidden_layer_sizes,
            weights_range=(config_data.weights_min, config_data.weights_max),
            bias_range=(config_data.bias_min, config_data.bias_max),
            input_activation=config_data.input_activation.get_class(),
            hidden_activation=config_data.hidden_activation.get_class(),
            output_activation=config_data.output_activation.get_class(),
        )

    @property
    def nn_layers(self) -> list[Layer]:
        return [self._input_layer, *self._hidden_layers, self._output_layer]

    @property
    def chromosome(self) -> tuple[list[Matrix], list[Matrix]]:
        return self._nn.weights, self._nn.bias

    @chromosome.setter
    def chromosome(self, new_chromosome: tuple[list[Matrix], list[Matrix]]) -> None:
        self._nn.weights = new_chromosome[0]
        self._nn.bias = new_chromosome[1]

    @property
    def fitness(self) -> float:
        return self._score

    @fitness.setter
    def fitness(self, score: float) -> None:
        self._score = score

    @staticmethod
    def crossover_genes(
        element: float, other_element: float, roll: float, mutation_rate: float, random_range: tuple[float, float]
    ) -> float:
        if roll < mutation_rate:
            return rng.uniform(low=random_range[0], high=random_range[1])

        return float(rng.choice([element, other_element], p=[0.5, 0.5]))

    def crossover(self, parent_a: NeuralNetworkMember, parent_b: NeuralNetworkMember, mutation_rate: float) -> None:
        """
        Crossover the chromosomes of two birds to create a new chromosome.

        Parameters:
            parent_a (NeuralNetworkMember): Used to construct new chromosome
            parent_b (NeuralNetworkMember): Used to construct new chromosome
            mutation_rate (float): Probability for mutations to occur
        """

        def crossover_weights(element: float, other_element: float, roll: float) -> float:
            return NeuralNetworkMember.crossover_genes(element, other_element, roll, mutation_rate, self._weights_range)

        def crossover_biases(element: float, other_element: float, roll: float) -> float:
            return NeuralNetworkMember.crossover_genes(element, other_element, roll, mutation_rate, self._bias_range)

        self._new_chromosome = NeuralNetwork.crossover(
            parent_a._nn,
            parent_b._nn,
            crossover_weights,
            crossover_biases,
        )
