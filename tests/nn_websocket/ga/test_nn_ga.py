"""Unit tests for the src/nn_websocket/ga/nn_ga.py module."""

import numpy as np

from nn_websocket.ga.nn_ga import NeuralNetworkGA
from nn_websocket.ga.nn_member import NeuralNetworkMember
from nn_websocket.protobuf.frame_data_types import FitnessData
from nn_websocket.protobuf.nn_websocket_data_types import GeneticAlgorithmConfigData, NeuralNetworkConfigData


class TestNeuralNetworkGA:
    """Test suite for the NeuralNetworkGA class."""

    def test_initialization(
        self,
        mock_neural_network_ga: NeuralNetworkGA,
        genetic_algorithm_config_data: GeneticAlgorithmConfigData,
        neural_network_config_data: NeuralNetworkConfigData,
    ) -> None:
        """Test initialization of NeuralNetworkGA."""
        assert len(mock_neural_network_ga.nn_members) == genetic_algorithm_config_data.population_size
        assert mock_neural_network_ga._mutation_rate == genetic_algorithm_config_data.mutation_rate

        assert all(
            member._num_inputs == neural_network_config_data.num_inputs for member in mock_neural_network_ga.nn_members
        )
        assert all(
            member._num_outputs == neural_network_config_data.num_outputs
            for member in mock_neural_network_ga.nn_members
        )
        assert all(
            member._hidden_layer_sizes == neural_network_config_data.hidden_layer_sizes
            for member in mock_neural_network_ga.nn_members
        )
        assert all(
            member._weights_range == (neural_network_config_data.weights_min, neural_network_config_data.weights_max)
            for member in mock_neural_network_ga.nn_members
        )
        assert all(
            member._bias_range == (neural_network_config_data.bias_min, neural_network_config_data.bias_max)
            for member in mock_neural_network_ga.nn_members
        )
        assert all(
            member._input_activation == neural_network_config_data.input_activation.get_class()
            for member in mock_neural_network_ga.nn_members
        )
        assert all(
            member._hidden_activation == neural_network_config_data.hidden_activation.get_class()
            for member in mock_neural_network_ga.nn_members
        )
        assert all(
            member._output_activation == neural_network_config_data.output_activation.get_class()
            for member in mock_neural_network_ga.nn_members
        )

    def test_nn_members_property(
        self,
        mock_neural_network_ga: NeuralNetworkGA,
        neural_network_config_data: NeuralNetworkConfigData,
    ) -> None:
        """Test the nn_members property."""
        members = mock_neural_network_ga.nn_members
        assert all(isinstance(member, NeuralNetworkMember) for member in members)

        for member in members:
            assert member._num_inputs == neural_network_config_data.num_inputs
            assert member._num_outputs == neural_network_config_data.num_outputs
            assert member._hidden_layer_sizes == neural_network_config_data.hidden_layer_sizes
            assert member._weights_range == (
                neural_network_config_data.weights_min,
                neural_network_config_data.weights_max,
            )
            assert member._bias_range == (neural_network_config_data.bias_min, neural_network_config_data.bias_max)
            assert member._input_activation == neural_network_config_data.input_activation.get_class()
            assert member._hidden_activation == neural_network_config_data.hidden_activation.get_class()
            assert member._output_activation == neural_network_config_data.output_activation.get_class()

    def test_population_size_property(
        self, mock_neural_network_ga: NeuralNetworkGA, genetic_algorithm_config_data: GeneticAlgorithmConfigData
    ) -> None:
        """Test the population_size property."""
        assert mock_neural_network_ga.population_size == genetic_algorithm_config_data.population_size

    def test_set_population_fitness(
        self, mock_neural_network_ga: NeuralNetworkGA, genetic_algorithm_config_data: GeneticAlgorithmConfigData
    ) -> None:
        """Test setting population fitness."""
        fitness_scores = np.arange(genetic_algorithm_config_data.population_size, dtype=float).tolist()
        mock_neural_network_ga.set_population_fitness(fitness_scores)

        for member, score in zip(mock_neural_network_ga.nn_members, fitness_scores, strict=False):
            assert member.fitness == score

    def test_evolve(
        self,
        mock_neural_network_ga: NeuralNetworkGA,
        fitness_data: FitnessData,
    ) -> None:
        """Test evolution of the neural network population."""
        initial_chromosomes = [member.chromosome for member in mock_neural_network_ga.nn_members]
        mock_neural_network_ga.evolve(fitness_data)
        new_chromosomes = [member.chromosome for member in mock_neural_network_ga.nn_members]

        for index, member in enumerate(mock_neural_network_ga.nn_members):
            assert member.fitness == fitness_data.values[index]

        assert all(mock_neural_network_ga._population._population_fitness == fitness_data.values)
        assert initial_chromosomes != new_chromosomes
