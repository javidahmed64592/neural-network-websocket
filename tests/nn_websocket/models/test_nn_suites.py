"""Test suite for the src/nn_websocket/models/nn_suites.py module."""

import numpy as np
import pytest

from nn_websocket.ga.nn_member import NeuralNetworkMember
from nn_websocket.models.nn_suites import FitnessSuite, NeuroevolutionSuite
from nn_websocket.protobuf.frame_data_types import FitnessData, ObservationData, TrainRequestData
from nn_websocket.protobuf.nn_websocket_data_types import (
    FitnessApproachConfigData,
    GeneticAlgorithmConfigData,
    NeuralNetworkConfigData,
    NeuroevolutionConfigData,
)

rng = np.random.default_rng()


class TestNeuroevolutionSuite:
    """Test suite for NeuroevolutionSuite methods and behaviors."""

    def test_networks_property(
        self,
        mock_neuroevolution_suite: NeuroevolutionSuite,
        genetic_algorithm_config_data: GeneticAlgorithmConfigData,
        neural_network_config_data: NeuralNetworkConfigData,
    ) -> None:
        """Test that the networks property returns the correct neural networks."""
        suite = mock_neuroevolution_suite
        networks = suite.networks

        assert len(networks) == genetic_algorithm_config_data.population_size
        for network in networks:
            assert network._input_layer.size == neural_network_config_data.num_inputs
            assert network._output_layer.size == neural_network_config_data.num_outputs
            assert all(
                layer.size == neural_network_config_data.hidden_layer_sizes[i]
                for i, layer in enumerate(network._hidden_layers)
            )

    def test_from_config_data(self, neuroevolution_config_data: NeuroevolutionConfigData) -> None:
        """Test that the NeuroevolutionSuite can be created from configuration data."""
        mock_neuroevolution_suite = NeuroevolutionSuite.from_config_data(neuroevolution_config_data)
        assert (
            len(mock_neuroevolution_suite.nn_ga.nn_members)
            == neuroevolution_config_data.genetic_algorithm.population_size
        )

    def test_from_bytes(
        self,
        neuroevolution_config_data: NeuroevolutionConfigData,
        genetic_algorithm_config_data: GeneticAlgorithmConfigData,
    ) -> None:
        """Test that the NeuroevolutionSuite can be created from bytes."""
        mock_neuroevolution_suite = NeuroevolutionSuite.from_bytes(
            NeuroevolutionConfigData.to_bytes(neuroevolution_config_data)
        )
        assert len(mock_neuroevolution_suite.nn_ga.nn_members) == genetic_algorithm_config_data.population_size
        assert mock_neuroevolution_suite.nn_ga._mutation_rate == pytest.approx(
            genetic_algorithm_config_data.mutation_rate
        )

    def test_crossover_networks(
        self,
        mock_neuroevolution_suite: NeuroevolutionSuite,
        fitness_data: FitnessData,
    ) -> None:
        """Test that the crossover of networks updates the fitness of each member."""
        suite = mock_neuroevolution_suite
        suite.crossover_networks(fitness_data)
        for index, member in enumerate(suite.nn_ga.nn_members):
            assert member.fitness == fitness_data.values[index]

    def test_feedforward_through_networks(
        self,
        mock_neuroevolution_suite: NeuroevolutionSuite,
        genetic_algorithm_config_data: GeneticAlgorithmConfigData,
        neural_network_config_data: NeuralNetworkConfigData,
        observation_data: ObservationData,
    ) -> None:
        """Test that the feedforward through networks processes observation data correctly."""
        suite = mock_neuroevolution_suite
        observation_data.inputs *= genetic_algorithm_config_data.population_size
        action_data_list = suite.feedforward_through_networks(observation_data)

        assert (
            len(action_data_list.outputs)
            == genetic_algorithm_config_data.population_size * neural_network_config_data.num_outputs
        )


class TestFitnessSuite:
    """Test suite for FitnessSuite methods and behaviors."""

    def test_from_config_data(
        self,
        fitness_approach_config_data: FitnessApproachConfigData,
    ) -> None:
        """Test that the FitnessSuite can be created from configuration data."""
        mock_fitness_suite = FitnessSuite.from_config_data(fitness_approach_config_data)
        assert isinstance(mock_fitness_suite.nn_member, NeuralNetworkMember)

    def test_from_bytes(
        self,
        fitness_approach_config_data: FitnessApproachConfigData,
    ) -> None:
        """Test that the FitnessSuite can be created from bytes."""
        mock_fitness_suite = FitnessSuite.from_bytes(FitnessApproachConfigData.to_bytes(fitness_approach_config_data))
        assert isinstance(mock_fitness_suite.nn_member, NeuralNetworkMember)

    def test_feedforward(
        self,
        mock_fitness_suite: FitnessSuite,
        observation_data: ObservationData,
    ) -> None:
        """Test that the feedforward method processes the observation data correctly."""
        action_data = mock_fitness_suite.feedforward(observation_data)
        assert len(action_data.outputs) == mock_fitness_suite.nn_member._nn._output_layer.size

    def test_train(self, mock_fitness_suite: FitnessSuite, train_request_data: TrainRequestData) -> None:
        """Test that the training of the neural network works correctly."""
        initial_weights, initial_biases = mock_fitness_suite.nn_member.chromosome
        mock_fitness_suite.train(train_request_data)
        new_weights, new_biases = mock_fitness_suite.nn_member.chromosome
        assert not np.array_equal(
            [weight.vals for weight in initial_weights], [new_weight.vals for new_weight in new_weights]
        )
        assert not np.array_equal([bias.vals for bias in initial_biases], [new_bias.vals for new_bias in new_biases])
