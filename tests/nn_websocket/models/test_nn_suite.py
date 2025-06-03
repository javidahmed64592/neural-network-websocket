import numpy as np
import pytest

from nn_websocket.models.nn_suite import NeuralNetworkSuite
from nn_websocket.protobuf.proto_types import (
    FitnessData,
    GeneticAlgorithmConfigData,
    NeuralNetworkConfigData,
    ObservationData,
)

rng = np.random.default_rng()


class TestNeuralNetworkSuite:
    def test_networks_property(
        self,
        mock_neural_network_suite: NeuralNetworkSuite,
        ga_config_data: GeneticAlgorithmConfigData,
        nn_config_data: NeuralNetworkConfigData,
    ) -> None:
        suite = mock_neural_network_suite
        networks = suite.networks

        assert len(networks) == ga_config_data.population_size
        for network in networks:
            assert network._input_layer.size == nn_config_data.num_inputs
            assert network._output_layer.size == nn_config_data.num_outputs
            assert all(
                layer.size == nn_config_data.hidden_layer_sizes[i] for i, layer in enumerate(network._hidden_layers)
            )

    def test_from_bytes(
        self,
        mock_neural_network_suite: NeuralNetworkSuite,
        ga_config_data: GeneticAlgorithmConfigData,
    ) -> None:
        assert len(mock_neural_network_suite.nn_ga.nn_members) == ga_config_data.population_size
        assert mock_neural_network_suite.nn_ga._mutation_rate == pytest.approx(ga_config_data.mutation_rate)

    def test_crossover_networks(
        self,
        mock_neural_network_suite: NeuralNetworkSuite,
        fitness_data: FitnessData,
    ) -> None:
        suite = mock_neural_network_suite
        suite.crossover_networks(fitness_data)
        for index, member in enumerate(suite.nn_ga.nn_members):
            assert member.fitness == fitness_data.values[index]

    def test_feedforward_through_network(
        self, mock_neural_network_suite: NeuralNetworkSuite, nn_config_data: NeuralNetworkConfigData
    ) -> None:
        suite = mock_neural_network_suite
        observation_data = rng.random(nn_config_data.num_inputs)

        for network in suite.networks:
            action_data = suite.feedforward_through_network(network, observation_data)
            assert len(action_data) == nn_config_data.num_outputs

    def test_feedforward_through_networks(
        self,
        mock_neural_network_suite: NeuralNetworkSuite,
        ga_config_data: GeneticAlgorithmConfigData,
        nn_config_data: NeuralNetworkConfigData,
        observation_data: ObservationData,
    ) -> None:
        suite = mock_neural_network_suite
        action_data_list = suite.feedforward_through_networks(observation_data)

        assert len(action_data_list.outputs) == ga_config_data.population_size * nn_config_data.num_outputs

    def test_feedforward_through_networks_from_bytes(
        self,
        mock_neural_network_suite: NeuralNetworkSuite,
        ga_config_data: GeneticAlgorithmConfigData,
        nn_config_data: NeuralNetworkConfigData,
        observation_data: ObservationData,
    ) -> None:
        observation_data_bytes = ObservationData.to_bytes(observation_data)
        action_data_list = mock_neural_network_suite.feedforward_through_networks_from_bytes(observation_data_bytes)

        assert len(action_data_list.outputs) == ga_config_data.population_size * nn_config_data.num_outputs
