import numpy as np
import pytest

from nn_websocket.models.nn_suites import NeuroevolutionSuite
from nn_websocket.protobuf.frame_data_types import FitnessData, ObservationData
from nn_websocket.protobuf.neural_network_types import (
    GeneticAlgorithmConfigData,
    NeuralNetworkConfigData,
    NeuroevolutionConfigData,
)

rng = np.random.default_rng()


class TestNeuralNetworkSuite:
    def test_networks_property(
        self,
        mock_neuroevolution_suite: NeuroevolutionSuite,
        genetic_algorithm_config_data: GeneticAlgorithmConfigData,
        neural_network_config_data: NeuralNetworkConfigData,
    ) -> None:
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

    def test_from_bytes(
        self,
        neuroevolution_config_data: NeuroevolutionConfigData,
        genetic_algorithm_config_data: GeneticAlgorithmConfigData,
    ) -> None:
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
        suite = mock_neuroevolution_suite
        observation_data.inputs *= genetic_algorithm_config_data.population_size
        action_data_list = suite.feedforward_through_networks(observation_data)

        assert (
            len(action_data_list.outputs)
            == genetic_algorithm_config_data.population_size * neural_network_config_data.num_outputs
        )
