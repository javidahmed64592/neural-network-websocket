import numpy as np

from nn_websocket.ga.nn_ga import NeuralNetworkGA
from nn_websocket.ga.nn_member import NeuralNetworkMember
from nn_websocket.protobuf.proto_types import GeneticAlgorithmConfigData, NeuralNetworkConfigData


class TestNeuralNetworkGA:
    def test_initialization(
        self,
        mock_neural_network_ga: NeuralNetworkGA,
        ga_config_data: GeneticAlgorithmConfigData,
        nn_config_data: NeuralNetworkConfigData,
    ) -> None:
        """Test initialization of NeuralNetworkGA."""
        assert len(mock_neural_network_ga.nn_members) == ga_config_data.population_size
        assert mock_neural_network_ga._mutation_rate == ga_config_data.mutation_rate

        assert all(member._num_inputs == nn_config_data.num_inputs for member in mock_neural_network_ga.nn_members)
        assert all(member._num_outputs == nn_config_data.num_outputs for member in mock_neural_network_ga.nn_members)
        assert all(
            member._hidden_layer_sizes == nn_config_data.hidden_layer_sizes
            for member in mock_neural_network_ga.nn_members
        )
        assert all(
            member._weights_range == (nn_config_data.weights_min, nn_config_data.weights_max)
            for member in mock_neural_network_ga.nn_members
        )
        assert all(
            member._bias_range == (nn_config_data.bias_min, nn_config_data.bias_max)
            for member in mock_neural_network_ga.nn_members
        )
        assert all(
            member._input_activation == nn_config_data.input_activation.get_class()
            for member in mock_neural_network_ga.nn_members
        )
        assert all(
            member._hidden_activation == nn_config_data.hidden_activation.get_class()
            for member in mock_neural_network_ga.nn_members
        )
        assert all(
            member._output_activation == nn_config_data.output_activation.get_class()
            for member in mock_neural_network_ga.nn_members
        )

    def test_nn_members_property(
        self,
        mock_neural_network_ga: NeuralNetworkGA,
        nn_config_data: NeuralNetworkConfigData,
    ) -> None:
        """Test the nn_members property."""
        members = mock_neural_network_ga.nn_members
        assert all(isinstance(member, NeuralNetworkMember) for member in members)

        for member in members:
            assert member._num_inputs == nn_config_data.num_inputs
            assert member._num_outputs == nn_config_data.num_outputs
            assert member._hidden_layer_sizes == nn_config_data.hidden_layer_sizes
            assert member._weights_range == (nn_config_data.weights_min, nn_config_data.weights_max)
            assert member._bias_range == (nn_config_data.bias_min, nn_config_data.bias_max)
            assert member._input_activation == nn_config_data.input_activation.get_class()
            assert member._hidden_activation == nn_config_data.hidden_activation.get_class()
            assert member._output_activation == nn_config_data.output_activation.get_class()

    def test_population_size_property(
        self, mock_neural_network_ga: NeuralNetworkGA, ga_config_data: GeneticAlgorithmConfigData
    ) -> None:
        """Test the population_size property."""
        assert mock_neural_network_ga.population_size == ga_config_data.population_size

    def test_set_population_fitness(
        self, mock_neural_network_ga: NeuralNetworkGA, ga_config_data: GeneticAlgorithmConfigData
    ) -> None:
        """Test setting population fitness."""
        fitness_scores = np.arange(ga_config_data.population_size, dtype=float).tolist()
        mock_neural_network_ga.set_population_fitness(fitness_scores)

        for member, score in zip(mock_neural_network_ga.nn_members, fitness_scores, strict=False):
            assert member.fitness == score
