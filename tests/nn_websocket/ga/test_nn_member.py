from neural_network.layer import HiddenLayer, InputLayer, OutputLayer
from neural_network.math.matrix import Matrix

from nn_websocket.ga.nn_member import NeuralNetworkMember
from nn_websocket.protobuf.neural_network_types import NeuralNetworkConfigData


class TestNeuralNetworkMember:
    def test_initialization(
        self, mock_neural_network_member: NeuralNetworkMember, nn_config_data: NeuralNetworkConfigData
    ) -> None:
        """Test initialization of NeuralNetworkMember."""
        assert mock_neural_network_member._num_inputs == nn_config_data.num_inputs
        assert mock_neural_network_member._num_outputs == nn_config_data.num_outputs
        assert mock_neural_network_member._hidden_layer_sizes == nn_config_data.hidden_layer_sizes
        assert mock_neural_network_member._weights_range == (nn_config_data.weights_min, nn_config_data.weights_max)
        assert mock_neural_network_member._bias_range == (nn_config_data.bias_min, nn_config_data.bias_max)
        assert mock_neural_network_member._input_activation == nn_config_data.input_activation.get_class()
        assert mock_neural_network_member._hidden_activation == nn_config_data.hidden_activation.get_class()
        assert mock_neural_network_member._output_activation == nn_config_data.output_activation.get_class()

    def test_nn_layers_property(
        self, mock_neural_network_member: NeuralNetworkMember, nn_config_data: NeuralNetworkConfigData
    ) -> None:
        """Test the nn_layers property."""
        layers = mock_neural_network_member.nn_layers
        assert len(layers) == len(nn_config_data.hidden_layer_sizes) + 2
        assert isinstance(layers[0], InputLayer)
        assert all(isinstance(layer, HiddenLayer) for layer in layers[1:-1])
        assert isinstance(layers[-1], OutputLayer)

    def test_chromosome_property(
        self, mock_neural_network_member: NeuralNetworkMember, nn_config_data: NeuralNetworkConfigData
    ) -> None:
        """Test the chromosome property."""
        weights, bias = mock_neural_network_member.chromosome
        assert len(weights) == len(nn_config_data.hidden_layer_sizes) + 2
        assert len(bias) == len(nn_config_data.hidden_layer_sizes) + 2

        for weight_matrix in weights:
            assert isinstance(weight_matrix, Matrix)

        for bias_matrix in bias:
            assert isinstance(bias_matrix, Matrix)

    def test_chromosome_setter(
        self, mock_neural_network_member: NeuralNetworkMember, nn_config_data: NeuralNetworkConfigData
    ) -> None:
        """Test the chromosome setter."""
        other_member = NeuralNetworkMember.from_config_data(nn_config_data)

        mock_neural_network_member.chromosome = (other_member.chromosome[0], other_member.chromosome[1])

        assert mock_neural_network_member._nn.weights == other_member.chromosome[0]
        assert mock_neural_network_member._nn.bias == other_member.chromosome[1]

    def test_fitness_property(self, mock_neural_network_member: NeuralNetworkMember) -> None:
        """Test the fitness property."""
        # Initially, the score is 0, so fitness should be 0
        assert mock_neural_network_member.fitness == 0

        # Set a score and check fitness
        score = 5
        mock_neural_network_member.fitness = score
        assert mock_neural_network_member.fitness == score

    def test_crossover_genes(self, mock_neural_network_member: NeuralNetworkMember) -> None:
        """Test the crossover_genes method."""
        element = 0.5
        other_element = 0.8
        mutation_rate = 0.2
        random_range = (0.0, 1.0)

        # Test without mutation
        roll = 0.3
        result = mock_neural_network_member.crossover_genes(element, other_element, roll, mutation_rate, random_range)
        assert result in (element, other_element)

        # Test with mutation
        roll = 0.1
        result = mock_neural_network_member.crossover_genes(element, other_element, roll, mutation_rate, random_range)
        assert result not in (element, other_element)

    def test_crossover(self, nn_config_data: NeuralNetworkConfigData) -> None:
        """Test the crossover method."""
        parent_a = NeuralNetworkMember.from_config_data(nn_config_data)
        parent_b = NeuralNetworkMember.from_config_data(nn_config_data)
        new_member = NeuralNetworkMember.from_config_data(nn_config_data)
        mutation_rate = 0.1

        # Set chromosomes to known values for testing
        parent_a.chromosome = (parent_a.chromosome[0], parent_a.chromosome[1])
        parent_b.chromosome = (parent_b.chromosome[0], parent_b.chromosome[1])

        new_member.crossover(parent_a, parent_b, mutation_rate)

        # Check that the new member's chromosome is a mix of both parents'
        assert len(new_member.chromosome[0]) == len(parent_a.chromosome[0])
        assert len(new_member.chromosome[1]) == len(parent_a.chromosome[1])
