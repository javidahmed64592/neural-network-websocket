from neural_network.layer import HiddenLayer, InputLayer, OutputLayer
from neural_network.math.matrix import Matrix

from nn_websocket.ga.nn_member import NeuralNetworkMember
from nn_websocket.protobuf.proto_types import NeuralNetworkConfigData


class TestNeuralNetworkMember:
    def test_initialization(self, nn_config_data: NeuralNetworkConfigData) -> None:
        """Test initialization of NeuralNetworkMember."""
        member = NeuralNetworkMember.from_config_data(nn_config_data)

        assert member._num_inputs == nn_config_data.num_inputs
        assert member._num_outputs == nn_config_data.num_outputs
        assert member._hidden_layer_sizes == nn_config_data.hidden_layer_sizes
        assert member._weights_range == (nn_config_data.weights_min, nn_config_data.weights_max)
        assert member._bias_range == (nn_config_data.bias_min, nn_config_data.bias_max)
        assert member._input_activation == nn_config_data.input_activation.get_class()
        assert member._hidden_activation == nn_config_data.hidden_activation.get_class()
        assert member._output_activation == nn_config_data.output_activation.get_class()

    def test_nn_layers_property(self, nn_config_data: NeuralNetworkConfigData) -> None:
        """Test the nn_layers property."""
        member = NeuralNetworkMember.from_config_data(nn_config_data)

        layers = member.nn_layers
        assert len(layers) == len(nn_config_data.hidden_layer_sizes) + 2
        assert isinstance(layers[0], InputLayer)
        assert all(isinstance(layer, HiddenLayer) for layer in layers[1:-1])
        assert isinstance(layers[-1], OutputLayer)

    def test_chromosome_property(self, nn_config_data: NeuralNetworkConfigData) -> None:
        """Test the chromosome property."""
        member = NeuralNetworkMember.from_config_data(nn_config_data)

        weights, bias = member.chromosome
        assert len(weights) == len(nn_config_data.hidden_layer_sizes) + 2
        assert len(bias) == len(nn_config_data.hidden_layer_sizes) + 2

        for weight_matrix in weights:
            assert isinstance(weight_matrix, Matrix)

        for bias_matrix in bias:
            assert isinstance(bias_matrix, Matrix)

    def test_chromosome_setter(self, nn_config_data: NeuralNetworkConfigData) -> None:
        """Test the chromosome setter."""
        member = NeuralNetworkMember.from_config_data(nn_config_data)
        other_member = NeuralNetworkMember.from_config_data(nn_config_data)

        member.chromosome = (other_member.chromosome[0], other_member.chromosome[1])

        assert member._nn.weights == other_member.chromosome[0]
        assert member._nn.bias == other_member.chromosome[1]

    def test_fitness_property(self, nn_config_data: NeuralNetworkConfigData) -> None:
        """Test the fitness property."""
        member = NeuralNetworkMember.from_config_data(nn_config_data)

        # Initially, the score is 0, so fitness should be 0
        assert member.fitness == 0

        # Set a score and check fitness
        score = 5
        member.fitness = score
        assert member.fitness == score

    def test_crossover_genes(self, nn_config_data: NeuralNetworkConfigData) -> None:
        """Test the crossover_genes method."""
        member = NeuralNetworkMember.from_config_data(nn_config_data)

        element = 0.5
        other_element = 0.8
        mutation_rate = 0.2
        random_range = (0.0, 1.0)

        # Test without mutation
        roll = 0.3
        result = member.crossover_genes(element, other_element, roll, mutation_rate, random_range)
        assert result in (element, other_element)

        # Test with mutation
        roll = 0.1
        result = member.crossover_genes(element, other_element, roll, mutation_rate, random_range)
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
