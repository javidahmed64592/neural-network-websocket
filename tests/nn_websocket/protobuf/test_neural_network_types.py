"""Unit tests for the src/nn_websocket/protobuf/neural_network_types.py module."""

import pytest
from neural_network.math.activation_functions import LinearActivation, ReluActivation, SigmoidActivation, TanhActivation

from nn_websocket.protobuf.compiled.NNWebsocketClasses_pb2 import (
    ActivationFunctionEnum,
    Configuration,
    FitnessApproachConfig,
    GeneticAlgorithmConfig,
    NeuralNetworkConfig,
    NeuroevolutionConfig,
)
from nn_websocket.protobuf.neural_network_types import (
    ActivationFunctionEnumData,
    ConfigurationData,
    FitnessApproachConfigData,
    GeneticAlgorithmConfigData,
    NeuralNetworkConfigData,
    NeuroevolutionConfigData,
)


class TestActivationFunctionEnumData:
    """Test cases for ActivationFunctionEnumData conversions and mapping."""

    def test_get_class(self) -> None:
        """Test getting the activation class from enum."""
        assert ActivationFunctionEnumData.LINEAR.get_class() == LinearActivation
        assert ActivationFunctionEnumData.RELU.get_class() == ReluActivation
        assert ActivationFunctionEnumData.SIGMOID.get_class() == SigmoidActivation
        assert ActivationFunctionEnumData.TANH.get_class() == TanhActivation

    def test_from_class(self) -> None:
        """Test getting the enum from the activation class."""
        assert ActivationFunctionEnumData.from_class(LinearActivation) == ActivationFunctionEnumData.LINEAR
        assert ActivationFunctionEnumData.from_class(ReluActivation) == ActivationFunctionEnumData.RELU
        assert ActivationFunctionEnumData.from_class(SigmoidActivation) == ActivationFunctionEnumData.SIGMOID
        assert ActivationFunctionEnumData.from_class(TanhActivation) == ActivationFunctionEnumData.TANH

    def test_from_protobuf(self) -> None:
        """Test creating enum from Protobuf enum value."""
        assert (
            ActivationFunctionEnumData.from_protobuf(ActivationFunctionEnum.LINEAR) == ActivationFunctionEnumData.LINEAR
        )
        assert ActivationFunctionEnumData.from_protobuf(ActivationFunctionEnum.RELU) == ActivationFunctionEnumData.RELU
        assert (
            ActivationFunctionEnumData.from_protobuf(ActivationFunctionEnum.SIGMOID)
            == ActivationFunctionEnumData.SIGMOID
        )
        assert ActivationFunctionEnumData.from_protobuf(ActivationFunctionEnum.TANH) == ActivationFunctionEnumData.TANH

    def test_to_protobuf(self) -> None:
        """Test getting the Protobuf enum value from enum."""
        assert (
            ActivationFunctionEnumData.to_protobuf(ActivationFunctionEnumData.LINEAR) == ActivationFunctionEnum.LINEAR
        )
        assert ActivationFunctionEnumData.to_protobuf(ActivationFunctionEnumData.RELU) == ActivationFunctionEnum.RELU
        assert (
            ActivationFunctionEnumData.to_protobuf(ActivationFunctionEnumData.SIGMOID) == ActivationFunctionEnum.SIGMOID
        )
        assert ActivationFunctionEnumData.to_protobuf(ActivationFunctionEnumData.TANH) == ActivationFunctionEnum.TANH


class TestNeuralNetworkDataType:
    """Test cases for NeuralNetworkConfigData conversions and serialization."""

    def test_from_protobuf(self, neural_network_config: NeuralNetworkConfig) -> None:
        """Test creating NeuralNetworkConfigData from NeuralNetworkConfig Protobuf."""
        neural_network_data_type = NeuralNetworkConfigData.from_protobuf(neural_network_config)

        assert neural_network_data_type.num_inputs == neural_network_config.num_inputs
        assert neural_network_data_type.hidden_layer_sizes == neural_network_config.hidden_layer_sizes
        assert neural_network_data_type.num_outputs == neural_network_config.num_outputs
        assert neural_network_data_type.weights_min == neural_network_config.weights_min
        assert neural_network_data_type.weights_max == neural_network_config.weights_max
        assert neural_network_data_type.bias_min == neural_network_config.bias_min
        assert neural_network_data_type.bias_max == neural_network_config.bias_max
        assert neural_network_data_type.input_activation == neural_network_config.input_activation
        assert neural_network_data_type.hidden_activation == neural_network_config.hidden_activation
        assert neural_network_data_type.output_activation == neural_network_config.output_activation
        assert neural_network_data_type.learning_rate == neural_network_config.learning_rate

    def test_to_protobuf(self, neural_network_config_data: NeuralNetworkConfigData) -> None:
        """Test converting NeuralNetworkConfigData to NeuralNetworkConfig Protobuf."""
        protobuf_data = NeuralNetworkConfigData.to_protobuf(neural_network_config_data)

        assert protobuf_data.num_inputs == neural_network_config_data.num_inputs
        assert protobuf_data.hidden_layer_sizes == neural_network_config_data.hidden_layer_sizes
        assert protobuf_data.num_outputs == neural_network_config_data.num_outputs
        assert protobuf_data.weights_min == neural_network_config_data.weights_min
        assert protobuf_data.weights_max == neural_network_config_data.weights_max
        assert protobuf_data.bias_min == neural_network_config_data.bias_min
        assert protobuf_data.bias_max == neural_network_config_data.bias_max
        assert protobuf_data.input_activation == neural_network_config_data.input_activation
        assert protobuf_data.hidden_activation == neural_network_config_data.hidden_activation
        assert protobuf_data.output_activation == neural_network_config_data.output_activation
        assert protobuf_data.learning_rate == neural_network_config_data.learning_rate

    def test_to_bytes(self, neural_network_config_data: NeuralNetworkConfigData) -> None:
        """Test serializing NeuralNetworkConfigData to bytes."""
        assert isinstance(NeuralNetworkConfigData.to_bytes(neural_network_config_data), bytes)

    def test_from_bytes(self, neural_network_config_data: NeuralNetworkConfigData) -> None:
        """Test deserializing bytes to NeuralNetworkConfigData."""
        msg_bytes = NeuralNetworkConfigData.to_bytes(neural_network_config_data)
        result = NeuralNetworkConfigData.from_bytes(msg_bytes)

        assert result.num_inputs == neural_network_config_data.num_inputs
        assert result.hidden_layer_sizes == neural_network_config_data.hidden_layer_sizes
        assert result.num_outputs == neural_network_config_data.num_outputs
        assert result.input_activation == neural_network_config_data.input_activation
        assert result.hidden_activation == neural_network_config_data.hidden_activation
        assert result.output_activation == neural_network_config_data.output_activation
        assert result.weights_min == pytest.approx(neural_network_config_data.weights_min)
        assert result.weights_max == pytest.approx(neural_network_config_data.weights_max)
        assert result.bias_min == pytest.approx(neural_network_config_data.bias_min)
        assert result.bias_max == pytest.approx(neural_network_config_data.bias_max)
        assert result.learning_rate == neural_network_config_data.learning_rate


# Training methods
class TestConfigurationData:
    """Test cases for ConfigurationData conversions and serialization."""

    def test_from_protobuf(
        self, configuration_neuroevolution: Configuration, configuration_fitness: Configuration
    ) -> None:
        """Test creating ConfigurationData from Configuration Protobufs."""
        config_data_neuroevolution = ConfigurationData.from_protobuf(configuration_neuroevolution)
        config_data_fitness = ConfigurationData.from_protobuf(configuration_fitness)

        assert isinstance(config_data_neuroevolution.neuroevolution, NeuroevolutionConfigData)
        assert isinstance(config_data_fitness.fitness_approach, FitnessApproachConfigData)

    def test_to_protobuf(
        self, configuration_data_neuroevolution: ConfigurationData, configuration_data_fitness: ConfigurationData
    ) -> None:
        """Test converting ConfigurationData to Configuration Protobufs."""
        protobuf_neuroevolution = ConfigurationData.to_protobuf(configuration_data_neuroevolution)
        protobuf_fitness = ConfigurationData.to_protobuf(configuration_data_fitness)

        assert isinstance(protobuf_neuroevolution.neuroevolution, NeuroevolutionConfig)
        assert isinstance(protobuf_fitness.fitness_approach, FitnessApproachConfig)

    def test_to_bytes(
        self, configuration_data_neuroevolution: ConfigurationData, configuration_data_fitness: ConfigurationData
    ) -> None:
        """Test serializing ConfigurationData to bytes."""
        assert isinstance(ConfigurationData.to_bytes(configuration_data_neuroevolution), bytes)
        assert isinstance(ConfigurationData.to_bytes(configuration_data_fitness), bytes)

    def test_from_bytes(
        self, configuration_data_neuroevolution: ConfigurationData, configuration_data_fitness: ConfigurationData
    ) -> None:
        """Test deserializing bytes to ConfigurationData."""
        msg_bytes_neuroevolution = ConfigurationData.to_bytes(configuration_data_neuroevolution)
        msg_bytes_fitness = ConfigurationData.to_bytes(configuration_data_fitness)

        result_neuroevolution = ConfigurationData.from_bytes(msg_bytes_neuroevolution)
        result_fitness = ConfigurationData.from_bytes(msg_bytes_fitness)

        assert isinstance(result_neuroevolution.neuroevolution, NeuroevolutionConfigData)
        assert isinstance(result_fitness.fitness_approach, FitnessApproachConfigData)


class TestGeneticAlgorithmConfigData:
    """Test cases for GeneticAlgorithmConfigData conversions and serialization."""

    def test_from_protobuf(self, genetic_algorithm_config: GeneticAlgorithmConfig) -> None:
        """Test creating GeneticAlgorithmConfigData from GeneticAlgorithmConfig Protobuf."""
        ga_data = GeneticAlgorithmConfigData.from_protobuf(genetic_algorithm_config)

        assert ga_data.population_size == genetic_algorithm_config.population_size
        assert ga_data.mutation_rate == pytest.approx(genetic_algorithm_config.mutation_rate)

    def test_to_protobuf(self, genetic_algorithm_config_data: GeneticAlgorithmConfigData) -> None:
        """Test converting GeneticAlgorithmConfigData to GeneticAlgorithmConfig Protobuf."""
        protobuf_data = GeneticAlgorithmConfigData.to_protobuf(genetic_algorithm_config_data)

        assert protobuf_data.population_size == genetic_algorithm_config_data.population_size
        assert protobuf_data.mutation_rate == pytest.approx(genetic_algorithm_config_data.mutation_rate)

    def test_to_bytes(self, genetic_algorithm_config_data: GeneticAlgorithmConfigData) -> None:
        """Test serializing GeneticAlgorithmConfigData to bytes."""
        assert isinstance(GeneticAlgorithmConfigData.to_bytes(genetic_algorithm_config_data), bytes)

    def test_from_bytes(self, genetic_algorithm_config_data: GeneticAlgorithmConfigData) -> None:
        """Test deserializing bytes to GeneticAlgorithmConfigData."""
        msg_bytes = GeneticAlgorithmConfigData.to_bytes(genetic_algorithm_config_data)
        result = GeneticAlgorithmConfigData.from_bytes(msg_bytes)

        assert result.population_size == genetic_algorithm_config_data.population_size
        assert result.mutation_rate == pytest.approx(genetic_algorithm_config_data.mutation_rate)


class TestNeuroevolutionConfigData:
    """Test cases for NeuroevolutionConfigData conversions and serialization."""

    def test_from_protobuf(self, neuroevolution_config: NeuroevolutionConfig) -> None:
        """Test creating NeuroevolutionConfigData from NeuroevolutionConfig Protobuf."""
        config_data = NeuroevolutionConfigData.from_protobuf(neuroevolution_config)

        assert isinstance(config_data.neural_network, NeuralNetworkConfigData)
        assert isinstance(config_data.genetic_algorithm, GeneticAlgorithmConfigData)

    def test_to_protobuf(self, neuroevolution_config_data: NeuroevolutionConfigData) -> None:
        """Test converting NeuroevolutionConfigData to NeuroevolutionConfig Protobuf."""
        protobuf_data = NeuroevolutionConfigData.to_protobuf(neuroevolution_config_data)

        assert isinstance(protobuf_data.neural_network, NeuralNetworkConfig)
        assert isinstance(protobuf_data.genetic_algorithm, GeneticAlgorithmConfig)

    def test_to_bytes(self, neuroevolution_config_data: NeuroevolutionConfigData) -> None:
        """Test serializing NeuroevolutionConfigData to bytes."""
        assert isinstance(NeuroevolutionConfigData.to_bytes(neuroevolution_config_data), bytes)

    def test_from_bytes(self, neuroevolution_config_data: NeuroevolutionConfigData) -> None:
        """Test deserializing bytes to NeuroevolutionConfigData."""
        msg_bytes = NeuroevolutionConfigData.to_bytes(neuroevolution_config_data)
        result = NeuroevolutionConfigData.from_bytes(msg_bytes)

        assert isinstance(result.neural_network, NeuralNetworkConfigData)
        assert isinstance(result.genetic_algorithm, GeneticAlgorithmConfigData)


class TestFitnessApproachConfigData:
    """Test cases for FitnessApproachConfigData conversions and serialization."""

    def test_from_protobuf(self, fitness_approach_config: FitnessApproachConfig) -> None:
        """Test creating FitnessApproachConfigData from FitnessApproachConfig Protobuf."""
        config_data = FitnessApproachConfigData.from_protobuf(fitness_approach_config)

        assert isinstance(config_data.neural_network, NeuralNetworkConfigData)

    def test_to_protobuf(self, fitness_approach_config_data: FitnessApproachConfigData) -> None:
        """Test converting FitnessApproachConfigData to FitnessApproachConfig Protobuf."""
        protobuf_data = FitnessApproachConfigData.to_protobuf(fitness_approach_config_data)

        assert isinstance(protobuf_data.neural_network, NeuralNetworkConfig)

    def test_to_bytes(self, fitness_approach_config_data: FitnessApproachConfigData) -> None:
        """Test serializing FitnessApproachConfigData to bytes."""
        assert isinstance(FitnessApproachConfigData.to_bytes(fitness_approach_config_data), bytes)

    def test_from_bytes(self, fitness_approach_config_data: FitnessApproachConfigData) -> None:
        """Test deserializing bytes to FitnessApproachConfigData."""
        msg_bytes = FitnessApproachConfigData.to_bytes(fitness_approach_config_data)
        result = FitnessApproachConfigData.from_bytes(msg_bytes)

        assert isinstance(result.neural_network, NeuralNetworkConfigData)
