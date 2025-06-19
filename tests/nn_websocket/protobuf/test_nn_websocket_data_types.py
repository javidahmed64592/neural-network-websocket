"""Unit tests for the src/nn_websocket/protobuf/neural_network_types.py module."""

import pytest

from nn_websocket.protobuf.compiled.NNWebsocketData_pb2 import (
    ConfigData,
    FitnessApproachConfig,
    GeneticAlgorithmConfig,
    NeuralNetworkConfig,
    NeuroevolutionConfig,
)
from nn_websocket.protobuf.nn_websocket_data_types import (
    ConfigDataType,
    FitnessApproachConfigType,
    GeneticAlgorithmConfigType,
    NeuralNetworkConfigType,
    NeuroevolutionConfigType,
)


class TestConfigDataType:
    """Test cases for ConfigDataType conversions and serialization."""

    def test_from_protobuf(self, configuration_neuroevolution: ConfigData, configuration_fitness: ConfigData) -> None:
        """Test creating ConfigDataType from ConfigData Protobufs."""
        config_data_neuroevolution = ConfigDataType.from_protobuf(configuration_neuroevolution)
        config_data_fitness = ConfigDataType.from_protobuf(configuration_fitness)

        assert isinstance(config_data_neuroevolution.neuroevolution, NeuroevolutionConfigType)
        assert isinstance(config_data_fitness.fitness_approach, FitnessApproachConfigType)

    def test_to_protobuf(
        self, configuration_data_neuroevolution: ConfigDataType, configuration_data_fitness: ConfigDataType
    ) -> None:
        """Test converting ConfigDataType to ConfigData Protobufs."""
        protobuf_neuroevolution = ConfigDataType.to_protobuf(configuration_data_neuroevolution)
        protobuf_fitness = ConfigDataType.to_protobuf(configuration_data_fitness)

        assert isinstance(protobuf_neuroevolution.neuroevolution, NeuroevolutionConfig)
        assert isinstance(protobuf_fitness.fitness_approach, FitnessApproachConfig)

    def test_to_bytes(
        self, configuration_data_neuroevolution: ConfigDataType, configuration_data_fitness: ConfigDataType
    ) -> None:
        """Test serializing ConfigDataType to bytes."""
        assert isinstance(ConfigDataType.to_bytes(configuration_data_neuroevolution), bytes)
        assert isinstance(ConfigDataType.to_bytes(configuration_data_fitness), bytes)

    def test_from_bytes(
        self, configuration_data_neuroevolution: ConfigDataType, configuration_data_fitness: ConfigDataType
    ) -> None:
        """Test deserializing bytes to ConfigDataType."""
        msg_bytes_neuroevolution = ConfigDataType.to_bytes(configuration_data_neuroevolution)
        msg_bytes_fitness = ConfigDataType.to_bytes(configuration_data_fitness)

        result_neuroevolution = ConfigDataType.from_bytes(msg_bytes_neuroevolution)
        result_fitness = ConfigDataType.from_bytes(msg_bytes_fitness)

        assert isinstance(result_neuroevolution.neuroevolution, NeuroevolutionConfigType)
        assert isinstance(result_fitness.fitness_approach, FitnessApproachConfigType)


class TestNeuralNetworkDataType:
    """Test cases for NeuralNetworkConfigType conversions and serialization."""

    def test_from_protobuf(self, neural_network_config: NeuralNetworkConfig) -> None:
        """Test creating NeuralNetworkConfigType from NeuralNetworkConfig Protobuf."""
        neural_network_data_type = NeuralNetworkConfigType.from_protobuf(neural_network_config)

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
        assert neural_network_data_type.optimizer.sgd.learning_rate == pytest.approx(
            neural_network_config.optimizer.sgd.learning_rate
        )
        assert neural_network_data_type.optimizer.learning_rate_scheduler.decay_rate == pytest.approx(
            neural_network_config.optimizer.learning_rate_scheduler.decay_rate
        )
        assert (
            neural_network_data_type.optimizer.learning_rate_scheduler.decay_steps
            == neural_network_config.optimizer.learning_rate_scheduler.decay_steps
        )
        assert (
            neural_network_data_type.optimizer.learning_rate_scheduler.method
            == neural_network_config.optimizer.learning_rate_scheduler.method
        )

    def test_to_protobuf(self, neural_network_config_data: NeuralNetworkConfigType) -> None:
        """Test converting NeuralNetworkConfigType to NeuralNetworkConfig Protobuf."""
        protobuf_data = NeuralNetworkConfigType.to_protobuf(neural_network_config_data)

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
        assert protobuf_data.optimizer.sgd.learning_rate == pytest.approx(
            neural_network_config_data.optimizer.sgd.learning_rate
        )
        assert protobuf_data.optimizer.learning_rate_scheduler.decay_rate == pytest.approx(
            neural_network_config_data.optimizer.learning_rate_scheduler.decay_rate
        )
        assert protobuf_data.optimizer.learning_rate_scheduler.decay_steps == pytest.approx(
            neural_network_config_data.optimizer.learning_rate_scheduler.decay_steps
        )
        assert protobuf_data.optimizer.learning_rate_scheduler.method == pytest.approx(
            neural_network_config_data.optimizer.learning_rate_scheduler.method
        )

    def test_to_bytes(self, neural_network_config_data: NeuralNetworkConfigType) -> None:
        """Test serializing NeuralNetworkConfigType to bytes."""
        assert isinstance(NeuralNetworkConfigType.to_bytes(neural_network_config_data), bytes)

    def test_from_bytes(self, neural_network_config_data: NeuralNetworkConfigType) -> None:
        """Test deserializing bytes to NeuralNetworkConfigType."""
        msg_bytes = NeuralNetworkConfigType.to_bytes(neural_network_config_data)
        result = NeuralNetworkConfigType.from_bytes(msg_bytes)

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
        assert result.optimizer.sgd.learning_rate == pytest.approx(
            neural_network_config_data.optimizer.sgd.learning_rate
        )
        assert result.optimizer.learning_rate_scheduler.decay_rate == pytest.approx(
            neural_network_config_data.optimizer.learning_rate_scheduler.decay_rate
        )
        assert result.optimizer.learning_rate_scheduler.decay_steps == pytest.approx(
            neural_network_config_data.optimizer.learning_rate_scheduler.decay_steps
        )
        assert result.optimizer.learning_rate_scheduler.method == pytest.approx(
            neural_network_config_data.optimizer.learning_rate_scheduler.method
        )


class TestGeneticAlgorithmConfigData:
    """Test cases for GeneticAlgorithmConfigType conversions and serialization."""

    def test_from_protobuf(self, genetic_algorithm_config: GeneticAlgorithmConfig) -> None:
        """Test creating GeneticAlgorithmConfigType from GeneticAlgorithmConfig Protobuf."""
        ga_data = GeneticAlgorithmConfigType.from_protobuf(genetic_algorithm_config)

        assert ga_data.population_size == genetic_algorithm_config.population_size
        assert ga_data.mutation_rate == pytest.approx(genetic_algorithm_config.mutation_rate)

    def test_to_protobuf(self, genetic_algorithm_config_data: GeneticAlgorithmConfigType) -> None:
        """Test converting GeneticAlgorithmConfigType to GeneticAlgorithmConfig Protobuf."""
        protobuf_data = GeneticAlgorithmConfigType.to_protobuf(genetic_algorithm_config_data)

        assert protobuf_data.population_size == genetic_algorithm_config_data.population_size
        assert protobuf_data.mutation_rate == pytest.approx(genetic_algorithm_config_data.mutation_rate)

    def test_to_bytes(self, genetic_algorithm_config_data: GeneticAlgorithmConfigType) -> None:
        """Test serializing GeneticAlgorithmConfigType to bytes."""
        assert isinstance(GeneticAlgorithmConfigType.to_bytes(genetic_algorithm_config_data), bytes)

    def test_from_bytes(self, genetic_algorithm_config_data: GeneticAlgorithmConfigType) -> None:
        """Test deserializing bytes to GeneticAlgorithmConfigType."""
        msg_bytes = GeneticAlgorithmConfigType.to_bytes(genetic_algorithm_config_data)
        result = GeneticAlgorithmConfigType.from_bytes(msg_bytes)

        assert result.population_size == genetic_algorithm_config_data.population_size
        assert result.mutation_rate == pytest.approx(genetic_algorithm_config_data.mutation_rate)


class TestNeuroevolutionConfigData:
    """Test cases for NeuroevolutionConfigType conversions and serialization."""

    def test_from_protobuf(self, neuroevolution_config: NeuroevolutionConfig) -> None:
        """Test creating NeuroevolutionConfigType from NeuroevolutionConfig Protobuf."""
        config_data = NeuroevolutionConfigType.from_protobuf(neuroevolution_config)

        assert isinstance(config_data.neural_network, NeuralNetworkConfigType)
        assert isinstance(config_data.genetic_algorithm, GeneticAlgorithmConfigType)

    def test_to_protobuf(self, neuroevolution_config_data: NeuroevolutionConfigType) -> None:
        """Test converting NeuroevolutionConfigType to NeuroevolutionConfig Protobuf."""
        protobuf_data = NeuroevolutionConfigType.to_protobuf(neuroevolution_config_data)

        assert isinstance(protobuf_data.neural_network, NeuralNetworkConfig)
        assert isinstance(protobuf_data.genetic_algorithm, GeneticAlgorithmConfig)

    def test_to_bytes(self, neuroevolution_config_data: NeuroevolutionConfigType) -> None:
        """Test serializing NeuroevolutionConfigType to bytes."""
        assert isinstance(NeuroevolutionConfigType.to_bytes(neuroevolution_config_data), bytes)

    def test_from_bytes(self, neuroevolution_config_data: NeuroevolutionConfigType) -> None:
        """Test deserializing bytes to NeuroevolutionConfigType."""
        msg_bytes = NeuroevolutionConfigType.to_bytes(neuroevolution_config_data)
        result = NeuroevolutionConfigType.from_bytes(msg_bytes)

        assert isinstance(result.neural_network, NeuralNetworkConfigType)
        assert isinstance(result.genetic_algorithm, GeneticAlgorithmConfigType)


class TestFitnessApproachConfigData:
    """Test cases for FitnessApproachConfigType conversions and serialization."""

    def test_from_protobuf(self, fitness_approach_config: FitnessApproachConfig) -> None:
        """Test creating FitnessApproachConfigType from FitnessApproachConfig Protobuf."""
        config_data = FitnessApproachConfigType.from_protobuf(fitness_approach_config)

        assert isinstance(config_data.neural_network, NeuralNetworkConfigType)

    def test_to_protobuf(self, fitness_approach_config_data: FitnessApproachConfigType) -> None:
        """Test converting FitnessApproachConfigType to FitnessApproachConfig Protobuf."""
        protobuf_data = FitnessApproachConfigType.to_protobuf(fitness_approach_config_data)

        assert isinstance(protobuf_data.neural_network, NeuralNetworkConfig)

    def test_to_bytes(self, fitness_approach_config_data: FitnessApproachConfigType) -> None:
        """Test serializing FitnessApproachConfigType to bytes."""
        assert isinstance(FitnessApproachConfigType.to_bytes(fitness_approach_config_data), bytes)

    def test_from_bytes(self, fitness_approach_config_data: FitnessApproachConfigType) -> None:
        """Test deserializing bytes to FitnessApproachConfigType."""
        msg_bytes = FitnessApproachConfigType.to_bytes(fitness_approach_config_data)
        result = FitnessApproachConfigType.from_bytes(msg_bytes)

        assert isinstance(result.neural_network, NeuralNetworkConfigType)
