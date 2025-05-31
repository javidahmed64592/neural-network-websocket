import pytest
from google.protobuf.message import DecodeError
from neural_network.math.activation_functions import LinearActivation, ReluActivation, SigmoidActivation

from nn_websocket.protobuf.compiled.NeuralNetwork_pb2 import ActivationFunction
from nn_websocket.protobuf.proto_types import (
    ActionData,
    ActivationFunctionEnum,
    ConfigurationData,
    GeneticAlgorithmConfigData,
    NeuralNetworkConfigData,
    ObservationData,
    PopulationFitnessData,
)


# neural_network.proto
class TestConfigurationData:
    def test_from_protobuf(self, config_data: ConfigurationData) -> None:
        assert isinstance(ConfigurationData.from_protobuf(config_data.to_protobuf()), ConfigurationData)

    def test_to_protobuf(self, config_data: ConfigurationData) -> None:
        proto_config = ConfigurationData.to_protobuf(config_data)

        assert proto_config.genetic_algorithm.population_size == config_data.genetic_algorithm.population_size
        assert proto_config.genetic_algorithm.mutation_rate == pytest.approx(
            config_data.genetic_algorithm.mutation_rate
        )

        assert proto_config.neural_network.num_inputs == config_data.neural_network.num_inputs
        assert proto_config.neural_network.num_outputs == config_data.neural_network.num_outputs
        assert proto_config.neural_network.hidden_layer_sizes == config_data.neural_network.hidden_layer_sizes
        assert proto_config.neural_network.weights_min == pytest.approx(config_data.neural_network.weights_min)
        assert proto_config.neural_network.weights_max == pytest.approx(config_data.neural_network.weights_max)
        assert proto_config.neural_network.bias_min == pytest.approx(config_data.neural_network.bias_min)
        assert proto_config.neural_network.bias_max == pytest.approx(config_data.neural_network.bias_max)
        assert proto_config.neural_network.input_activation == ActivationFunctionEnum.to_protobuf(
            config_data.neural_network.input_activation
        )
        assert proto_config.neural_network.hidden_activation == ActivationFunctionEnum.to_protobuf(
            config_data.neural_network.hidden_activation
        )
        assert proto_config.neural_network.output_activation == ActivationFunctionEnum.to_protobuf(
            config_data.neural_network.output_activation
        )


class TestGeneticAlgorithmConfigData:
    def test_to_bytes(self, ga_config_data: GeneticAlgorithmConfigData) -> None:
        assert isinstance(GeneticAlgorithmConfigData.to_bytes(ga_config_data), bytes)

    def test_from_protobuf(self, ga_config_data: GeneticAlgorithmConfigData) -> None:
        msg_bytes = GeneticAlgorithmConfigData.to_bytes(ga_config_data)
        result = GeneticAlgorithmConfigData.from_bytes(msg_bytes)

        assert result.population_size == ga_config_data.population_size
        assert result.mutation_rate == pytest.approx(ga_config_data.mutation_rate)

    def test_from_bytes_invalid(self) -> None:
        with pytest.raises(DecodeError):
            GeneticAlgorithmConfigData.from_bytes(b"invalid data")


class TestNeuralNetworkConfigData:
    def test_to_bytes(self, nn_config_data: NeuralNetworkConfigData) -> None:
        assert isinstance(NeuralNetworkConfigData.to_bytes(nn_config_data), bytes)

    def test_from_protobuf(self, nn_config_data: NeuralNetworkConfigData) -> None:
        msg_bytes = NeuralNetworkConfigData.to_bytes(nn_config_data)
        result = NeuralNetworkConfigData.from_bytes(msg_bytes)

        assert result.num_inputs == nn_config_data.num_inputs
        assert result.num_outputs == nn_config_data.num_outputs
        assert result.hidden_layer_sizes == nn_config_data.hidden_layer_sizes
        assert result.weights_min == pytest.approx(nn_config_data.weights_min)
        assert result.weights_max == pytest.approx(nn_config_data.weights_max)
        assert result.bias_min == pytest.approx(nn_config_data.bias_min)
        assert result.bias_max == pytest.approx(nn_config_data.bias_max)
        assert result.input_activation == nn_config_data.input_activation
        assert result.hidden_activation == nn_config_data.hidden_activation
        assert result.output_activation == nn_config_data.output_activation

    def test_from_bytes_invalid(self) -> None:
        with pytest.raises(DecodeError):
            NeuralNetworkConfigData.from_bytes(b"invalid data")


class TestActivationFunctionEnum:
    def test_get_class(self) -> None:
        assert ActivationFunctionEnum.LINEAR.get_class() == LinearActivation
        assert ActivationFunctionEnum.RELU.get_class() == ReluActivation
        assert ActivationFunctionEnum.SIGMOID.get_class() == SigmoidActivation

    def test_from_protobuf(self) -> None:
        assert ActivationFunctionEnum.from_protobuf(ActivationFunction.LINEAR) == ActivationFunctionEnum.LINEAR
        assert ActivationFunctionEnum.from_protobuf(ActivationFunction.RELU) == ActivationFunctionEnum.RELU
        assert ActivationFunctionEnum.from_protobuf(ActivationFunction.SIGMOID) == ActivationFunctionEnum.SIGMOID

    def test_to_protobuf(self) -> None:
        assert ActivationFunctionEnum.to_protobuf(ActivationFunctionEnum.LINEAR) == ActivationFunction.LINEAR
        assert ActivationFunctionEnum.to_protobuf(ActivationFunctionEnum.RELU) == ActivationFunction.RELU
        assert ActivationFunctionEnum.to_protobuf(ActivationFunctionEnum.SIGMOID) == ActivationFunction.SIGMOID

    def test_end_to_end_conversion(self) -> None:
        for enum_value in ActivationFunctionEnum:
            proto_enum = ActivationFunctionEnum.to_protobuf(enum_value)
            converted_enum = ActivationFunctionEnum.from_protobuf(proto_enum)

            assert converted_enum == enum_value
            assert converted_enum.get_class() == enum_value.get_class()


# frame_data.proto
class TestObservationData:
    def test_to_bytes(self, observation_data: ObservationData) -> None:
        assert isinstance(ObservationData.to_bytes(observation_data), bytes)

    def test_from_bytes(self, observation_data: ObservationData) -> None:
        msg_bytes = ObservationData.to_bytes(observation_data)
        result = ObservationData.from_bytes(msg_bytes)

        assert result.inputs == pytest.approx(observation_data.inputs)

    def test_from_bytes_invalid(self) -> None:
        with pytest.raises(DecodeError):
            ObservationData.from_bytes(b"invalid data")


class TestActionData:
    def test_to_bytes(self, action_data: ActionData) -> None:
        assert isinstance(ActionData.to_bytes(action_data), bytes)

    def test_from_bytes(self, action_data: ActionData) -> None:
        msg_bytes = ActionData.to_bytes(action_data)
        result = ActionData.from_bytes(msg_bytes)

        assert result.outputs == pytest.approx(action_data.outputs)

    def test_from_bytes_invalid(self) -> None:
        with pytest.raises(DecodeError):
            ActionData.from_bytes(b"invalid data")


class TestPopulationFitnessData:
    def test_to_bytes(self, population_fitness_data: PopulationFitnessData) -> None:
        assert isinstance(PopulationFitnessData.to_bytes(population_fitness_data), bytes)

    def test_from_bytes(self, population_fitness_data: PopulationFitnessData) -> None:
        msg_bytes = PopulationFitnessData.to_bytes(population_fitness_data)
        result = PopulationFitnessData.from_bytes(msg_bytes)

        assert result.fitness == pytest.approx(population_fitness_data.fitness)

    def test_from_bytes_invalid(self) -> None:
        with pytest.raises(DecodeError):
            PopulationFitnessData.from_bytes(b"invalid data")
