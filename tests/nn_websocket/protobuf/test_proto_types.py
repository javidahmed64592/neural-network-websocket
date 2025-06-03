import pytest
from neural_network.math.activation_functions import LinearActivation, ReluActivation, SigmoidActivation

from nn_websocket.protobuf.compiled.NeuralNetwork_pb2 import ActivationFunction
from nn_websocket.protobuf.proto_types import (
    ActionData,
    ActivationFunctionEnum,
    ConfigurationData,
    FitnessData,
    FrameRequestData,
    GeneticAlgorithmConfigData,
    NeuralNetworkConfigData,
    ObservationData,
)


# neural_network.proto
class TestConfigurationData:
    def test_to_bytes(self, configuration_data: ConfigurationData) -> None:
        assert isinstance(ConfigurationData.to_bytes(configuration_data), bytes)

    def test_from_bytes(self, configuration_data: ConfigurationData) -> None:
        msg_bytes = ConfigurationData.to_bytes(configuration_data)
        result = ConfigurationData.from_bytes(msg_bytes)

        assert isinstance(result.neural_network, NeuralNetworkConfigData)
        assert isinstance(result.genetic_algorithm, GeneticAlgorithmConfigData)


class TestNeuralNetworkConfigData:
    def test_to_bytes(self, nn_config_data: NeuralNetworkConfigData) -> None:
        assert isinstance(NeuralNetworkConfigData.to_bytes(nn_config_data), bytes)

    def test_from_bytes(self, nn_config_data: NeuralNetworkConfigData) -> None:
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


class TestGeneticAlgorithmConfigData:
    def test_to_bytes(self, ga_config_data: GeneticAlgorithmConfigData) -> None:
        assert isinstance(GeneticAlgorithmConfigData.to_bytes(ga_config_data), bytes)

    def test_from_bytes(self, ga_config_data: GeneticAlgorithmConfigData) -> None:
        msg_bytes = GeneticAlgorithmConfigData.to_bytes(ga_config_data)
        result = GeneticAlgorithmConfigData.from_bytes(msg_bytes)

        assert result.population_size == ga_config_data.population_size
        assert result.mutation_rate == pytest.approx(ga_config_data.mutation_rate)


# frame_data.proto
class TestFrameRequestData:
    def test_to_bytes_with_observation(self, frame_request_data_observation: FrameRequestData) -> None:
        """Test serializing a FrameRequestData with observation data."""
        assert isinstance(FrameRequestData.to_bytes(frame_request_data_observation), bytes)

    def test_to_bytes_with_population_fitness(self, frame_request_data_population: FrameRequestData) -> None:
        """Test serializing a FrameRequestData with population fitness data."""
        assert isinstance(FrameRequestData.to_bytes(frame_request_data_population), bytes)

    def test_from_bytes_with_observation(self, frame_request_data_observation: FrameRequestData) -> None:
        """Test deserializing a FrameRequestData with observation data."""
        assert frame_request_data_observation.observation is not None

        msg_bytes = FrameRequestData.to_bytes(frame_request_data_observation)
        result = FrameRequestData.from_bytes(msg_bytes)

        assert isinstance(result.observation, ObservationData)
        assert result.observation.inputs == pytest.approx(frame_request_data_observation.observation.inputs)

    def test_from_bytes_with_population_fitness(self, frame_request_data_population: FrameRequestData) -> None:
        """Test deserializing a FrameRequestData with population fitness data."""
        assert frame_request_data_population.population_fitness is not None

        msg_bytes = FrameRequestData.to_bytes(frame_request_data_population)
        result = FrameRequestData.from_bytes(msg_bytes)

        assert isinstance(result.population_fitness, FitnessData)
        assert result.population_fitness.values == pytest.approx(
            frame_request_data_population.population_fitness.values
        )


class TestObservationData:
    def test_to_bytes(self, observation_data: ObservationData) -> None:
        assert isinstance(ObservationData.to_bytes(observation_data), bytes)

    def test_from_bytes(self, observation_data: ObservationData) -> None:
        msg_bytes = ObservationData.to_bytes(observation_data)
        result = ObservationData.from_bytes(msg_bytes)

        assert result.inputs == pytest.approx(observation_data.inputs)


class TestActionData:
    def test_to_bytes(self, action_data: ActionData) -> None:
        assert isinstance(ActionData.to_bytes(action_data), bytes)

    def test_from_bytes(self, action_data: ActionData) -> None:
        msg_bytes = ActionData.to_bytes(action_data)
        result = ActionData.from_bytes(msg_bytes)

        assert result.outputs == pytest.approx(action_data.outputs)


class TestFitnessData:
    def test_to_bytes(self, fitness_data: FitnessData) -> None:
        assert isinstance(FitnessData.to_bytes(fitness_data), bytes)

    def test_from_bytes(self, fitness_data: FitnessData) -> None:
        msg_bytes = FitnessData.to_bytes(fitness_data)
        result = FitnessData.from_bytes(msg_bytes)

        assert result.values == pytest.approx(fitness_data.values)
