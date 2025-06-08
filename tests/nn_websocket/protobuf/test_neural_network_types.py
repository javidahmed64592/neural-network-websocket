import pytest
from neural_network.math.activation_functions import LinearActivation, ReluActivation, SigmoidActivation, TanhActivation

from nn_websocket.protobuf.compiled.NeuralNetwork_pb2 import ActivationFunctionData, NeuralNetworkConfig
from nn_websocket.protobuf.neural_network_types import ActivationFunctionEnum, NeuralNetworkConfigData


class TestActivationFunctionEnum:
    def test_get_class(self) -> None:
        assert ActivationFunctionEnum.LINEAR.get_class() == LinearActivation
        assert ActivationFunctionEnum.RELU.get_class() == ReluActivation
        assert ActivationFunctionEnum.SIGMOID.get_class() == SigmoidActivation
        assert ActivationFunctionEnum.TANH.get_class() == TanhActivation

    def test_from_class(self) -> None:
        assert ActivationFunctionEnum.from_class(LinearActivation) == ActivationFunctionEnum.LINEAR
        assert ActivationFunctionEnum.from_class(ReluActivation) == ActivationFunctionEnum.RELU
        assert ActivationFunctionEnum.from_class(SigmoidActivation) == ActivationFunctionEnum.SIGMOID
        assert ActivationFunctionEnum.from_class(TanhActivation) == ActivationFunctionEnum.TANH

    def test_from_protobuf(self) -> None:
        assert ActivationFunctionEnum.from_protobuf(ActivationFunctionData.LINEAR) == ActivationFunctionEnum.LINEAR
        assert ActivationFunctionEnum.from_protobuf(ActivationFunctionData.RELU) == ActivationFunctionEnum.RELU
        assert ActivationFunctionEnum.from_protobuf(ActivationFunctionData.SIGMOID) == ActivationFunctionEnum.SIGMOID
        assert ActivationFunctionEnum.from_protobuf(ActivationFunctionData.TANH) == ActivationFunctionEnum.TANH

    def test_to_protobuf(self) -> None:
        assert ActivationFunctionEnum.to_protobuf(ActivationFunctionEnum.LINEAR) == ActivationFunctionData.LINEAR
        assert ActivationFunctionEnum.to_protobuf(ActivationFunctionEnum.RELU) == ActivationFunctionData.RELU
        assert ActivationFunctionEnum.to_protobuf(ActivationFunctionEnum.SIGMOID) == ActivationFunctionData.SIGMOID
        assert ActivationFunctionEnum.to_protobuf(ActivationFunctionEnum.TANH) == ActivationFunctionData.TANH


class TestNeuralNetworkDataType:
    @pytest.fixture
    def neural_network_config(self) -> NeuralNetworkConfig:
        test_num_inputs = 2
        test_num_outputs = 1
        test_hidden_layer_sizes = [3]
        test_weights_range = (-1, 1)
        test_bias_range = (-1, 1)
        test_input_activation = ActivationFunctionData.RELU
        test_hidden_activation = ActivationFunctionData.SIGMOID
        test_output_activation = ActivationFunctionData.LINEAR
        test_learning_rate = 0.01

        return NeuralNetworkConfig(
            num_inputs=test_num_inputs,
            num_outputs=test_num_outputs,
            hidden_layer_sizes=test_hidden_layer_sizes,
            weights_min=test_weights_range[0],
            weights_max=test_weights_range[1],
            bias_min=test_bias_range[0],
            bias_max=test_bias_range[1],
            input_activation=test_input_activation,
            hidden_activation=test_hidden_activation,
            output_activation=test_output_activation,
            learning_rate=test_learning_rate,
        )

    @pytest.fixture
    def neural_network_config_data(self, neural_network_config: NeuralNetworkConfig) -> NeuralNetworkConfigData:
        return NeuralNetworkConfigData(
            num_inputs=neural_network_config.num_inputs,
            num_outputs=neural_network_config.num_outputs,
            hidden_layer_sizes=neural_network_config.hidden_layer_sizes,
            weights_min=neural_network_config.weights_min,
            weights_max=neural_network_config.weights_max,
            bias_min=neural_network_config.bias_min,
            bias_max=neural_network_config.bias_max,
            input_activation=ActivationFunctionEnum.from_protobuf(neural_network_config.input_activation),
            hidden_activation=ActivationFunctionEnum.from_protobuf(neural_network_config.hidden_activation),
            output_activation=ActivationFunctionEnum.from_protobuf(neural_network_config.output_activation),
            learning_rate=neural_network_config.learning_rate,
        )

    def test_from_protobuf(self, neural_network_config: NeuralNetworkConfig) -> None:
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
        assert isinstance(NeuralNetworkConfigData.to_bytes(neural_network_config_data), bytes)

    def test_from_bytes(self, neural_network_config_data: NeuralNetworkConfigData) -> None:
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
