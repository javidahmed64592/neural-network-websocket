import pytest
from google.protobuf.message import DecodeError
from neural_network.math.activation_functions import LinearActivation, ReluActivation, SigmoidActivation

from nn_websocket.protobuf.neural_network_pb2 import ActivationFunction
from nn_websocket.protobuf.proto_types import ActivationFunctionEnum, NeuralNetworkConfigData


class TestNeuralNetworkConfigData:
    def test_to_bytes(self) -> None:
        config_data = NeuralNetworkConfigData(
            num_networks=2,
            num_inputs=3,
            num_outputs=1,
            hidden_layer_sizes=[4, 5],
            weights_min=-0.5,
            weights_max=0.5,
            bias_min=-0.1,
            bias_max=0.1,
            input_activation=ActivationFunctionEnum.RELU,
            hidden_activation=ActivationFunctionEnum.SIGMOID,
            output_activation=ActivationFunctionEnum.LINEAR,
        )

        assert isinstance(NeuralNetworkConfigData.to_bytes(config_data), bytes)

    def test_from_protobuf(self) -> None:
        config_data = NeuralNetworkConfigData(
            num_networks=2,
            num_inputs=3,
            num_outputs=1,
            hidden_layer_sizes=[4, 5],
            weights_min=-0.5,
            weights_max=0.5,
            bias_min=-0.1,
            bias_max=0.1,
            input_activation=ActivationFunctionEnum.RELU,
            hidden_activation=ActivationFunctionEnum.SIGMOID,
            output_activation=ActivationFunctionEnum.LINEAR,
        )

        msg_bytes = NeuralNetworkConfigData.to_bytes(config_data)
        result = NeuralNetworkConfigData.from_bytes(msg_bytes)

        assert result.num_networks == config_data.num_networks
        assert result.num_inputs == config_data.num_inputs
        assert result.num_outputs == config_data.num_outputs
        assert result.hidden_layer_sizes == config_data.hidden_layer_sizes
        assert result.weights_min == pytest.approx(config_data.weights_min)
        assert result.weights_max == pytest.approx(config_data.weights_max)
        assert result.bias_min == pytest.approx(config_data.bias_min)
        assert result.bias_max == pytest.approx(config_data.bias_max)
        assert result.input_activation == config_data.input_activation
        assert result.hidden_activation == config_data.hidden_activation
        assert result.output_activation == config_data.output_activation

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
