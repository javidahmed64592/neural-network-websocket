from __future__ import annotations

from enum import IntEnum

from neural_network.math.activation_functions import LinearActivation, ReluActivation, SigmoidActivation, TanhActivation
from pydantic.dataclasses import dataclass

from nn_websocket.protobuf.compiled.NeuralNetwork_pb2 import (
    ActivationFunctionData,
    NeuralNetworkConfig,
)


class ActivationFunctionEnum(IntEnum):
    LINEAR = 0
    RELU = 1
    SIGMOID = 2
    TANH = 3

    @property
    def map(self) -> dict[ActivationFunctionEnum, type[ActivationFunctionData]]:
        """Maps the enum to the corresponding activation function."""
        return {
            ActivationFunctionEnum.LINEAR: LinearActivation,
            ActivationFunctionEnum.RELU: ReluActivation,
            ActivationFunctionEnum.SIGMOID: SigmoidActivation,
            ActivationFunctionEnum.TANH: TanhActivation,
        }

    def get_class(self) -> type:
        """Returns the corresponding activation function class."""
        return self.map[self]

    @classmethod
    def from_class(cls, activation_function: type[ActivationFunctionData]) -> ActivationFunctionEnum:
        """Maps an ActivationFunctionData class to ActivationFunctionEnum."""
        reverse_map = {v: k for k, v in cls.LINEAR.map.items()}
        return reverse_map[activation_function]

    @classmethod
    def from_protobuf(cls, proto_enum_value: ActivationFunctionData) -> ActivationFunctionEnum:
        """Maps a Protobuf ActivationFunctionData value to ActivationFunctionEnum."""
        return cls(proto_enum_value)

    @staticmethod
    def to_protobuf(enum_value: ActivationFunctionEnum) -> ActivationFunctionData:
        """Maps an ActivationFunctionEnum value to Protobuf ActivationFunctionData."""
        return ActivationFunctionData.Value(enum_value.name)  # type: ignore[no-any-return]


@dataclass
class NeuralNetworkConfigData:
    """Data class to hold neural network configuration."""

    num_inputs: int
    num_outputs: int
    hidden_layer_sizes: list[int]
    weights_min: float
    weights_max: float
    bias_min: float
    bias_max: float
    input_activation: ActivationFunctionEnum
    hidden_activation: ActivationFunctionEnum
    output_activation: ActivationFunctionEnum
    learning_rate: float = 0.01

    @classmethod
    def from_protobuf(cls, config: NeuralNetworkConfig) -> NeuralNetworkConfigData:
        """Creates a NeuralNetworkConfigData instance from Protobuf."""
        return cls(
            num_inputs=config.num_inputs,
            num_outputs=config.num_outputs,
            hidden_layer_sizes=list(config.hidden_layer_sizes),
            weights_min=config.weights_min,
            weights_max=config.weights_max,
            bias_min=config.bias_min,
            bias_max=config.bias_max,
            input_activation=ActivationFunctionEnum.from_protobuf(config.input_activation),
            hidden_activation=ActivationFunctionEnum.from_protobuf(config.hidden_activation),
            output_activation=ActivationFunctionEnum.from_protobuf(config.output_activation),
            learning_rate=config.learning_rate,
        )

    @staticmethod
    def to_protobuf(config_data: NeuralNetworkConfigData) -> NeuralNetworkConfig:
        """Converts NeuralNetworkConfigData to Protobuf."""
        return NeuralNetworkConfig(
            num_inputs=config_data.num_inputs,
            num_outputs=config_data.num_outputs,
            hidden_layer_sizes=config_data.hidden_layer_sizes,
            weights_min=config_data.weights_min,
            weights_max=config_data.weights_max,
            bias_min=config_data.bias_min,
            bias_max=config_data.bias_max,
            input_activation=ActivationFunctionEnum.to_protobuf(config_data.input_activation),
            hidden_activation=ActivationFunctionEnum.to_protobuf(config_data.hidden_activation),
            output_activation=ActivationFunctionEnum.to_protobuf(config_data.output_activation),
            learning_rate=config_data.learning_rate,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> NeuralNetworkConfigData:
        """Creates a NeuralNetworkConfigData instance from Protobuf bytes."""
        config = NeuralNetworkConfig()
        config.ParseFromString(data)
        return cls.from_protobuf(config)

    @staticmethod
    def to_bytes(config_data: NeuralNetworkConfigData) -> bytes:
        """Converts NeuralNetworkConfigData to Protobuf bytes."""
        config = NeuralNetworkConfigData.to_protobuf(config_data)
        return config.SerializeToString()
