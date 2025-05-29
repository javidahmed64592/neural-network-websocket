from __future__ import annotations

from enum import IntEnum

from neural_network.math.activation_functions import LinearActivation, ReluActivation, SigmoidActivation
from pydantic.dataclasses import dataclass

from nn_websocket.protobuf.compiled.frame_data_pb2 import Observation, ObservationBatch, Output, OutputBatch
from nn_websocket.protobuf.compiled.neural_network_pb2 import ActivationFunction, NeuralNetworkConfig


# neural_network.proto
@dataclass
class NeuralNetworkConfigData:
    """Data class to hold neural network configuration."""

    num_networks: int
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

    @classmethod
    def from_bytes(cls, data: bytes) -> NeuralNetworkConfigData:
        """Creates a NeuralNetworkConfigData instance from Protobuf bytes."""
        config = NeuralNetworkConfig()
        config.ParseFromString(data)

        return cls(
            num_networks=config.num_networks,
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
        )

    @staticmethod
    def to_bytes(config_data: NeuralNetworkConfigData) -> bytes:
        """Converts NeuralNetworkConfigData to Protobuf bytes."""
        config = NeuralNetworkConfig(
            num_networks=config_data.num_networks,
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
        )
        return config.SerializeToString()


class ActivationFunctionEnum(IntEnum):
    LINEAR = 0
    RELU = 1
    SIGMOID = 2

    def get_class(self) -> type:
        """Returns the corresponding activation function class."""
        _map: dict[ActivationFunctionEnum, type] = {
            ActivationFunctionEnum.LINEAR: LinearActivation,
            ActivationFunctionEnum.RELU: ReluActivation,
            ActivationFunctionEnum.SIGMOID: SigmoidActivation,
        }
        return _map[self]

    @staticmethod
    def from_protobuf(proto_enum_value: ActivationFunction) -> ActivationFunctionEnum:
        """Maps a Protobuf ActivationFunction value to ActivationFunctionEnum."""
        return ActivationFunctionEnum(proto_enum_value)

    @staticmethod
    def to_protobuf(enum_value: ActivationFunctionEnum) -> ActivationFunction:
        """Maps an ActivationFunctionEnum value to Protobuf ActivationFunction."""
        return ActivationFunction.Value(enum_value.name)  # type: ignore


# frame_data.proto
@dataclass
class ObservationData:
    """Data class to hold observation data."""

    agent_id: int
    inputs: list[float]

    @classmethod
    def from_bytes(cls, data: bytes) -> ObservationData:
        """Creates an ObservationData instance from Protobuf bytes."""
        observation = Observation()
        observation.ParseFromString(data)

        return cls(agent_id=observation.agent_id, inputs=list(observation.inputs))

    @staticmethod
    def to_bytes(observation_data: ObservationData) -> bytes:
        """Converts ObservationData to Protobuf bytes."""
        observation = Observation(agent_id=observation_data.agent_id, inputs=observation_data.inputs)
        return observation.SerializeToString()


@dataclass
class ObservationBatchData:
    """Data class to hold a batch of observation data."""

    observations: list[ObservationData]

    @classmethod
    def from_bytes(cls, data: bytes) -> ObservationBatchData:
        """Creates an ObservationBatchData instance from Protobuf bytes."""
        observation_batch = ObservationBatch()
        observation_batch.ParseFromString(data)

        return cls(
            observations=[ObservationData.from_bytes(obs.SerializeToString()) for obs in observation_batch.observations]
        )

    @staticmethod
    def to_bytes(observation_batch_data: ObservationBatchData) -> bytes:
        """Converts ObservationBatchData to Protobuf bytes."""
        observation_batch = ObservationBatch(
            observations=[
                Observation().FromString(ObservationData.to_bytes(obs)) for obs in observation_batch_data.observations
            ]
        )
        return observation_batch.SerializeToString()


@dataclass
class OutputData:
    """Data class to hold output data."""

    agent_id: int
    actions: list[float]

    @classmethod
    def from_bytes(cls, data: bytes) -> OutputData:
        """Creates an OutputData instance from Protobuf bytes."""
        output = Output()
        output.ParseFromString(data)

        return cls(agent_id=output.agent_id, actions=list(output.actions))

    @staticmethod
    def to_bytes(output_data: OutputData) -> bytes:
        """Converts OutputData to Protobuf bytes."""
        output = Output(agent_id=output_data.agent_id, actions=output_data.actions)
        return output.SerializeToString()


@dataclass
class OutputBatchData:
    """Data class to hold a batch of output data."""

    outputs: list[OutputData]

    @classmethod
    def from_bytes(cls, data: bytes) -> OutputBatchData:
        """Creates an OutputBatchData instance from Protobuf bytes."""
        output_batch = OutputBatch()
        output_batch.ParseFromString(data)

        return cls(outputs=[OutputData.from_bytes(out.SerializeToString()) for out in output_batch.outputs])

    @staticmethod
    def to_bytes(output_batch_data: OutputBatchData) -> bytes:
        """Converts OutputBatchData to Protobuf bytes."""
        output_batch = OutputBatch(
            outputs=[Output().FromString(OutputData.to_bytes(out)) for out in output_batch_data.outputs]
        )
        return output_batch.SerializeToString()
