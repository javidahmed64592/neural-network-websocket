from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ActivationFunctionData(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    LINEAR: _ClassVar[ActivationFunctionData]
    RELU: _ClassVar[ActivationFunctionData]
    SIGMOID: _ClassVar[ActivationFunctionData]
    TANH: _ClassVar[ActivationFunctionData]
LINEAR: ActivationFunctionData
RELU: ActivationFunctionData
SIGMOID: ActivationFunctionData
TANH: ActivationFunctionData

class NeuralNetworkConfig(_message.Message):
    __slots__ = ("num_inputs", "num_outputs", "hidden_layer_sizes", "weights_min", "weights_max", "bias_min", "bias_max", "input_activation", "hidden_activation", "output_activation", "learning_rate")
    NUM_INPUTS_FIELD_NUMBER: _ClassVar[int]
    NUM_OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    HIDDEN_LAYER_SIZES_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_MIN_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_MAX_FIELD_NUMBER: _ClassVar[int]
    BIAS_MIN_FIELD_NUMBER: _ClassVar[int]
    BIAS_MAX_FIELD_NUMBER: _ClassVar[int]
    INPUT_ACTIVATION_FIELD_NUMBER: _ClassVar[int]
    HIDDEN_ACTIVATION_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_ACTIVATION_FIELD_NUMBER: _ClassVar[int]
    LEARNING_RATE_FIELD_NUMBER: _ClassVar[int]
    num_inputs: int
    num_outputs: int
    hidden_layer_sizes: _containers.RepeatedScalarFieldContainer[int]
    weights_min: float
    weights_max: float
    bias_min: float
    bias_max: float
    input_activation: ActivationFunctionData
    hidden_activation: ActivationFunctionData
    output_activation: ActivationFunctionData
    learning_rate: float
    def __init__(self, num_inputs: _Optional[int] = ..., num_outputs: _Optional[int] = ..., hidden_layer_sizes: _Optional[_Iterable[int]] = ..., weights_min: _Optional[float] = ..., weights_max: _Optional[float] = ..., bias_min: _Optional[float] = ..., bias_max: _Optional[float] = ..., input_activation: _Optional[_Union[ActivationFunctionData, str]] = ..., hidden_activation: _Optional[_Union[ActivationFunctionData, str]] = ..., output_activation: _Optional[_Union[ActivationFunctionData, str]] = ..., learning_rate: _Optional[float] = ...) -> None: ...
