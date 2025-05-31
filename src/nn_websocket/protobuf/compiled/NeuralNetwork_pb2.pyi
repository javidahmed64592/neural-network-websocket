from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ActivationFunction(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    LINEAR: _ClassVar[ActivationFunction]
    RELU: _ClassVar[ActivationFunction]
    SIGMOID: _ClassVar[ActivationFunction]
LINEAR: ActivationFunction
RELU: ActivationFunction
SIGMOID: ActivationFunction

class GeneticAlgorithmConfig(_message.Message):
    __slots__ = ("population_size", "mutation_rate")
    POPULATION_SIZE_FIELD_NUMBER: _ClassVar[int]
    MUTATION_RATE_FIELD_NUMBER: _ClassVar[int]
    population_size: int
    mutation_rate: float
    def __init__(self, population_size: _Optional[int] = ..., mutation_rate: _Optional[float] = ...) -> None: ...

class NeuralNetworkConfig(_message.Message):
    __slots__ = ("num_inputs", "num_outputs", "hidden_layer_sizes", "weights_min", "weights_max", "bias_min", "bias_max", "input_activation", "hidden_activation", "output_activation")
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
    num_inputs: int
    num_outputs: int
    hidden_layer_sizes: _containers.RepeatedScalarFieldContainer[int]
    weights_min: float
    weights_max: float
    bias_min: float
    bias_max: float
    input_activation: ActivationFunction
    hidden_activation: ActivationFunction
    output_activation: ActivationFunction
    def __init__(self, num_inputs: _Optional[int] = ..., num_outputs: _Optional[int] = ..., hidden_layer_sizes: _Optional[_Iterable[int]] = ..., weights_min: _Optional[float] = ..., weights_max: _Optional[float] = ..., bias_min: _Optional[float] = ..., bias_max: _Optional[float] = ..., input_activation: _Optional[_Union[ActivationFunction, str]] = ..., hidden_activation: _Optional[_Union[ActivationFunction, str]] = ..., output_activation: _Optional[_Union[ActivationFunction, str]] = ...) -> None: ...
