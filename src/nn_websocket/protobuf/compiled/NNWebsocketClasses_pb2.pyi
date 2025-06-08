from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ActivationFunctionEnum(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    LINEAR: _ClassVar[ActivationFunctionEnum]
    RELU: _ClassVar[ActivationFunctionEnum]
    SIGMOID: _ClassVar[ActivationFunctionEnum]
    TANH: _ClassVar[ActivationFunctionEnum]
LINEAR: ActivationFunctionEnum
RELU: ActivationFunctionEnum
SIGMOID: ActivationFunctionEnum
TANH: ActivationFunctionEnum

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
    input_activation: ActivationFunctionEnum
    hidden_activation: ActivationFunctionEnum
    output_activation: ActivationFunctionEnum
    learning_rate: float
    def __init__(self, num_inputs: _Optional[int] = ..., num_outputs: _Optional[int] = ..., hidden_layer_sizes: _Optional[_Iterable[int]] = ..., weights_min: _Optional[float] = ..., weights_max: _Optional[float] = ..., bias_min: _Optional[float] = ..., bias_max: _Optional[float] = ..., input_activation: _Optional[_Union[ActivationFunctionEnum, str]] = ..., hidden_activation: _Optional[_Union[ActivationFunctionEnum, str]] = ..., output_activation: _Optional[_Union[ActivationFunctionEnum, str]] = ..., learning_rate: _Optional[float] = ...) -> None: ...

class Configuration(_message.Message):
    __slots__ = ("neuroevolution", "fitness_approach")
    NEUROEVOLUTION_FIELD_NUMBER: _ClassVar[int]
    FITNESS_APPROACH_FIELD_NUMBER: _ClassVar[int]
    neuroevolution: NeuroevolutionConfig
    fitness_approach: FitnessApproachConfig
    def __init__(self, neuroevolution: _Optional[_Union[NeuroevolutionConfig, _Mapping]] = ..., fitness_approach: _Optional[_Union[FitnessApproachConfig, _Mapping]] = ...) -> None: ...

class GeneticAlgorithmConfig(_message.Message):
    __slots__ = ("population_size", "mutation_rate")
    POPULATION_SIZE_FIELD_NUMBER: _ClassVar[int]
    MUTATION_RATE_FIELD_NUMBER: _ClassVar[int]
    population_size: int
    mutation_rate: float
    def __init__(self, population_size: _Optional[int] = ..., mutation_rate: _Optional[float] = ...) -> None: ...

class NeuroevolutionConfig(_message.Message):
    __slots__ = ("neural_network", "genetic_algorithm")
    NEURAL_NETWORK_FIELD_NUMBER: _ClassVar[int]
    GENETIC_ALGORITHM_FIELD_NUMBER: _ClassVar[int]
    neural_network: NeuralNetworkConfig
    genetic_algorithm: GeneticAlgorithmConfig
    def __init__(self, neural_network: _Optional[_Union[NeuralNetworkConfig, _Mapping]] = ..., genetic_algorithm: _Optional[_Union[GeneticAlgorithmConfig, _Mapping]] = ...) -> None: ...

class FitnessApproachConfig(_message.Message):
    __slots__ = ("neural_network",)
    NEURAL_NETWORK_FIELD_NUMBER: _ClassVar[int]
    neural_network: NeuralNetworkConfig
    def __init__(self, neural_network: _Optional[_Union[NeuralNetworkConfig, _Mapping]] = ...) -> None: ...
