from __future__ import annotations

from enum import IntEnum
from typing import cast

from neural_network.math.activation_functions import LinearActivation, ReluActivation, SigmoidActivation
from pydantic.dataclasses import dataclass

from nn_websocket.protobuf.compiled.FrameData_pb2 import Action, Observation, PopulationFitness
from nn_websocket.protobuf.compiled.NeuralNetwork_pb2 import (
    ActivationFunction,
    Configuration,
    GeneticAlgorithmConfig,
    NeuralNetworkConfig,
)


# neural_network.proto
@dataclass
class ConfigurationData:
    """Data class to hold configuration data."""

    genetic_algorithm: GeneticAlgorithmConfigData
    neural_network: NeuralNetworkConfigData

    @classmethod
    def from_protobuf(cls, config: Configuration) -> ConfigurationData:
        """Creates a ConfigurationData instance from Protobuf."""
        return cls(
            genetic_algorithm=GeneticAlgorithmConfigData.from_protobuf(config.genetic_algorithm),
            neural_network=NeuralNetworkConfigData.from_protobuf(config.neural_network),
        )

    @classmethod
    def to_protobuf(cls, config_data: ConfigurationData) -> Configuration:
        """Converts ConfigurationData to Protobuf."""
        return Configuration(
            genetic_algorithm=GeneticAlgorithmConfigData.to_protobuf(config_data.genetic_algorithm),
            neural_network=NeuralNetworkConfigData.to_protobuf(config_data.neural_network),
        )


@dataclass
class GeneticAlgorithmConfigData:
    """Data class to hold genetic algorithm configuration."""

    population_size: int
    mutation_rate: float

    @classmethod
    def from_protobuf(cls, config: GeneticAlgorithmConfig) -> GeneticAlgorithmConfigData:
        """Creates a GeneticAlgorithmConfigData instance from Protobuf."""
        return cls(
            population_size=config.population_size,
            mutation_rate=config.mutation_rate,
        )

    @classmethod
    def to_protobuf(cls, config_data: GeneticAlgorithmConfigData) -> GeneticAlgorithmConfig:
        """Converts GeneticAlgorithmConfigData to Protobuf."""
        return GeneticAlgorithmConfig(
            population_size=config_data.population_size,
            mutation_rate=config_data.mutation_rate,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> GeneticAlgorithmConfigData:
        """Creates a GeneticAlgorithmConfigData instance from Protobuf bytes."""
        config = GeneticAlgorithmConfig()
        config.ParseFromString(data)
        return cls.from_protobuf(config)

    @staticmethod
    def to_bytes(config_data: GeneticAlgorithmConfigData) -> bytes:
        """Converts GeneticAlgorithmConfigData to Protobuf bytes."""
        config = GeneticAlgorithmConfigData.to_protobuf(config_data)
        return cast(bytes, config.SerializeToString())


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
        )

    @classmethod
    def to_protobuf(cls, config_data: NeuralNetworkConfigData) -> NeuralNetworkConfig:
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
        return cast(bytes, config.SerializeToString())


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

    inputs: list[float]

    @classmethod
    def from_bytes(cls, data: bytes) -> ObservationData:
        """Creates an ObservationData instance from Protobuf bytes."""
        observation = Observation()
        observation.ParseFromString(data)

        return cls(inputs=list(observation.inputs))

    @staticmethod
    def to_bytes(observation_data: ObservationData) -> bytes:
        """Converts ObservationData to Protobuf bytes."""
        observation = Observation(inputs=observation_data.inputs)
        return cast(bytes, observation.SerializeToString())


@dataclass
class ActionData:
    """Data class to hold action data."""

    outputs: list[float]

    @classmethod
    def from_bytes(cls, data: bytes) -> ActionData:
        """Creates an ActionData instance from Protobuf bytes."""
        action = Action()
        action.ParseFromString(data)

        return cls(outputs=list(action.outputs))

    @staticmethod
    def to_bytes(action_data: ActionData) -> bytes:
        """Converts ActionData to Protobuf bytes."""
        action = Action(outputs=action_data.outputs)
        return cast(bytes, action.SerializeToString())


@dataclass
class PopulationFitnessData:
    """Data class to hold population fitness data."""

    fitness: list[float]

    @classmethod
    def from_bytes(cls, data: bytes) -> PopulationFitnessData:
        """Creates a PopulationFitnessData instance from Protobuf bytes."""
        population_fitness = PopulationFitness()
        population_fitness.ParseFromString(data)

        return cls(fitness=list(population_fitness.fitness))

    @staticmethod
    def to_bytes(population_fitness_data: PopulationFitnessData) -> bytes:
        """Converts PopulationFitnessData to Protobuf bytes."""
        population_fitness = PopulationFitness(fitness=population_fitness_data.fitness)
        return cast(bytes, population_fitness.SerializeToString())
