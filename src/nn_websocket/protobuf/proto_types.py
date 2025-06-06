from __future__ import annotations

from enum import IntEnum

from neural_network.math.activation_functions import LinearActivation, ReluActivation, SigmoidActivation
from pydantic.dataclasses import dataclass

from nn_websocket.protobuf.compiled.FrameData_pb2 import Action, FrameRequest, Observation, PopulationFitness
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

    @classmethod
    def from_bytes(cls, data: bytes) -> ConfigurationData:
        """Creates a ConfigurationData instance from Protobuf bytes."""
        config = Configuration()
        config.ParseFromString(data)
        return cls.from_protobuf(config)

    @staticmethod
    def to_bytes(config_data: ConfigurationData) -> bytes:
        """Converts ConfigurationData to Protobuf bytes."""
        config = ConfigurationData.to_protobuf(config_data)
        return config.SerializeToString()


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

    @staticmethod
    def to_protobuf(config_data: GeneticAlgorithmConfigData) -> GeneticAlgorithmConfig:
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
        return config.SerializeToString()


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

    @classmethod
    def from_protobuf(cls, proto_enum_value: ActivationFunction) -> ActivationFunctionEnum:
        """Maps a Protobuf ActivationFunction value to ActivationFunctionEnum."""
        return cls(proto_enum_value)

    @staticmethod
    def to_protobuf(enum_value: ActivationFunctionEnum) -> ActivationFunction:
        """Maps an ActivationFunctionEnum value to Protobuf ActivationFunction."""
        return ActivationFunction.Value(enum_value.name)  # type: ignore


# frame_data.proto
@dataclass
class FrameRequestData:
    """Data class to hold frame request data."""

    population_fitness: PopulationFitnessData | None = None
    observation: ObservationData | None = None

    @classmethod
    def from_bytes(cls, data: bytes) -> FrameRequestData:
        """Creates a FrameRequestData instance from Protobuf bytes."""
        frame_request = FrameRequest()
        frame_request.ParseFromString(data)

        result = cls()

        which_oneof = frame_request.WhichOneof("msg")
        if which_oneof == "population_fitness":
            result.population_fitness = PopulationFitnessData.from_protobuf(frame_request.population_fitness)
        elif which_oneof == "observation":
            result.observation = ObservationData.from_protobuf(frame_request.observation)

        return result

    @staticmethod
    def to_bytes(frame_request_data: FrameRequestData) -> bytes:
        """Converts FrameRequestData to Protobuf bytes."""
        frame_request = FrameRequest()

        if frame_request_data.population_fitness is not None:
            frame_request.population_fitness.CopyFrom(
                PopulationFitnessData.to_protobuf(frame_request_data.population_fitness)
            )
        elif frame_request_data.observation is not None:
            frame_request.observation.CopyFrom(ObservationData.to_protobuf(frame_request_data.observation))

        return frame_request.SerializeToString()


@dataclass
class PopulationFitnessData:
    """Data class to hold population fitness data."""

    fitness: list[float]

    @classmethod
    def from_protobuf(cls, population_fitness: PopulationFitness) -> PopulationFitnessData:
        """Creates a PopulationFitnessData instance from Protobuf."""
        return cls(fitness=list(population_fitness.fitness))

    @staticmethod
    def to_protobuf(population_fitness_data: PopulationFitnessData) -> PopulationFitness:
        """Converts PopulationFitnessData to Protobuf."""
        return PopulationFitness(fitness=population_fitness_data.fitness)

    @classmethod
    def from_bytes(cls, data: bytes) -> PopulationFitnessData:
        """Creates a PopulationFitnessData instance from Protobuf bytes."""
        population_fitness = PopulationFitness()
        population_fitness.ParseFromString(data)
        return cls.from_protobuf(population_fitness)

    @staticmethod
    def to_bytes(population_fitness_data: PopulationFitnessData) -> bytes:
        """Converts PopulationFitnessData to Protobuf bytes."""
        population_fitness = PopulationFitnessData.to_protobuf(population_fitness_data)
        return population_fitness.SerializeToString()


@dataclass
class ObservationData:
    """Data class to hold observation data."""

    inputs: list[float]

    @classmethod
    def from_protobuf(cls, observation: Observation) -> ObservationData:
        """Creates an ObservationData instance from Protobuf."""
        return cls(inputs=list(observation.inputs))

    @staticmethod
    def to_protobuf(observation_data: ObservationData) -> Observation:
        """Converts ObservationData to Protobuf."""
        return Observation(inputs=observation_data.inputs)

    @classmethod
    def from_bytes(cls, data: bytes) -> ObservationData:
        """Creates an ObservationData instance from Protobuf bytes."""
        observation = Observation()
        observation.ParseFromString(data)
        return cls.from_protobuf(observation)

    @staticmethod
    def to_bytes(observation_data: ObservationData) -> bytes:
        """Converts ObservationData to Protobuf bytes."""
        observation = ObservationData.to_protobuf(observation_data)
        return observation.SerializeToString()


@dataclass
class ActionData:
    """Data class to hold action data."""

    outputs: list[float]

    @classmethod
    def from_protobuf(cls, action: Action) -> ActionData:
        """Creates an ActionData instance from Protobuf."""
        return cls(outputs=list(action.outputs))

    @staticmethod
    def to_protobuf(action_data: ActionData) -> Action:
        """Converts ActionData to Protobuf."""
        return Action(outputs=action_data.outputs)

    @classmethod
    def from_bytes(cls, data: bytes) -> ActionData:
        """Creates an ActionData instance from Protobuf bytes."""
        action = Action()
        action.ParseFromString(data)
        return cls.from_protobuf(action)

    @staticmethod
    def to_bytes(action_data: ActionData) -> bytes:
        """Converts ActionData to Protobuf bytes."""
        action = ActionData.to_protobuf(action_data)
        return action.SerializeToString()
