from __future__ import annotations

from enum import IntEnum

from neural_network.math.activation_functions import LinearActivation, ReluActivation, SigmoidActivation
from pydantic.dataclasses import dataclass

from nn_websocket.protobuf.compiled.FrameData_pb2 import Action, Fitness, FrameRequest, Observation, TrainRequest
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

    neural_network: NeuralNetworkConfigData
    genetic_algorithm: GeneticAlgorithmConfigData | None = None

    @classmethod
    def from_protobuf(cls, config: Configuration) -> ConfigurationData:
        """Creates a ConfigurationData instance from Protobuf."""
        return cls(
            neural_network=NeuralNetworkConfigData.from_protobuf(config.neural_network),
            genetic_algorithm=GeneticAlgorithmConfigData.from_protobuf(config.genetic_algorithm),
        )

    @classmethod
    def to_protobuf(cls, config_data: ConfigurationData) -> Configuration:
        """Converts ConfigurationData to Protobuf."""
        neural_network = config_data.neural_network
        genetic_algorithm = config_data.genetic_algorithm or None
        return Configuration(
            neural_network=NeuralNetworkConfigData.to_protobuf(neural_network),
            genetic_algorithm=GeneticAlgorithmConfigData.to_protobuf(genetic_algorithm) if genetic_algorithm else None,
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


# frame_data.proto
@dataclass
class FrameRequestData:
    """Data class to hold frame request data."""

    observation: ObservationData | None = None
    fitness: FitnessData | None = None
    train_request: TrainRequestData | None = None

    @classmethod
    def from_bytes(cls, data: bytes) -> FrameRequestData:
        """Creates a FrameRequestData instance from Protobuf bytes."""
        frame_request = FrameRequest()
        frame_request.ParseFromString(data)

        result = cls()

        which_oneof = frame_request.WhichOneof("msg")
        match which_oneof:
            case "observation":
                result.observation = ObservationData.from_protobuf(frame_request.observation)
            case "fitness":
                result.fitness = FitnessData.from_protobuf(frame_request.fitness)
            case "train_request":
                result.train_request = TrainRequestData.from_protobuf(frame_request.train_request)
            case _:
                pass

        return result

    @staticmethod
    def to_bytes(frame_request_data: FrameRequestData) -> bytes:
        """Converts FrameRequestData to Protobuf bytes."""
        frame_request = FrameRequest()

        if frame_request_data.observation is not None:
            frame_request.observation.CopyFrom(ObservationData.to_protobuf(frame_request_data.observation))
        elif frame_request_data.fitness is not None:
            frame_request.fitness.CopyFrom(FitnessData.to_protobuf(frame_request_data.fitness))
        elif frame_request_data.train_request is not None:
            frame_request.train_request.CopyFrom(TrainRequestData.to_protobuf(frame_request_data.train_request))

        return frame_request.SerializeToString()


@dataclass
class TrainRequestData:
    """Data class to hold training request data."""

    observation: list[ObservationData]
    action: list[ActionData]
    fitness: list[FitnessData]

    @classmethod
    def from_protobuf(cls, train_request: TrainRequest) -> TrainRequestData:
        """Creates a TrainRequestData instance from Protobuf."""
        return cls(
            observation=[ObservationData.from_protobuf(obs) for obs in train_request.observation],
            action=[ActionData.from_protobuf(act) for act in train_request.action],
            fitness=[FitnessData.from_protobuf(fit) for fit in train_request.fitness],
        )

    @classmethod
    def to_protobuf(cls, train_request: TrainRequestData) -> TrainRequest:
        """Converts TrainRequestData to Protobuf."""
        return TrainRequest(
            observation=[ObservationData.to_protobuf(observation) for observation in train_request.observation],
            action=[ActionData.to_protobuf(action) for action in train_request.action],
            fitness=[FitnessData.to_protobuf(fitness) for fitness in train_request.fitness],
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> TrainRequestData:
        """Creates a TrainRequestData instance from Protobuf bytes."""
        train_request = TrainRequest()
        train_request.ParseFromString(data)
        return cls.from_protobuf(train_request)

    @staticmethod
    def to_bytes(train_request: TrainRequestData) -> bytes:
        """Converts TrainRequestData to Protobuf bytes."""
        train_request_proto = TrainRequestData.to_protobuf(train_request)
        return train_request_proto.SerializeToString()


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


@dataclass
class FitnessData:
    """Data class to hold population fitness data."""

    values: list[float]

    @classmethod
    def from_protobuf(cls, fitness: Fitness) -> FitnessData:
        """Creates a FitnessData instance from Protobuf."""
        return cls(values=list(fitness.values))

    @staticmethod
    def to_protobuf(fitness_data: FitnessData) -> Fitness:
        """Converts FitnessData to Protobuf."""
        return Fitness(values=fitness_data.values)

    @classmethod
    def from_bytes(cls, data: bytes) -> FitnessData:
        """Creates a FitnessData instance from Protobuf bytes."""
        fitness = Fitness()
        fitness.ParseFromString(data)
        return cls.from_protobuf(fitness)

    @staticmethod
    def to_bytes(fitness_data: FitnessData) -> bytes:
        """Converts FitnessData to Protobuf bytes."""
        fitness = FitnessData.to_protobuf(fitness_data)
        return fitness.SerializeToString()
