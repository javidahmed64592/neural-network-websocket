from __future__ import annotations

from enum import IntEnum

from neural_network.math.activation_functions import LinearActivation, ReluActivation, SigmoidActivation, TanhActivation
from pydantic.dataclasses import dataclass

from nn_websocket.protobuf.compiled.NeuralNetwork_pb2 import (
    ActivationFunctionData,
    Configuration,
    FitnessApproachConfig,
    GeneticAlgorithmConfig,
    NeuralNetworkConfig,
    NeuroevolutionConfig,
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


# Training methods
@dataclass
class ConfigurationData:
    """Data class to hold configuration data."""

    neuroevolution: NeuroevolutionConfigData | None = None
    fitness_approach: FitnessApproachConfigData | None = None

    @classmethod
    def from_protobuf(cls, config: Configuration) -> ConfigurationData:
        """Creates a ConfigurationData instance from Protobuf."""
        result = cls()

        which_oneof = config.WhichOneof("msg")
        match which_oneof:
            case "neuroevolution":
                result.neuroevolution = NeuroevolutionConfigData.from_protobuf(config.neuroevolution)
            case "fitness_approach":
                result.fitness_approach = FitnessApproachConfigData.from_protobuf(config.fitness_approach)
            case _:
                pass

        return result

    @classmethod
    def to_protobuf(cls, config_data: ConfigurationData) -> Configuration:
        """Converts ConfigurationData to Protobuf."""
        neuroevolution = config_data.neuroevolution
        fitness_approach = config_data.fitness_approach
        return Configuration(
            neuroevolution=NeuroevolutionConfigData.to_protobuf(neuroevolution) if neuroevolution else None,
            fitness_approach=FitnessApproachConfigData.to_protobuf(fitness_approach) if fitness_approach else None,
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
class NeuroevolutionConfigData:
    """Data class to hold neuroevolution configuration."""

    neural_network: NeuralNetworkConfigData
    genetic_algorithm: GeneticAlgorithmConfigData

    @classmethod
    def from_protobuf(cls, config: NeuroevolutionConfig) -> NeuroevolutionConfigData:
        """Creates a NeuroevolutionConfigData instance from Protobuf."""
        return cls(
            neural_network=NeuralNetworkConfigData.from_protobuf(config.neural_network),
            genetic_algorithm=GeneticAlgorithmConfigData.from_protobuf(config.genetic_algorithm),
        )

    @staticmethod
    def to_protobuf(config_data: NeuroevolutionConfigData) -> NeuroevolutionConfig:
        """Converts NeuroevolutionConfigData to Protobuf."""
        return NeuroevolutionConfig(
            neural_network=NeuralNetworkConfigData.to_protobuf(config_data.neural_network),
            genetic_algorithm=GeneticAlgorithmConfigData.to_protobuf(config_data.genetic_algorithm),
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> NeuroevolutionConfigData:
        """Creates a NeuroevolutionConfigData instance from Protobuf bytes."""
        config = NeuroevolutionConfig()
        config.ParseFromString(data)
        return cls.from_protobuf(config)

    @staticmethod
    def to_bytes(config_data: NeuroevolutionConfigData) -> bytes:
        """Converts NeuroevolutionConfigData to Protobuf bytes."""
        config = NeuroevolutionConfigData.to_protobuf(config_data)
        return config.SerializeToString()


@dataclass
class FitnessApproachConfigData:
    """Data class to hold fitness approach configuration."""

    neural_network: NeuralNetworkConfigData

    @classmethod
    def from_protobuf(cls, config: FitnessApproachConfig) -> FitnessApproachConfigData:
        """Creates a FitnessApproachConfigData instance from Protobuf."""
        return cls(
            neural_network=NeuralNetworkConfigData.from_protobuf(config.neural_network),
        )

    @staticmethod
    def to_protobuf(config_data: FitnessApproachConfigData) -> FitnessApproachConfig:
        """Converts FitnessApproachConfigData to Protobuf."""
        return FitnessApproachConfig(
            neural_network=NeuralNetworkConfigData.to_protobuf(config_data.neural_network),
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> FitnessApproachConfigData:
        """Creates a FitnessApproachConfigData instance from Protobuf bytes."""
        config = FitnessApproachConfig()
        config.ParseFromString(data)
        return cls.from_protobuf(config)

    @staticmethod
    def to_bytes(config_data: FitnessApproachConfigData) -> bytes:
        """Converts FitnessApproachConfigData to Protobuf bytes."""
        config = FitnessApproachConfigData.to_protobuf(config_data)
        return config.SerializeToString()
