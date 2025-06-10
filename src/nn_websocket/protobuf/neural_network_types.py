"""Dataclasses and enums for neural network and configuration Protobuf messages."""

from __future__ import annotations

from enum import IntEnum

from neural_network.math.activation_functions import (
    ActivationFunction,
    LinearActivation,
    ReluActivation,
    SigmoidActivation,
    TanhActivation,
)
from pydantic.dataclasses import dataclass

from nn_websocket.protobuf.compiled.NNWebsocketClasses_pb2 import (
    ActivationFunctionEnum,
    Configuration,
    FitnessApproachConfig,
    GeneticAlgorithmConfig,
    NeuralNetworkConfig,
    NeuroevolutionConfig,
)


class ActivationFunctionEnumData(IntEnum):
    """Enum for supported activation functions."""

    LINEAR = 0
    RELU = 1
    SIGMOID = 2
    TANH = 3

    @property
    def map(self) -> dict[ActivationFunctionEnumData, type[ActivationFunctionEnum]]:
        """Return a mapping from enum to activation function class.

        :return dict[ActivationFunctionEnumData, type[ActivationFunctionEnum]]:
            Mapping from enum to activation function class.
        """
        return {
            ActivationFunctionEnumData.LINEAR: LinearActivation,
            ActivationFunctionEnumData.RELU: ReluActivation,
            ActivationFunctionEnumData.SIGMOID: SigmoidActivation,
            ActivationFunctionEnumData.TANH: TanhActivation,
        }

    def get_class(self) -> type[ActivationFunction]:
        """Return the corresponding activation function class.

        :return type[ActivationFunction]:
            The activation function class.
        """
        return self.map[self]

    @classmethod
    def from_class(cls, activation_function: type[ActivationFunction]) -> ActivationFunctionEnumData:
        """Return the enum value for a given activation function class.

        :param type[ActivationFunction] activation_function:
            The activation function class.
        :return ActivationFunctionEnumData:
            The corresponding enum value.
        """
        reverse_map = {v: k for k, v in cls.LINEAR.map.items()}
        return reverse_map[activation_function]

    @classmethod
    def from_protobuf(cls, proto_enum_value: ActivationFunctionEnum) -> ActivationFunctionEnumData:
        """Return the enum value from a Protobuf ActivationFunctionEnum value.

        :param ActivationFunctionEnum proto_enum_value:
            The Protobuf enum value.
        :return ActivationFunctionEnumData:
            The corresponding enum value.
        """
        return cls(proto_enum_value)

    @staticmethod
    def to_protobuf(enum_value: ActivationFunctionEnumData) -> ActivationFunctionEnum:
        """Return the Protobuf ActivationFunctionEnum from an enum value.

        :param ActivationFunctionEnumData enum_value:
            The enum value.
        :return ActivationFunctionEnum:
            The Protobuf enum value.
        """
        return ActivationFunctionEnum.Value(enum_value.name)  # type: ignore[return-value]


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
    input_activation: ActivationFunctionEnumData
    hidden_activation: ActivationFunctionEnumData
    output_activation: ActivationFunctionEnumData
    learning_rate: float

    @classmethod
    def from_protobuf(cls, config: NeuralNetworkConfig) -> NeuralNetworkConfigData:
        """Create a NeuralNetworkConfigData instance from Protobuf.

        :param NeuralNetworkConfig config:
            The Protobuf neural network config.
        :return NeuralNetworkConfigData:
            The created data instance.
        """
        return cls(
            num_inputs=config.num_inputs,
            num_outputs=config.num_outputs,
            hidden_layer_sizes=list(config.hidden_layer_sizes),
            weights_min=config.weights_min,
            weights_max=config.weights_max,
            bias_min=config.bias_min,
            bias_max=config.bias_max,
            input_activation=ActivationFunctionEnumData.from_protobuf(config.input_activation),
            hidden_activation=ActivationFunctionEnumData.from_protobuf(config.hidden_activation),
            output_activation=ActivationFunctionEnumData.from_protobuf(config.output_activation),
            learning_rate=config.learning_rate,
        )

    @staticmethod
    def to_protobuf(config_data: NeuralNetworkConfigData) -> NeuralNetworkConfig:
        """Convert NeuralNetworkConfigData to Protobuf.

        :param NeuralNetworkConfigData config_data:
            The data instance to convert.
        :return NeuralNetworkConfig:
            The Protobuf config.
        """
        return NeuralNetworkConfig(
            num_inputs=config_data.num_inputs,
            num_outputs=config_data.num_outputs,
            hidden_layer_sizes=config_data.hidden_layer_sizes,
            weights_min=config_data.weights_min,
            weights_max=config_data.weights_max,
            bias_min=config_data.bias_min,
            bias_max=config_data.bias_max,
            input_activation=ActivationFunctionEnumData.to_protobuf(config_data.input_activation),
            hidden_activation=ActivationFunctionEnumData.to_protobuf(config_data.hidden_activation),
            output_activation=ActivationFunctionEnumData.to_protobuf(config_data.output_activation),
            learning_rate=config_data.learning_rate,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> NeuralNetworkConfigData:
        """Create a NeuralNetworkConfigData instance from Protobuf bytes.

        :param bytes data:
            The Protobuf bytes.
        :return NeuralNetworkConfigData:
            The created data instance.
        """
        config = NeuralNetworkConfig()
        config.ParseFromString(data)
        return cls.from_protobuf(config)

    @staticmethod
    def to_bytes(config_data: NeuralNetworkConfigData) -> bytes:
        """Convert NeuralNetworkConfigData to Protobuf bytes.

        :param NeuralNetworkConfigData config_data:
            The data instance to convert.
        :return bytes:
            The Protobuf bytes.
        """
        config = NeuralNetworkConfigData.to_protobuf(config_data)
        return config.SerializeToString()


# Training methods
@dataclass
class ConfigurationData:
    """Data class to hold configuration data for the websocket server."""

    neuroevolution: NeuroevolutionConfigData | None = None
    fitness_approach: FitnessApproachConfigData | None = None

    @classmethod
    def from_protobuf(cls, config: Configuration) -> ConfigurationData:
        """Create a ConfigurationData instance from Protobuf.

        :param Configuration config:
            The Protobuf configuration.
        :return ConfigurationData:
            The created data instance.
        """
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
        """Convert ConfigurationData to Protobuf.

        :param ConfigurationData config_data:
            The data instance to convert.
        :return Configuration:
            The Protobuf configuration.
        """
        neuroevolution = config_data.neuroevolution
        fitness_approach = config_data.fitness_approach
        return Configuration(
            neuroevolution=NeuroevolutionConfigData.to_protobuf(neuroevolution) if neuroevolution else None,
            fitness_approach=FitnessApproachConfigData.to_protobuf(fitness_approach) if fitness_approach else None,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> ConfigurationData:
        """Create a ConfigurationData instance from Protobuf bytes.

        :param bytes data:
            The Protobuf bytes.
        :return ConfigurationData:
            The created data instance.
        """
        config = Configuration()
        config.ParseFromString(data)
        return cls.from_protobuf(config)

    @staticmethod
    def to_bytes(config_data: ConfigurationData) -> bytes:
        """Convert ConfigurationData to Protobuf bytes.

        :param ConfigurationData config_data:
            The data instance to convert.
        :return bytes:
            The Protobuf bytes.
        """
        config = ConfigurationData.to_protobuf(config_data)
        return config.SerializeToString()


@dataclass
class GeneticAlgorithmConfigData:
    """Data class to hold genetic algorithm configuration."""

    population_size: int
    mutation_rate: float

    @classmethod
    def from_protobuf(cls, config: GeneticAlgorithmConfig) -> GeneticAlgorithmConfigData:
        """Create a GeneticAlgorithmConfigData instance from Protobuf.

        :param GeneticAlgorithmConfig config:
            The Protobuf genetic algorithm config.
        :return GeneticAlgorithmConfigData:
            The created data instance.
        """
        return cls(
            population_size=config.population_size,
            mutation_rate=config.mutation_rate,
        )

    @staticmethod
    def to_protobuf(config_data: GeneticAlgorithmConfigData) -> GeneticAlgorithmConfig:
        """Convert GeneticAlgorithmConfigData to Protobuf.

        :param GeneticAlgorithmConfigData config_data:
            The data instance to convert.
        :return GeneticAlgorithmConfig:
            The Protobuf config.
        """
        return GeneticAlgorithmConfig(
            population_size=config_data.population_size,
            mutation_rate=config_data.mutation_rate,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> GeneticAlgorithmConfigData:
        """Create a GeneticAlgorithmConfigData instance from Protobuf bytes.

        :param bytes data:
            The Protobuf bytes.
        :return GeneticAlgorithmConfigData:
            The created data instance.
        """
        config = GeneticAlgorithmConfig()
        config.ParseFromString(data)
        return cls.from_protobuf(config)

    @staticmethod
    def to_bytes(config_data: GeneticAlgorithmConfigData) -> bytes:
        """Convert GeneticAlgorithmConfigData to Protobuf bytes.

        :param GeneticAlgorithmConfigData config_data:
            The data instance to convert.
        :return bytes:
            The Protobuf bytes.
        """
        config = GeneticAlgorithmConfigData.to_protobuf(config_data)
        return config.SerializeToString()


@dataclass
class NeuroevolutionConfigData:
    """Data class to hold neuroevolution configuration."""

    neural_network: NeuralNetworkConfigData
    genetic_algorithm: GeneticAlgorithmConfigData

    @classmethod
    def from_protobuf(cls, config: NeuroevolutionConfig) -> NeuroevolutionConfigData:
        """Create a NeuroevolutionConfigData instance from Protobuf.

        :param NeuroevolutionConfig config:
            The Protobuf neuroevolution config.
        :return NeuroevolutionConfigData:
            The created data instance.
        """
        return cls(
            neural_network=NeuralNetworkConfigData.from_protobuf(config.neural_network),
            genetic_algorithm=GeneticAlgorithmConfigData.from_protobuf(config.genetic_algorithm),
        )

    @staticmethod
    def to_protobuf(config_data: NeuroevolutionConfigData) -> NeuroevolutionConfig:
        """Convert NeuroevolutionConfigData to Protobuf.

        :param NeuroevolutionConfigData config_data:
            The data instance to convert.
        :return NeuroevolutionConfig:
            The Protobuf config.
        """
        return NeuroevolutionConfig(
            neural_network=NeuralNetworkConfigData.to_protobuf(config_data.neural_network),
            genetic_algorithm=GeneticAlgorithmConfigData.to_protobuf(config_data.genetic_algorithm),
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> NeuroevolutionConfigData:
        """Create a NeuroevolutionConfigData instance from Protobuf bytes.

        :param bytes data:
            The Protobuf bytes.
        :return NeuroevolutionConfigData:
            The created data instance.
        """
        config = NeuroevolutionConfig()
        config.ParseFromString(data)
        return cls.from_protobuf(config)

    @staticmethod
    def to_bytes(config_data: NeuroevolutionConfigData) -> bytes:
        """Convert NeuroevolutionConfigData to Protobuf bytes.

        :param NeuroevolutionConfigData config_data:
            The data instance to convert.
        :return bytes:
            The Protobuf bytes.
        """
        config = NeuroevolutionConfigData.to_protobuf(config_data)
        return config.SerializeToString()


@dataclass
class FitnessApproachConfigData:
    """Data class to hold fitness approach configuration."""

    neural_network: NeuralNetworkConfigData

    @classmethod
    def from_protobuf(cls, config: FitnessApproachConfig) -> FitnessApproachConfigData:
        """Create a FitnessApproachConfigData instance from Protobuf.

        :param FitnessApproachConfig config:
            The Protobuf fitness approach config.
        :return FitnessApproachConfigData:
            The created data instance.
        """
        return cls(
            neural_network=NeuralNetworkConfigData.from_protobuf(config.neural_network),
        )

    @staticmethod
    def to_protobuf(config_data: FitnessApproachConfigData) -> FitnessApproachConfig:
        """Convert FitnessApproachConfigData to Protobuf.

        :param FitnessApproachConfigData config_data:
            The data instance to convert.
        :return FitnessApproachConfig:
            The Protobuf config.
        """
        return FitnessApproachConfig(
            neural_network=NeuralNetworkConfigData.to_protobuf(config_data.neural_network),
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> FitnessApproachConfigData:
        """Create a FitnessApproachConfigData instance from Protobuf bytes.

        :param bytes data:
            The Protobuf bytes.
        :return FitnessApproachConfigData:
            The created data instance.
        """
        config = FitnessApproachConfig()
        config.ParseFromString(data)
        return cls.from_protobuf(config)

    @staticmethod
    def to_bytes(config_data: FitnessApproachConfigData) -> bytes:
        """Convert FitnessApproachConfigData to Protobuf bytes.

        :param FitnessApproachConfigData config_data:
            The data instance to convert.
        :return bytes:
            The Protobuf bytes.
        """
        config = FitnessApproachConfigData.to_protobuf(config_data)
        return config.SerializeToString()
