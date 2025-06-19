"""Dataclasses and enums for neural network and configuration Protobuf messages."""

from __future__ import annotations

from neural_network.protobuf.neural_network_types import (
    ActivationFunctionEnum,
    OptimizerDataType,
)
from pydantic.dataclasses import dataclass

from nn_websocket.protobuf.compiled.NNWebsocketData_pb2 import (
    ConfigData,
    FitnessApproachConfig,
    GeneticAlgorithmConfig,
    NeuralNetworkConfig,
    NeuroevolutionConfig,
)


@dataclass
class ConfigDataType:
    """Data class to hold configuration data for the websocket server."""

    neuroevolution: NeuroevolutionConfigType | None = None
    fitness_approach: FitnessApproachConfigType | None = None

    @classmethod
    def from_protobuf(cls, config: ConfigData) -> ConfigDataType:
        """Create a ConfigDataType instance from Protobuf.

        :param ConfigData config:
            The Protobuf configuration.
        :return ConfigDataType:
            The created data instance.
        """
        result = cls()

        which_oneof = config.WhichOneof("msg")
        match which_oneof:
            case "neuroevolution":
                result.neuroevolution = NeuroevolutionConfigType.from_protobuf(config.neuroevolution)
            case "fitness_approach":
                result.fitness_approach = FitnessApproachConfigType.from_protobuf(config.fitness_approach)
            case _:
                pass

        return result

    @classmethod
    def to_protobuf(cls, config_data: ConfigDataType) -> ConfigData:
        """Convert ConfigDataType to Protobuf.

        :param ConfigDataType config_data:
            The data instance to convert.
        :return ConfigData:
            The Protobuf configuration.
        """
        neuroevolution = config_data.neuroevolution
        fitness_approach = config_data.fitness_approach
        return ConfigData(
            neuroevolution=NeuroevolutionConfigType.to_protobuf(neuroevolution) if neuroevolution else None,
            fitness_approach=FitnessApproachConfigType.to_protobuf(fitness_approach) if fitness_approach else None,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> ConfigDataType:
        """Create a ConfigDataType instance from Protobuf bytes.

        :param bytes data:
            The Protobuf bytes.
        :return ConfigDataType:
            The created data instance.
        """
        config = ConfigData()
        config.ParseFromString(data)
        return cls.from_protobuf(config)

    @staticmethod
    def to_bytes(config_data: ConfigDataType) -> bytes:
        """Convert ConfigDataType to Protobuf bytes.

        :param ConfigDataType config_data:
            The data instance to convert.
        :return bytes:
            The Protobuf bytes.
        """
        config = ConfigDataType.to_protobuf(config_data)
        return config.SerializeToString()


@dataclass
class NeuralNetworkConfigType:
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
    optimizer: OptimizerDataType

    @classmethod
    def from_protobuf(cls, config: NeuralNetworkConfig) -> NeuralNetworkConfigType:
        """Create a NeuralNetworkConfigType instance from Protobuf.

        :param NeuralNetworkConfig config:
            The Protobuf neural network config.
        :return NeuralNetworkConfigType:
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
            input_activation=ActivationFunctionEnum.from_protobuf(config.input_activation),
            hidden_activation=ActivationFunctionEnum.from_protobuf(config.hidden_activation),
            output_activation=ActivationFunctionEnum.from_protobuf(config.output_activation),
            optimizer=OptimizerDataType.from_protobuf(config.optimizer),
        )

    @staticmethod
    def to_protobuf(config_data: NeuralNetworkConfigType) -> NeuralNetworkConfig:
        """Convert NeuralNetworkConfigType to Protobuf.

        :param NeuralNetworkConfigType config_data:
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
            input_activation=ActivationFunctionEnum.to_protobuf(config_data.input_activation),
            hidden_activation=ActivationFunctionEnum.to_protobuf(config_data.hidden_activation),
            output_activation=ActivationFunctionEnum.to_protobuf(config_data.output_activation),
            optimizer=OptimizerDataType.to_protobuf(config_data.optimizer),
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> NeuralNetworkConfigType:
        """Create a NeuralNetworkConfigType instance from Protobuf bytes.

        :param bytes data:
            The Protobuf bytes.
        :return NeuralNetworkConfigType:
            The created data instance.
        """
        config = NeuralNetworkConfig()
        config.ParseFromString(data)
        return cls.from_protobuf(config)

    @staticmethod
    def to_bytes(config_data: NeuralNetworkConfigType) -> bytes:
        """Convert NeuralNetworkConfigType to Protobuf bytes.

        :param NeuralNetworkConfigType config_data:
            The data instance to convert.
        :return bytes:
            The Protobuf bytes.
        """
        config = NeuralNetworkConfigType.to_protobuf(config_data)
        return config.SerializeToString()


@dataclass
class GeneticAlgorithmConfigType:
    """Data class to hold genetic algorithm configuration."""

    population_size: int
    mutation_rate: float

    @classmethod
    def from_protobuf(cls, config: GeneticAlgorithmConfig) -> GeneticAlgorithmConfigType:
        """Create a GeneticAlgorithmConfigType instance from Protobuf.

        :param GeneticAlgorithmConfig config:
            The Protobuf genetic algorithm config.
        :return GeneticAlgorithmConfigType:
            The created data instance.
        """
        return cls(
            population_size=config.population_size,
            mutation_rate=config.mutation_rate,
        )

    @staticmethod
    def to_protobuf(config_data: GeneticAlgorithmConfigType) -> GeneticAlgorithmConfig:
        """Convert GeneticAlgorithmConfigType to Protobuf.

        :param GeneticAlgorithmConfigType config_data:
            The data instance to convert.
        :return GeneticAlgorithmConfig:
            The Protobuf config.
        """
        return GeneticAlgorithmConfig(
            population_size=config_data.population_size,
            mutation_rate=config_data.mutation_rate,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> GeneticAlgorithmConfigType:
        """Create a GeneticAlgorithmConfigType instance from Protobuf bytes.

        :param bytes data:
            The Protobuf bytes.
        :return GeneticAlgorithmConfigType:
            The created data instance.
        """
        config = GeneticAlgorithmConfig()
        config.ParseFromString(data)
        return cls.from_protobuf(config)

    @staticmethod
    def to_bytes(config_data: GeneticAlgorithmConfigType) -> bytes:
        """Convert GeneticAlgorithmConfigType to Protobuf bytes.

        :param GeneticAlgorithmConfigType config_data:
            The data instance to convert.
        :return bytes:
            The Protobuf bytes.
        """
        config = GeneticAlgorithmConfigType.to_protobuf(config_data)
        return config.SerializeToString()


@dataclass
class NeuroevolutionConfigType:
    """Data class to hold neuroevolution configuration."""

    neural_network: NeuralNetworkConfigType
    genetic_algorithm: GeneticAlgorithmConfigType

    @classmethod
    def from_protobuf(cls, config: NeuroevolutionConfig) -> NeuroevolutionConfigType:
        """Create a NeuroevolutionConfigType instance from Protobuf.

        :param NeuroevolutionConfig config:
            The Protobuf neuroevolution config.
        :return NeuroevolutionConfigType:
            The created data instance.
        """
        return cls(
            neural_network=NeuralNetworkConfigType.from_protobuf(config.neural_network),
            genetic_algorithm=GeneticAlgorithmConfigType.from_protobuf(config.genetic_algorithm),
        )

    @staticmethod
    def to_protobuf(config_data: NeuroevolutionConfigType) -> NeuroevolutionConfig:
        """Convert NeuroevolutionConfigType to Protobuf.

        :param NeuroevolutionConfigType config_data:
            The data instance to convert.
        :return NeuroevolutionConfig:
            The Protobuf config.
        """
        return NeuroevolutionConfig(
            neural_network=NeuralNetworkConfigType.to_protobuf(config_data.neural_network),
            genetic_algorithm=GeneticAlgorithmConfigType.to_protobuf(config_data.genetic_algorithm),
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> NeuroevolutionConfigType:
        """Create a NeuroevolutionConfigType instance from Protobuf bytes.

        :param bytes data:
            The Protobuf bytes.
        :return NeuroevolutionConfigType:
            The created data instance.
        """
        config = NeuroevolutionConfig()
        config.ParseFromString(data)
        return cls.from_protobuf(config)

    @staticmethod
    def to_bytes(config_data: NeuroevolutionConfigType) -> bytes:
        """Convert NeuroevolutionConfigType to Protobuf bytes.

        :param NeuroevolutionConfigType config_data:
            The data instance to convert.
        :return bytes:
            The Protobuf bytes.
        """
        config = NeuroevolutionConfigType.to_protobuf(config_data)
        return config.SerializeToString()


@dataclass
class FitnessApproachConfigType:
    """Data class to hold fitness approach configuration."""

    neural_network: NeuralNetworkConfigType

    @classmethod
    def from_protobuf(cls, config: FitnessApproachConfig) -> FitnessApproachConfigType:
        """Create a FitnessApproachConfigType instance from Protobuf.

        :param FitnessApproachConfig config:
            The Protobuf fitness approach config.
        :return FitnessApproachConfigType:
            The created data instance.
        """
        return cls(
            neural_network=NeuralNetworkConfigType.from_protobuf(config.neural_network),
        )

    @staticmethod
    def to_protobuf(config_data: FitnessApproachConfigType) -> FitnessApproachConfig:
        """Convert FitnessApproachConfigType to Protobuf.

        :param FitnessApproachConfigType config_data:
            The data instance to convert.
        :return FitnessApproachConfig:
            The Protobuf config.
        """
        return FitnessApproachConfig(
            neural_network=NeuralNetworkConfigType.to_protobuf(config_data.neural_network),
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> FitnessApproachConfigType:
        """Create a FitnessApproachConfigType instance from Protobuf bytes.

        :param bytes data:
            The Protobuf bytes.
        :return FitnessApproachConfigType:
            The created data instance.
        """
        config = FitnessApproachConfig()
        config.ParseFromString(data)
        return cls.from_protobuf(config)

    @staticmethod
    def to_bytes(config_data: FitnessApproachConfigType) -> bytes:
        """Convert FitnessApproachConfigType to Protobuf bytes.

        :param FitnessApproachConfigType config_data:
            The data instance to convert.
        :return bytes:
            The Protobuf bytes.
        """
        config = FitnessApproachConfigType.to_protobuf(config_data)
        return config.SerializeToString()
