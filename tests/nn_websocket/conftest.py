from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from nn_websocket.ga.nn_ga import NeuralNetworkGA
from nn_websocket.ga.nn_member import NeuralNetworkMember

# from nn_websocket.main import NeuralNetworkWebsocketServer
from nn_websocket.models.config import Config
from nn_websocket.protobuf.compiled.FrameRequestClasses_pb2 import (
    Action,
    Fitness,
    FrameRequest,
    Observation,
    TrainRequest,
)
from nn_websocket.protobuf.compiled.NNWebsocketClasses_pb2 import (
    Configuration,
    FitnessApproachConfig,
    GeneticAlgorithmConfig,
    NeuralNetworkConfig,
    NeuroevolutionConfig,
)
from nn_websocket.protobuf.frame_data_types import (
    ActionData,
    FitnessData,
    FrameRequestData,
    ObservationData,
    TrainRequestData,
)
from nn_websocket.protobuf.neural_network_types import (
    ActivationFunctionEnumData,
    ConfigurationData,
    FitnessApproachConfigData,
    GeneticAlgorithmConfigData,
    NeuralNetworkConfigData,
    NeuroevolutionConfigData,
)

# from nn_websocket.models.nn_suite import NeuralNetworkSuite

rng = np.random.default_rng()

# Constants
MOCK_CONFIG_FILEPATH = Path("path/to/websocket_config.json")
MOCK_NUM_INPUTS = 5
MOCK_NUM_OUTPUTS = 2
MOCK_HIDDEN_LAYER_SIZES = [4, 4]
MOCK_WEIGHTS_MIN = -1.0
MOCK_WEIGHTS_MAX = 1.0
MOCK_BIAS_MIN = -1.0
MOCK_BIAS_MAX = 1.0
MOCK_INPUT_ACTIVATION = ActivationFunctionEnumData.LINEAR
MOCK_HIDDEN_ACTIVATION = ActivationFunctionEnumData.RELU
MOCK_OUTPUT_ACTIVATION = ActivationFunctionEnumData.SIGMOID
MOCK_LEARNING_RATE = 0.01

MOCK_POPULATION_SIZE = 10
MOCK_MUTATION_RATE = 0.1


# NeuralNetwork.proto
@pytest.fixture
def neural_network_config() -> NeuralNetworkConfig:
    return NeuralNetworkConfig(
        num_inputs=MOCK_NUM_INPUTS,
        num_outputs=MOCK_NUM_OUTPUTS,
        hidden_layer_sizes=MOCK_HIDDEN_LAYER_SIZES,
        weights_min=MOCK_WEIGHTS_MIN,
        weights_max=MOCK_WEIGHTS_MAX,
        bias_min=MOCK_BIAS_MIN,
        bias_max=MOCK_BIAS_MAX,
        input_activation=MOCK_INPUT_ACTIVATION,
        hidden_activation=MOCK_HIDDEN_ACTIVATION,
        output_activation=MOCK_OUTPUT_ACTIVATION,
        learning_rate=MOCK_LEARNING_RATE,
    )


@pytest.fixture
def neural_network_config_data(neural_network_config: NeuralNetworkConfig) -> NeuralNetworkConfigData:
    return NeuralNetworkConfigData(
        num_inputs=neural_network_config.num_inputs,
        num_outputs=neural_network_config.num_outputs,
        hidden_layer_sizes=neural_network_config.hidden_layer_sizes,
        weights_min=neural_network_config.weights_min,
        weights_max=neural_network_config.weights_max,
        bias_min=neural_network_config.bias_min,
        bias_max=neural_network_config.bias_max,
        input_activation=MOCK_INPUT_ACTIVATION,
        hidden_activation=MOCK_HIDDEN_ACTIVATION,
        output_activation=MOCK_OUTPUT_ACTIVATION,
        learning_rate=neural_network_config.learning_rate,
    )


@pytest.fixture
def configuration_neuroevolution(neural_network_config: NeuralNetworkConfig) -> Configuration:
    return Configuration(neuroevolution=NeuroevolutionConfig(neural_network=neural_network_config))


@pytest.fixture
def configuration_fitness(neural_network_config: NeuralNetworkConfig) -> Configuration:
    return Configuration(fitness_approach=FitnessApproachConfig(neural_network=neural_network_config))


@pytest.fixture
def configuration_data_neuroevolution(configuration_neuroevolution: Configuration) -> ConfigurationData:
    return ConfigurationData.from_protobuf(configuration_neuroevolution)


@pytest.fixture
def configuration_data_fitness(configuration_fitness: Configuration) -> ConfigurationData:
    return ConfigurationData.from_protobuf(configuration_fitness)


@pytest.fixture
def genetic_algorithm_config() -> GeneticAlgorithmConfig:
    return GeneticAlgorithmConfig(
        population_size=MOCK_POPULATION_SIZE,
        mutation_rate=MOCK_MUTATION_RATE,
    )


@pytest.fixture
def genetic_algorithm_config_data(genetic_algorithm_config: GeneticAlgorithmConfig) -> GeneticAlgorithmConfigData:
    return GeneticAlgorithmConfigData.from_protobuf(genetic_algorithm_config)


@pytest.fixture
def neuroevolution_config(
    neural_network_config: NeuralNetworkConfig, genetic_algorithm_config: GeneticAlgorithmConfig
) -> NeuroevolutionConfig:
    return NeuroevolutionConfig(
        neural_network=neural_network_config,
        genetic_algorithm=genetic_algorithm_config,
    )


@pytest.fixture
def neuroevolution_config_data(neuroevolution_config: NeuroevolutionConfig) -> NeuroevolutionConfigData:
    return NeuroevolutionConfigData.from_protobuf(neuroevolution_config)


@pytest.fixture
def fitness_approach_config(neural_network_config: NeuralNetworkConfig) -> FitnessApproachConfig:
    return FitnessApproachConfig(neural_network=neural_network_config)


@pytest.fixture
def fitness_approach_config_data(fitness_approach_config: FitnessApproachConfig) -> FitnessApproachConfigData:
    return FitnessApproachConfigData.from_protobuf(fitness_approach_config)


# FrameRequestClasses_pb2.proto
@pytest.fixture
def frame_request_observation(observation: Observation) -> FrameRequest:
    return FrameRequest(observation=observation)


@pytest.fixture
def frame_request_fitness(fitness: Fitness) -> FrameRequest:
    return FrameRequest(fitness=fitness)


@pytest.fixture
def frame_request_train(train_request: TrainRequest) -> FrameRequest:
    return FrameRequest(train_request=train_request)


@pytest.fixture
def frame_request_data_observation(frame_request_observation: FrameRequest) -> FrameRequestData:
    return FrameRequestData.from_protobuf(frame_request_observation)


@pytest.fixture
def frame_request_data_fitness(frame_request_fitness: FrameRequest) -> FrameRequestData:
    return FrameRequestData.from_protobuf(frame_request_fitness)


@pytest.fixture
def frame_request_data_train(frame_request_train: FrameRequest) -> FrameRequestData:
    return FrameRequestData.from_protobuf(frame_request_train)


@pytest.fixture
def observation() -> Observation:
    return Observation(inputs=rng.uniform(0.0, 1.0, size=MOCK_NUM_INPUTS).tolist())


@pytest.fixture
def observation_data(observation: Observation) -> ObservationData:
    return ObservationData.from_protobuf(observation)


@pytest.fixture
def action() -> Action:
    return Action(outputs=rng.uniform(0.0, 1.0, size=MOCK_NUM_OUTPUTS).tolist())


@pytest.fixture
def action_data(action: Action) -> ActionData:
    return ActionData.from_protobuf(action)


@pytest.fixture
def fitness() -> Fitness:
    return Fitness(values=rng.uniform(0.0, 1.0, size=MOCK_POPULATION_SIZE).tolist())


@pytest.fixture
def fitness_data(fitness: Fitness) -> FitnessData:
    return FitnessData.from_protobuf(fitness)


@pytest.fixture
def train_request(observation: Observation, fitness: Fitness) -> TrainRequest:
    return TrainRequest(observation=[observation], fitness=[fitness])


@pytest.fixture
def train_request_data(train_request: TrainRequest) -> TrainRequestData:
    return TrainRequestData.from_protobuf(train_request)


# Genetic Algorithm fixtures
@pytest.fixture
def mock_neural_network_member(neural_network_config_data: NeuralNetworkConfigData) -> NeuralNetworkMember:
    """Fixture for NeuralNetworkMember."""
    return NeuralNetworkMember.from_config_data(neural_network_config_data)


@pytest.fixture
def mock_neural_network_ga(
    neural_network_config_data: NeuralNetworkConfigData, genetic_algorithm_config_data: GeneticAlgorithmConfigData
) -> NeuralNetworkGA:
    """Fixture for NeuralNetworkGA."""
    return NeuralNetworkGA.from_config_data(neural_network_config_data, genetic_algorithm_config_data)


# Model fixtures
@pytest.fixture
def mock_config() -> Config:
    """Fixture for Config object."""
    return Config(host="localhost", port=1111)


@pytest.fixture
def mock_load_config(mock_config: Config) -> Generator[MagicMock, None, None]:
    """Patch the load_config function to return the mock config."""
    with patch("nn_websocket.models.config.Config.load_config") as mock_load_config:
        mock_load_config.return_value = mock_config
        yield mock_load_config


# @pytest.fixture
# def mock_neural_network_suite(
#     configuration_data: ConfigurationData,
# ) -> NeuralNetworkSuite:
#     """Fixture for NeuralNetworkSuite."""
#     return NeuralNetworkSuite.from_bytes(ConfigurationData.to_bytes(configuration_data))


# # Main fixtures
# @pytest.fixture
# def mock_neural_network_websocket_server(
#     mock_load_config: MagicMock,
# ) -> NeuralNetworkWebsocketServer:
#     """Fixture for NeuralNetworkWebsocketServer."""
#     return NeuralNetworkWebsocketServer(MOCK_CONFIG_FILEPATH)


# @pytest.fixture
# def mock_configure_neural_networks(
#     mock_neural_network_suite: NeuralNetworkSuite,
# ) -> Generator[MagicMock, None, None]:
#     """Patch the configure_neural_networks function."""
#     with patch("nn_websocket.main.NeuralNetworkWebsocketServer.configure_neural_networks") as mock_configure:
#         mock_configure.return_value = mock_neural_network_suite
#         yield mock_configure


# @pytest.fixture
# def mock_crossover_neural_networks() -> Generator[MagicMock, None, None]:
#     """Patch the crossover_neural_networks function."""
#     with patch("nn_websocket.main.NeuralNetworkWebsocketServer.crossover_neural_networks") as mock_crossover:
#         mock_crossover.return_value = None
#         yield mock_crossover


# @pytest.fixture
# def mock_process_observations() -> Generator[MagicMock, None, None]:
#     """Patch the process_observations function."""
#     with patch("nn_websocket.main.NeuralNetworkWebsocketServer.process_observations") as mock_process:
#         mock_process.return_value = ActionData(
#             outputs=np.arange(MOCK_NUM_OUTPUTS * MOCK_NUM_AGENTS, dtype=float).tolist()
#         )
#         yield mock_process


# @pytest.fixture
# def mock_websocket(
#     nn_config_data: NeuralNetworkConfigData,
#     frame_request_data_observation: FrameRequestData,
#     frame_request_data_population: FrameRequestData,
# ) -> AsyncMock:
#     """Fixture for a mock websocket connection."""
#     mock_websocket = AsyncMock()
#     mock_websocket.__aiter__.return_value = [
#         NeuralNetworkConfigData.to_bytes(nn_config_data),
#         FrameRequestData.to_bytes(frame_request_data_observation),
#         FrameRequestData.to_bytes(frame_request_data_population),
#     ]
#     return mock_websocket
