"""Fixtures for testing nn_websocket."""

from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from neural_network.protobuf.neural_network_types import (
    ActivationFunctionEnum,
    LearningRateMethodEnum,
    LearningRateSchedulerDataType,
    OptimizerDataType,
    SGDOptimizerDataType,
)

from nn_websocket.ga.nn_ga import NeuralNetworkGA
from nn_websocket.ga.nn_member import NeuralNetworkMember
from nn_websocket.main import NeuralNetworkWebsocketServer
from nn_websocket.models.config import Config
from nn_websocket.models.nn_suites import FitnessSuite, NeuroevolutionSuite
from nn_websocket.protobuf.compiled.FrameData_pb2 import (
    Action,
    Fitness,
    FrameRequestData,
    Observation,
    TrainRequest,
)
from nn_websocket.protobuf.compiled.NNWebsocketData_pb2 import (
    ConfigData,
    FitnessApproachConfig,
    GeneticAlgorithmConfig,
    NeuralNetworkConfig,
    NeuroevolutionConfig,
)
from nn_websocket.protobuf.frame_data_types import (
    ActionType,
    FitnessType,
    FrameRequestDataType,
    ObservationType,
    TrainRequestType,
)
from nn_websocket.protobuf.nn_websocket_data_types import (
    ConfigDataType,
    FitnessApproachConfigType,
    GeneticAlgorithmConfigType,
    NeuralNetworkConfigType,
    NeuroevolutionConfigType,
)
from nn_websocket.tools.base_client import BaseClient

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
MOCK_INPUT_ACTIVATION = ActivationFunctionEnum.LINEAR
MOCK_HIDDEN_ACTIVATION = ActivationFunctionEnum.RELU
MOCK_OUTPUT_ACTIVATION = ActivationFunctionEnum.SIGMOID
MOCK_LEARNING_RATE = 0.01
MOCK_DECAY_RATE = 0.1
MOCK_DECAY_STEPS = 1000
MOCK_LEARNING_RATE_METHOD = LearningRateMethodEnum.STEP_DECAY
MOCK_OPTIMIZER = OptimizerDataType(
    adam=None,
    sgd=SGDOptimizerDataType(learning_rate=MOCK_LEARNING_RATE),
    learning_rate_scheduler=LearningRateSchedulerDataType(
        decay_rate=MOCK_DECAY_RATE, decay_steps=MOCK_DECAY_STEPS, method=MOCK_LEARNING_RATE_METHOD
    ),
)

MOCK_POPULATION_SIZE = 10
MOCK_MUTATION_RATE = 0.1


# NeuralNetwork.proto
@pytest.fixture
def configuration_neuroevolution(neural_network_config: NeuralNetworkConfig) -> ConfigData:
    """Fixture for ConfigData with NeuroevolutionConfig."""
    return ConfigData(neuroevolution=NeuroevolutionConfig(neural_network=neural_network_config))


@pytest.fixture
def configuration_fitness(neural_network_config: NeuralNetworkConfig) -> ConfigData:
    """Fixture for ConfigData with FitnessApproachConfig."""
    return ConfigData(fitness_approach=FitnessApproachConfig(neural_network=neural_network_config))


@pytest.fixture
def configuration_data_neuroevolution(configuration_neuroevolution: ConfigData) -> ConfigDataType:
    """Fixture for ConfigDataType with NeuroevolutionConfig."""
    return ConfigDataType.from_protobuf(configuration_neuroevolution)


@pytest.fixture
def configuration_data_fitness(configuration_fitness: ConfigData) -> ConfigDataType:
    """Fixture for ConfigDataType with FitnessApproachConfig."""
    return ConfigDataType.from_protobuf(configuration_fitness)


@pytest.fixture
def neural_network_config() -> NeuralNetworkConfig:
    """Fixture for NeuralNetworkConfig."""
    return NeuralNetworkConfig(
        num_inputs=MOCK_NUM_INPUTS,
        num_outputs=MOCK_NUM_OUTPUTS,
        hidden_layer_sizes=MOCK_HIDDEN_LAYER_SIZES,
        weights_min=MOCK_WEIGHTS_MIN,
        weights_max=MOCK_WEIGHTS_MAX,
        bias_min=MOCK_BIAS_MIN,
        bias_max=MOCK_BIAS_MAX,
        input_activation=ActivationFunctionEnum.to_protobuf(MOCK_INPUT_ACTIVATION),
        hidden_activation=ActivationFunctionEnum.to_protobuf(MOCK_HIDDEN_ACTIVATION),
        output_activation=ActivationFunctionEnum.to_protobuf(MOCK_OUTPUT_ACTIVATION),
        optimizer=OptimizerDataType.to_protobuf(MOCK_OPTIMIZER),
    )


@pytest.fixture
def neural_network_config_data(neural_network_config: NeuralNetworkConfig) -> NeuralNetworkConfigType:
    """Fixture for NeuralNetworkConfigType."""
    return NeuralNetworkConfigType(
        num_inputs=neural_network_config.num_inputs,
        num_outputs=neural_network_config.num_outputs,
        hidden_layer_sizes=list(neural_network_config.hidden_layer_sizes),
        weights_min=neural_network_config.weights_min,
        weights_max=neural_network_config.weights_max,
        bias_min=neural_network_config.bias_min,
        bias_max=neural_network_config.bias_max,
        input_activation=MOCK_INPUT_ACTIVATION,
        hidden_activation=MOCK_HIDDEN_ACTIVATION,
        output_activation=MOCK_OUTPUT_ACTIVATION,
        optimizer=MOCK_OPTIMIZER,
    )


@pytest.fixture
def genetic_algorithm_config() -> GeneticAlgorithmConfig:
    """Fixture for GeneticAlgorithmConfig."""
    return GeneticAlgorithmConfig(
        population_size=MOCK_POPULATION_SIZE,
        mutation_rate=MOCK_MUTATION_RATE,
    )


@pytest.fixture
def genetic_algorithm_config_data(genetic_algorithm_config: GeneticAlgorithmConfig) -> GeneticAlgorithmConfigType:
    """Fixture for GeneticAlgorithmConfigType."""
    return GeneticAlgorithmConfigType.from_protobuf(genetic_algorithm_config)


@pytest.fixture
def neuroevolution_config(
    neural_network_config: NeuralNetworkConfig, genetic_algorithm_config: GeneticAlgorithmConfig
) -> NeuroevolutionConfig:
    """Fixture for NeuroevolutionConfig."""
    return NeuroevolutionConfig(
        neural_network=neural_network_config,
        genetic_algorithm=genetic_algorithm_config,
    )


@pytest.fixture
def neuroevolution_config_data(neuroevolution_config: NeuroevolutionConfig) -> NeuroevolutionConfigType:
    """Fixture for NeuroevolutionConfigType."""
    return NeuroevolutionConfigType.from_protobuf(neuroevolution_config)


@pytest.fixture
def fitness_approach_config(neural_network_config: NeuralNetworkConfig) -> FitnessApproachConfig:
    """Fixture for FitnessApproachConfig."""
    return FitnessApproachConfig(neural_network=neural_network_config)


@pytest.fixture
def fitness_approach_config_data(fitness_approach_config: FitnessApproachConfig) -> FitnessApproachConfigType:
    """Fixture for FitnessApproachConfigType."""
    return FitnessApproachConfigType.from_protobuf(fitness_approach_config)


# FrameRequestClasses_pb2.proto
@pytest.fixture
def frame_request_observation(observation: Observation) -> FrameRequestData:
    """Fixture for FrameRequestData with Observation."""
    return FrameRequestData(observation=observation)


@pytest.fixture
def frame_request_fitness(fitness: Fitness) -> FrameRequestData:
    """Fixture for FrameRequestData with Fitness."""
    return FrameRequestData(fitness=fitness)


@pytest.fixture
def frame_request_train(train_request: TrainRequest) -> FrameRequestData:
    """Fixture for FrameRequestData with TrainRequest."""
    return FrameRequestData(train_request=train_request)


@pytest.fixture
def frame_request_data_observation(frame_request_observation: FrameRequestData) -> FrameRequestDataType:
    """Fixture for FrameRequestDataType with Observation."""
    return FrameRequestDataType.from_protobuf(frame_request_observation)


@pytest.fixture
def frame_request_data_fitness(frame_request_fitness: FrameRequestData) -> FrameRequestDataType:
    """Fixture for FrameRequestDataType with Fitness."""
    return FrameRequestDataType.from_protobuf(frame_request_fitness)


@pytest.fixture
def frame_request_data_train(frame_request_train: FrameRequestData) -> FrameRequestDataType:
    """Fixture for FrameRequestDataType with TrainRequest."""
    return FrameRequestDataType.from_protobuf(frame_request_train)


@pytest.fixture
def observation() -> Observation:
    """Fixture for Observation."""
    return Observation(inputs=rng.uniform(0.0, 1.0, size=MOCK_NUM_INPUTS).tolist())


@pytest.fixture
def observation_data(observation: Observation) -> ObservationType:
    """Fixture for ObservationType."""
    return ObservationType.from_protobuf(observation)


@pytest.fixture
def action() -> Action:
    """Fixture for Action."""
    return Action(outputs=rng.uniform(0.0, 1.0, size=MOCK_NUM_OUTPUTS).tolist())


@pytest.fixture
def action_data(action: Action) -> ActionType:
    """Fixture for ActionType."""
    return ActionType.from_protobuf(action)


@pytest.fixture
def fitness() -> Fitness:
    """Fixture for Fitness."""
    return Fitness(values=rng.uniform(0.0, 1.0, size=MOCK_POPULATION_SIZE).tolist())


@pytest.fixture
def fitness_data(fitness: Fitness) -> FitnessType:
    """Fixture for FitnessType."""
    return FitnessType.from_protobuf(fitness)


@pytest.fixture
def train_request(observation: Observation, fitness: Fitness) -> TrainRequest:
    """Fixture for TrainRequest."""
    return TrainRequest(observation=[observation], fitness=[fitness])


@pytest.fixture
def train_request_data(train_request: TrainRequest) -> TrainRequestType:
    """Fixture for TrainRequestType."""
    return TrainRequestType.from_protobuf(train_request)


# Genetic Algorithm fixtures
@pytest.fixture
def mock_neural_network_member(neural_network_config_data: NeuralNetworkConfigType) -> NeuralNetworkMember:
    """Fixture for NeuralNetworkMember."""
    return NeuralNetworkMember.from_config_data(neural_network_config_data)


@pytest.fixture
def mock_neural_network_ga(
    neural_network_config_data: NeuralNetworkConfigType, genetic_algorithm_config_data: GeneticAlgorithmConfigType
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


@pytest.fixture
def mock_neuroevolution_suite(neuroevolution_config_data: NeuroevolutionConfigType) -> NeuroevolutionSuite:
    """Fixture for NeuroevolutionSuite."""
    return NeuroevolutionSuite.from_config_data(neuroevolution_config_data)


@pytest.fixture
def mock_fitness_suite(fitness_approach_config_data: FitnessApproachConfigType) -> FitnessSuite:
    """Fixture for FitnessSuite."""
    return FitnessSuite.from_config_data(fitness_approach_config_data)


# Main fixtures
@pytest.fixture
def mock_neural_network_websocket_server(
    mock_load_config: MagicMock,
) -> NeuralNetworkWebsocketServer:
    """Fixture for NeuralNetworkWebsocketServer."""
    return NeuralNetworkWebsocketServer(MOCK_CONFIG_FILEPATH)


@pytest.fixture
def mock_configure_neural_networks_neuroevolution(
    mock_neuroevolution_suite: NeuroevolutionSuite,
) -> Generator[MagicMock, None, None]:
    """Patch the configure_neural_network_suite function."""
    with patch("nn_websocket.main.NeuralNetworkWebsocketServer.configure_neural_network_suite") as mock_configure:
        mock_configure.return_value = mock_neuroevolution_suite
        yield mock_configure


@pytest.fixture
def mock_configure_neural_networks_fitness(
    mock_fitness_suite: FitnessSuite,
) -> Generator[MagicMock, None, None]:
    """Patch the configure_neural_network_suite function for fitness approach."""
    with patch("nn_websocket.main.NeuralNetworkWebsocketServer.configure_neural_network_suite") as mock_configure:
        mock_configure.return_value = mock_fitness_suite
        yield mock_configure


@pytest.fixture
def mock_process_observations(action_data: ActionType) -> Generator[MagicMock, None, None]:
    """Patch the process_observations function."""
    with patch("nn_websocket.main.NeuralNetworkWebsocketServer.process_observations") as mock_process:
        mock_process.return_value = action_data
        yield mock_process


@pytest.fixture
def mock_crossover_neural_networks() -> Generator[MagicMock, None, None]:
    """Patch the crossover_neural_networks function."""
    with patch("nn_websocket.main.NeuralNetworkWebsocketServer.crossover_neural_networks") as mock_crossover:
        mock_crossover.return_value = None
        yield mock_crossover


@pytest.fixture
def mock_train_neural_network() -> Generator[MagicMock, None, None]:
    """Patch the train function."""
    with patch("nn_websocket.main.NeuralNetworkWebsocketServer.train") as mock_train:
        mock_train.return_value = None
        yield mock_train


@pytest.fixture
def mock_websocket_neuroevolution(
    configuration_data_neuroevolution: ConfigDataType,
    frame_request_data_observation: FrameRequestDataType,
    frame_request_data_fitness: FrameRequestDataType,
) -> AsyncMock:
    """Fixture for a mock websocket connection."""
    mock_websocket = AsyncMock()
    mock_websocket.__aiter__.return_value = [
        ConfigDataType.to_bytes(configuration_data_neuroevolution),
        FrameRequestDataType.to_bytes(frame_request_data_observation),
        FrameRequestDataType.to_bytes(frame_request_data_fitness),
    ]
    return mock_websocket


@pytest.fixture
def mock_websocket_fitness(
    configuration_data_fitness: ConfigDataType,
    frame_request_data_observation: FrameRequestDataType,
    frame_request_data_train: FrameRequestDataType,
) -> AsyncMock:
    """Fixture for a mock websocket connection for fitness approach."""
    mock_websocket = AsyncMock()
    mock_websocket.__aiter__.return_value = [
        ConfigDataType.to_bytes(configuration_data_fitness),
        FrameRequestDataType.to_bytes(frame_request_data_observation),
        FrameRequestDataType.to_bytes(frame_request_data_train),
    ]
    return mock_websocket


# Base Client fixtures
@pytest.fixture
def mock_base_client(configuration_data_neuroevolution: ConfigDataType, mock_load_config: MagicMock) -> BaseClient:
    """Fixture for BaseClient."""
    return BaseClient(configuration_data_neuroevolution)


@pytest.fixture
def mock_client_websocket() -> AsyncMock:
    """Fixture for a mock websocket connection for client testing."""
    mock_websocket = AsyncMock()
    mock_websocket.recv.return_value = b"mock_response"
    return mock_websocket


@pytest.fixture
def mock_sleep() -> Generator[AsyncMock, None, None]:
    """Fixture for mocking asyncio.sleep."""
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        yield mock_sleep
