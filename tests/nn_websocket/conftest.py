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
    ConfigurationData,
    FitnessApproachConfigData,
    GeneticAlgorithmConfigData,
    NeuralNetworkConfigData,
    NeuroevolutionConfigData,
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
def neural_network_config_data(neural_network_config: NeuralNetworkConfig) -> NeuralNetworkConfigData:
    """Fixture for NeuralNetworkConfigData."""
    return NeuralNetworkConfigData(
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
def configuration_neuroevolution(neural_network_config: NeuralNetworkConfig) -> Configuration:
    """Fixture for Configuration with NeuroevolutionConfig."""
    return Configuration(neuroevolution=NeuroevolutionConfig(neural_network=neural_network_config))


@pytest.fixture
def configuration_fitness(neural_network_config: NeuralNetworkConfig) -> Configuration:
    """Fixture for Configuration with FitnessApproachConfig."""
    return Configuration(fitness_approach=FitnessApproachConfig(neural_network=neural_network_config))


@pytest.fixture
def configuration_data_neuroevolution(configuration_neuroevolution: Configuration) -> ConfigurationData:
    """Fixture for ConfigurationData with NeuroevolutionConfig."""
    return ConfigurationData.from_protobuf(configuration_neuroevolution)


@pytest.fixture
def configuration_data_fitness(configuration_fitness: Configuration) -> ConfigurationData:
    """Fixture for ConfigurationData with FitnessApproachConfig."""
    return ConfigurationData.from_protobuf(configuration_fitness)


@pytest.fixture
def genetic_algorithm_config() -> GeneticAlgorithmConfig:
    """Fixture for GeneticAlgorithmConfig."""
    return GeneticAlgorithmConfig(
        population_size=MOCK_POPULATION_SIZE,
        mutation_rate=MOCK_MUTATION_RATE,
    )


@pytest.fixture
def genetic_algorithm_config_data(genetic_algorithm_config: GeneticAlgorithmConfig) -> GeneticAlgorithmConfigData:
    """Fixture for GeneticAlgorithmConfigData."""
    return GeneticAlgorithmConfigData.from_protobuf(genetic_algorithm_config)


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
def neuroevolution_config_data(neuroevolution_config: NeuroevolutionConfig) -> NeuroevolutionConfigData:
    """Fixture for NeuroevolutionConfigData."""
    return NeuroevolutionConfigData.from_protobuf(neuroevolution_config)


@pytest.fixture
def fitness_approach_config(neural_network_config: NeuralNetworkConfig) -> FitnessApproachConfig:
    """Fixture for FitnessApproachConfig."""
    return FitnessApproachConfig(neural_network=neural_network_config)


@pytest.fixture
def fitness_approach_config_data(fitness_approach_config: FitnessApproachConfig) -> FitnessApproachConfigData:
    """Fixture for FitnessApproachConfigData."""
    return FitnessApproachConfigData.from_protobuf(fitness_approach_config)


# FrameRequestClasses_pb2.proto
@pytest.fixture
def frame_request_observation(observation: Observation) -> FrameRequest:
    """Fixture for FrameRequest with Observation."""
    return FrameRequest(observation=observation)


@pytest.fixture
def frame_request_fitness(fitness: Fitness) -> FrameRequest:
    """Fixture for FrameRequest with Fitness."""
    return FrameRequest(fitness=fitness)


@pytest.fixture
def frame_request_train(train_request: TrainRequest) -> FrameRequest:
    """Fixture for FrameRequest with TrainRequest."""
    return FrameRequest(train_request=train_request)


@pytest.fixture
def frame_request_data_observation(frame_request_observation: FrameRequest) -> FrameRequestData:
    """Fixture for FrameRequestData with Observation."""
    return FrameRequestData.from_protobuf(frame_request_observation)


@pytest.fixture
def frame_request_data_fitness(frame_request_fitness: FrameRequest) -> FrameRequestData:
    """Fixture for FrameRequestData with Fitness."""
    return FrameRequestData.from_protobuf(frame_request_fitness)


@pytest.fixture
def frame_request_data_train(frame_request_train: FrameRequest) -> FrameRequestData:
    """Fixture for FrameRequestData with TrainRequest."""
    return FrameRequestData.from_protobuf(frame_request_train)


@pytest.fixture
def observation() -> Observation:
    """Fixture for Observation."""
    return Observation(inputs=rng.uniform(0.0, 1.0, size=MOCK_NUM_INPUTS).tolist())


@pytest.fixture
def observation_data(observation: Observation) -> ObservationData:
    """Fixture for ObservationData."""
    return ObservationData.from_protobuf(observation)


@pytest.fixture
def action() -> Action:
    """Fixture for Action."""
    return Action(outputs=rng.uniform(0.0, 1.0, size=MOCK_NUM_OUTPUTS).tolist())


@pytest.fixture
def action_data(action: Action) -> ActionData:
    """Fixture for ActionData."""
    return ActionData.from_protobuf(action)


@pytest.fixture
def fitness() -> Fitness:
    """Fixture for Fitness."""
    return Fitness(values=rng.uniform(0.0, 1.0, size=MOCK_POPULATION_SIZE).tolist())


@pytest.fixture
def fitness_data(fitness: Fitness) -> FitnessData:
    """Fixture for FitnessData."""
    return FitnessData.from_protobuf(fitness)


@pytest.fixture
def train_request(observation: Observation, fitness: Fitness) -> TrainRequest:
    """Fixture for TrainRequest."""
    return TrainRequest(observation=[observation], fitness=[fitness])


@pytest.fixture
def train_request_data(train_request: TrainRequest) -> TrainRequestData:
    """Fixture for TrainRequestData."""
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


@pytest.fixture
def mock_neuroevolution_suite(neuroevolution_config_data: NeuroevolutionConfigData) -> NeuroevolutionSuite:
    """Fixture for NeuroevolutionSuite."""
    return NeuroevolutionSuite.from_config_data(neuroevolution_config_data)


@pytest.fixture
def mock_fitness_suite(fitness_approach_config_data: FitnessApproachConfigData) -> FitnessSuite:
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
def mock_process_observations(action_data: ActionData) -> Generator[MagicMock, None, None]:
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
    configuration_data_neuroevolution: ConfigurationData,
    frame_request_data_observation: FrameRequestData,
    frame_request_data_fitness: FrameRequestData,
) -> AsyncMock:
    """Fixture for a mock websocket connection."""
    mock_websocket = AsyncMock()
    mock_websocket.__aiter__.return_value = [
        ConfigurationData.to_bytes(configuration_data_neuroevolution),
        FrameRequestData.to_bytes(frame_request_data_observation),
        FrameRequestData.to_bytes(frame_request_data_fitness),
    ]
    return mock_websocket


@pytest.fixture
def mock_websocket_fitness(
    configuration_data_fitness: ConfigurationData,
    frame_request_data_observation: FrameRequestData,
    frame_request_data_train: FrameRequestData,
) -> AsyncMock:
    """Fixture for a mock websocket connection for fitness approach."""
    mock_websocket = AsyncMock()
    mock_websocket.__aiter__.return_value = [
        ConfigurationData.to_bytes(configuration_data_fitness),
        FrameRequestData.to_bytes(frame_request_data_observation),
        FrameRequestData.to_bytes(frame_request_data_train),
    ]
    return mock_websocket


# Base Client fixtures
@pytest.fixture
def mock_base_client(configuration_data_neuroevolution: ConfigurationData, mock_load_config: MagicMock) -> BaseClient:
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
