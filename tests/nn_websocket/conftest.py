from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from nn_websocket.ga.nn_ga import NeuralNetworkGA
from nn_websocket.ga.nn_member import NeuralNetworkMember
from nn_websocket.main import NeuralNetworkWebsocketServer
from nn_websocket.models.config import Config
from nn_websocket.models.nn_suite import NeuralNetworkSuite
from nn_websocket.protobuf.proto_types import (
    ActionData,
    ActivationFunctionEnum,
    ConfigurationData,
    FitnessData,
    FrameRequestData,
    GeneticAlgorithmConfigData,
    NeuralNetworkConfigData,
    ObservationData,
    TrainRequestData,
)

# As an example, we will assume 10 agents with 5 inputs and 2 outputs each.
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
MOCK_NUM_AGENTS = 10
MOCK_MUTATION_RATE = 0.1


# Protobuf fixtures
@pytest.fixture
def configuration_data(
    nn_config_data: NeuralNetworkConfigData, ga_config_data: GeneticAlgorithmConfigData
) -> ConfigurationData:
    """Fixture for ConfigurationData."""
    return ConfigurationData(
        neural_network=nn_config_data,
        genetic_algorithm=ga_config_data,
    )


@pytest.fixture
def nn_config_data() -> NeuralNetworkConfigData:
    """Fixture for NeuralNetworkConfigData."""
    return NeuralNetworkConfigData(
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
    )


@pytest.fixture
def ga_config_data() -> GeneticAlgorithmConfigData:
    """Fixture for GeneticAlgorithmConfigData."""
    return GeneticAlgorithmConfigData(
        population_size=MOCK_NUM_AGENTS,
        mutation_rate=MOCK_MUTATION_RATE,
    )


@pytest.fixture
def frame_request_data_observation(
    observation_data: ObservationData,
) -> FrameRequestData:
    """Fixture for FrameRequestData with observation data."""
    return FrameRequestData(
        observation=observation_data,
    )


@pytest.fixture
def frame_request_data_population(
    fitness_data: FitnessData,
) -> FrameRequestData:
    """Fixture for FrameRequestData."""
    return FrameRequestData(
        population_fitness=fitness_data,
    )


@pytest.fixture
def frame_request_data_train(
    train_request_data: TrainRequestData,
) -> FrameRequestData:
    """Fixture for FrameRequestData with training data."""
    return FrameRequestData(
        train=train_request_data,
    )


@pytest.fixture
def train_request_data(
    observation_data: ObservationData,
    action_data: ActionData,
    fitness_data: FitnessData,
) -> TrainRequestData:
    """Fixture for TrainRequestData."""
    return TrainRequestData(
        observation=observation_data,
        action=action_data,
        fitness=fitness_data,
    )


@pytest.fixture
def observation_data() -> ObservationData:
    """Fixture for ObservationData."""
    return ObservationData(
        inputs=np.arange(MOCK_NUM_INPUTS * MOCK_NUM_AGENTS, dtype=np.float32).tolist(),
    )


@pytest.fixture
def action_data() -> ActionData:
    """Fixture for ActionData."""
    return ActionData(
        outputs=np.arange(MOCK_NUM_OUTPUTS * MOCK_NUM_AGENTS, dtype=np.float32).tolist(),
    )


@pytest.fixture
def fitness_data() -> FitnessData:
    """Fixture for FitnessData."""
    return FitnessData(
        values=np.arange(MOCK_NUM_AGENTS, dtype=float).tolist(),
    )


# Genetic Algorithm fixtures
@pytest.fixture
def mock_neural_network_member(nn_config_data: NeuralNetworkConfigData) -> NeuralNetworkMember:
    """Fixture for NeuralNetworkMember."""
    return NeuralNetworkMember.from_config_data(nn_config_data)


@pytest.fixture
def mock_neural_network_ga(
    nn_config_data: NeuralNetworkConfigData, ga_config_data: GeneticAlgorithmConfigData
) -> NeuralNetworkGA:
    """Fixture for NeuralNetworkGA."""
    return NeuralNetworkGA.from_config_data(nn_config_data, ga_config_data)


# Model fixtures
@pytest.fixture
def mock_config() -> Config:
    """Fixture for Config object."""
    return Config(host="localhost", port=8765)


@pytest.fixture
def mock_load_config(mock_config: Config) -> Generator[MagicMock, None, None]:
    """Patch the load_config function to return the mock config."""
    with patch("nn_websocket.models.config.Config.load_config") as mock_load_config:
        mock_load_config.return_value = mock_config
        yield mock_load_config


@pytest.fixture
def mock_neural_network_suite(
    configuration_data: ConfigurationData,
) -> NeuralNetworkSuite:
    """Fixture for NeuralNetworkSuite."""
    return NeuralNetworkSuite.from_bytes(ConfigurationData.to_bytes(configuration_data))


# Main fixtures
@pytest.fixture
def mock_neural_network_websocket_server(
    mock_load_config: MagicMock,
) -> NeuralNetworkWebsocketServer:
    """Fixture for NeuralNetworkWebsocketServer."""
    return NeuralNetworkWebsocketServer(MOCK_CONFIG_FILEPATH)


@pytest.fixture
def mock_configure_neural_networks(
    mock_neural_network_suite: NeuralNetworkSuite,
) -> Generator[MagicMock, None, None]:
    """Patch the configure_neural_networks function."""
    with patch("nn_websocket.main.NeuralNetworkWebsocketServer.configure_neural_networks") as mock_configure:
        mock_configure.return_value = mock_neural_network_suite
        yield mock_configure


@pytest.fixture
def mock_crossover_neural_networks() -> Generator[MagicMock, None, None]:
    """Patch the crossover_neural_networks function."""
    with patch("nn_websocket.main.NeuralNetworkWebsocketServer.crossover_neural_networks") as mock_crossover:
        mock_crossover.return_value = None
        yield mock_crossover


@pytest.fixture
def mock_process_observations() -> Generator[MagicMock, None, None]:
    """Patch the process_observations function."""
    with patch("nn_websocket.main.NeuralNetworkWebsocketServer.process_observations") as mock_process:
        mock_process.return_value = ActionData(
            outputs=np.arange(MOCK_NUM_OUTPUTS * MOCK_NUM_AGENTS, dtype=float).tolist()
        )
        yield mock_process


@pytest.fixture
def mock_websocket(
    nn_config_data: NeuralNetworkConfigData,
    frame_request_data_observation: FrameRequestData,
    frame_request_data_population: FrameRequestData,
) -> AsyncMock:
    """Fixture for a mock websocket connection."""
    mock_websocket = AsyncMock()
    mock_websocket.__aiter__.return_value = [
        NeuralNetworkConfigData.to_bytes(nn_config_data),
        FrameRequestData.to_bytes(frame_request_data_observation),
        FrameRequestData.to_bytes(frame_request_data_population),
    ]
    return mock_websocket
