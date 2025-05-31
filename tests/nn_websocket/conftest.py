from collections.abc import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nn_websocket.ga.nn_ga import NeuralNetworkGA
from nn_websocket.ga.nn_member import NeuralNetworkMember
from nn_websocket.models.config import Config
from nn_websocket.protobuf.proto_types import (
    ActionData,
    ActivationFunctionEnum,
    ConfigurationData,
    GeneticAlgorithmConfigData,
    NeuralNetworkConfigData,
    ObservationData,
    PopulationFitnessData,
)

# As an example, we will assume 10 agents with 5 inputs and 2 outputs each.
MOCK_NUM_AGENTS = 10
MOCK_MUTATION_RATE = 0.1
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


# Config fixtures
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


# Protobuf fixtures
@pytest.fixture
def ga_config_data() -> GeneticAlgorithmConfigData:
    """Fixture for GeneticAlgorithmConfigData."""
    return GeneticAlgorithmConfigData(
        population_size=MOCK_NUM_AGENTS,
        mutation_rate=MOCK_MUTATION_RATE,
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
def configuration_data(
    ga_config_data: GeneticAlgorithmConfigData, nn_config_data: NeuralNetworkConfigData
) -> ConfigurationData:
    """Fixture for ConfigurationData."""
    return ConfigurationData(
        genetic_algorithm=ga_config_data,
        neural_network=nn_config_data,
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
def population_fitness_data() -> PopulationFitnessData:
    """Fixture for PopulationFitnessData."""
    return PopulationFitnessData(
        fitness=np.arange(MOCK_NUM_AGENTS).tolist(),
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
