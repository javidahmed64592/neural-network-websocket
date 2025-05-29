import numpy as np
import pytest

from nn_websocket.protobuf.proto_types import (
    ActionData,
    ActivationFunctionEnum,
    NeuralNetworkConfigData,
    ObservationData,
)

# Protobuf fixtures
# As an example, we will assume 10 agents with 5 inputs and 2 outputs each.
MOCK_NUM_AGENTS = 10
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


@pytest.fixture
def nn_config_data() -> NeuralNetworkConfigData:
    """Fixture for NeuralNetworkConfigData."""
    return NeuralNetworkConfigData(
        num_networks=MOCK_NUM_AGENTS,
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
