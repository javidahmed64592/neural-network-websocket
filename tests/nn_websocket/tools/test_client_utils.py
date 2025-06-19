"""Unit tests for the src/nn_websocket/tools/client_utils.py module."""

from unittest.mock import MagicMock

from nn_websocket.models.config import Config
from nn_websocket.protobuf.frame_data_types import FitnessData, FrameRequestDataType, ObservationData, TrainRequestData
from nn_websocket.tools.client_utils import (
    CONFIG_FILEPATH,
    get_config,
    get_random_fitness_frame,
    get_random_observation_frame,
    get_random_train_request_frame,
)


def test_get_config(mock_load_config: MagicMock, mock_config: Config) -> None:
    """Test that the configuration is loaded correctly from file."""
    config = get_config()
    mock_load_config.assert_called_once_with(CONFIG_FILEPATH)
    assert config == mock_config


def test_get_random_observation_frame() -> None:
    """Test that a random observation frame is generated correctly."""
    num_inputs = 5
    frame = get_random_observation_frame(num_inputs)

    assert isinstance(frame, FrameRequestDataType)
    assert isinstance(frame.observation, ObservationData)
    assert len(frame.observation.inputs) == num_inputs


def test_get_random_fitness_frame() -> None:
    """Test that a random fitness frame is generated correctly."""
    num_agents = 10
    frame = get_random_fitness_frame(num_agents)

    assert isinstance(frame, FrameRequestDataType)
    assert isinstance(frame.fitness, FitnessData)
    assert len(frame.fitness.values) == num_agents


def test_get_random_train_request_frame() -> None:
    """Test that a random train request frame is generated correctly."""
    batch_size = 5
    num_inputs = 3
    frame = get_random_train_request_frame(batch_size, num_inputs)

    assert isinstance(frame, FrameRequestDataType)
    assert isinstance(frame.train_request, TrainRequestData)
    assert len(frame.train_request.observation) == batch_size
    assert len(frame.train_request.fitness) == batch_size

    for observation in frame.train_request.observation:
        assert isinstance(observation, ObservationData)
        assert len(observation.inputs) == num_inputs

    for fitness in frame.train_request.fitness:
        assert isinstance(fitness, FitnessData)
        assert len(fitness.values) == 1
