from unittest.mock import AsyncMock, patch

import pytest

from nn_websocket.protobuf.frame_data_types import FrameRequestData
from nn_websocket.tools.fitness_client import (
    CONFIG_DATA,
    FitnessClient,
    main,
)


@pytest.fixture
def mock_fitness_client() -> FitnessClient:
    """Fixture for FitnessClient."""
    return FitnessClient(CONFIG_DATA)


class TestFitnessClient:
    @pytest.mark.asyncio
    async def test_send_observation(self, mock_fitness_client: FitnessClient, mock_client_websocket: AsyncMock) -> None:
        """Test that send_observation sends the correct frame data."""
        with patch("nn_websocket.tools.client_utils.get_random_observation_frame") as mock_get_obs:
            mock_get_obs.return_value = FrameRequestData()
            await mock_fitness_client.send_observation(mock_client_websocket)

        mock_client_websocket.send.assert_called_once()
        assert isinstance(mock_client_websocket.send.call_args[0][0], bytes)
        assert mock_client_websocket.method_calls[0][0] == "send"

    @pytest.mark.asyncio
    async def test_send_training(self, mock_fitness_client: FitnessClient, mock_client_websocket: AsyncMock) -> None:
        """Test that send_training sends the correct frame data."""
        with patch("nn_websocket.tools.client_utils.get_random_train_request_frame") as mock_get_train:
            mock_get_train.return_value = FrameRequestData()
            await mock_fitness_client.send_training(mock_client_websocket)

        mock_client_websocket.send.assert_called_once()
        assert isinstance(mock_client_websocket.send.call_args[0][0], bytes)
        assert mock_client_websocket.method_calls[0][0] == "send"


class TestMainFunction:
    def test_main(self) -> None:
        """Test that the main function runs without exceptions."""
        with patch("nn_websocket.tools.fitness_client.run") as mock_run:
            main()
            mock_run.assert_called_once()
            assert isinstance(mock_run.call_args[0][0], FitnessClient)
            assert mock_run.call_args[0][0].config_data == CONFIG_DATA
