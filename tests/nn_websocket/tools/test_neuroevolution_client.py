"""Unit tests for the src/nn_websocket/tools/neuroevolution_client.py module."""

from unittest.mock import AsyncMock, patch

import pytest

from nn_websocket.protobuf.frame_data_types import FrameRequestDataType
from nn_websocket.tools.neuroevolution_client import (
    CONFIG_DATA,
    NeuroevolutionClient,
    main,
)


@pytest.fixture
def mock_neuroevolution_client() -> NeuroevolutionClient:
    """Fixture for NeuroevolutionClient."""
    return NeuroevolutionClient(CONFIG_DATA)


class TestNeuroevolutionClient:
    """Test cases for NeuroevolutionClient send_observation and send_training methods."""

    @pytest.mark.asyncio
    async def test_send_observation(
        self, mock_neuroevolution_client: NeuroevolutionClient, mock_client_websocket: AsyncMock
    ) -> None:
        """Test that send_observation sends the correct frame data as bytes."""
        with patch("nn_websocket.tools.client_utils.get_random_observation_frame") as mock_get_obs:
            mock_get_obs.return_value = FrameRequestDataType()
            await mock_neuroevolution_client.send_observation(mock_client_websocket)

        mock_client_websocket.send.assert_called_once()
        assert isinstance(mock_client_websocket.send.call_args[0][0], bytes)
        assert mock_client_websocket.method_calls[0][0] == "send"

    @pytest.mark.asyncio
    async def test_send_training(
        self, mock_neuroevolution_client: NeuroevolutionClient, mock_client_websocket: AsyncMock
    ) -> None:
        """Test that send_training sends the correct frame data as bytes."""
        with patch("nn_websocket.tools.client_utils.get_random_fitness_frame") as mock_get_fitness:
            mock_get_fitness.return_value = FrameRequestDataType()
            await mock_neuroevolution_client.send_training(mock_client_websocket)

        mock_client_websocket.send.assert_called_once()
        assert isinstance(mock_client_websocket.send.call_args[0][0], bytes)
        assert mock_client_websocket.method_calls[0][0] == "send"


class TestMainFunction:
    """Test cases for the main function of the neuroevolution client."""

    def test_main(self) -> None:
        """Test that the main function runs without exceptions."""
        with patch("nn_websocket.tools.neuroevolution_client.run") as mock_run:
            main()
            mock_run.assert_called_once()
            assert isinstance(mock_run.call_args[0][0], NeuroevolutionClient)
            assert mock_run.call_args[0][0].config_data == CONFIG_DATA
