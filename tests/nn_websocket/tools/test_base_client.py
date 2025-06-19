"""Unit tests for the src/nn_websocket/tools/base_client.py module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nn_websocket.protobuf.nn_websocket_data_types import ConfigurationData
from nn_websocket.tools.base_client import EPISODE_LENGTH, BaseClient, run


class TestBaseClient:
    """Test cases for BaseClient websocket client logic."""

    def test_init(self, mock_base_client: BaseClient, configuration_data_neuroevolution: ConfigurationData) -> None:
        """Test BaseClient initialization."""
        assert mock_base_client.config_data == configuration_data_neuroevolution

    @pytest.mark.asyncio
    async def test_send_configuration(
        self, mock_base_client: BaseClient, mock_client_websocket: AsyncMock, mock_sleep: MagicMock
    ) -> None:
        """Test that send_configuration sends the correct data."""
        await mock_base_client.send_configuration(mock_client_websocket)

        mock_client_websocket.send.assert_called_once()
        assert isinstance(mock_client_websocket.send.call_args[0][0], bytes)

    @pytest.mark.asyncio
    async def test_send_observation(self, mock_base_client: BaseClient, mock_client_websocket: AsyncMock) -> None:
        """Test send_observation method."""
        await mock_base_client.send_observation(mock_client_websocket)
        mock_client_websocket.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_training(self, mock_base_client: BaseClient, mock_client_websocket: AsyncMock) -> None:
        """Test send_training method."""
        await mock_base_client.send_training(mock_client_websocket)
        mock_client_websocket.send.assert_not_called()

    @pytest.fixture
    def mock_send_configuration(self, mock_base_client: BaseClient) -> AsyncMock:
        """Fixture to mock send_configuration method."""
        return patch.object(mock_base_client, "send_configuration", return_value=None).start()

    @pytest.fixture
    def mock_send_observation(self, mock_base_client: BaseClient) -> AsyncMock:
        """Fixture to mock send_observation method."""
        return patch.object(mock_base_client, "send_observation", return_value=None).start()

    @pytest.fixture
    def mock_send_training(self, mock_base_client: BaseClient) -> AsyncMock:
        """Fixture to mock send_training method."""
        return patch.object(mock_base_client, "send_training", return_value=None).start()

    @pytest.mark.asyncio
    async def test_start(
        self,
        mock_base_client: BaseClient,
        mock_client_websocket: AsyncMock,
        mock_sleep: MagicMock,
        mock_send_configuration: AsyncMock,
        mock_send_observation: AsyncMock,
        mock_send_training: AsyncMock,
    ) -> None:
        """Test that the start method correctly manages the client lifecycle."""
        cm_mock = AsyncMock()
        cm_mock.__aenter__.return_value = mock_client_websocket
        cm_mock.__aexit__.return_value = None

        episodes = [None] * EPISODE_LENGTH

        with patch("websockets.connect", return_value=cm_mock) as mock_connect:
            mock_sleep.side_effect = [*episodes, KeyboardInterrupt]

            with pytest.raises(KeyboardInterrupt):
                await mock_base_client.start()

            mock_connect.assert_called_once()
            mock_send_configuration.assert_called_once()
            assert mock_send_observation.call_count == EPISODE_LENGTH + 1
            mock_send_training.assert_called_once()


class TestRunFunction:
    """Test cases for the run utility function."""

    def test_run_success(self, mock_base_client: BaseClient) -> None:
        """Test that the run function executes without exceptions."""
        with patch("asyncio.run") as mock_run:
            run(mock_base_client)
            mock_run.assert_called_once()

    def test_run_keyboard_interrupt(self, mock_base_client: BaseClient) -> None:
        """Test that KeyboardInterrupt is properly handled."""
        with patch("asyncio.run", side_effect=KeyboardInterrupt):
            run(mock_base_client)

    def test_run_system_exit(self, mock_base_client: BaseClient) -> None:
        """Test that SystemExit is properly handled."""
        with patch("asyncio.run", side_effect=SystemExit):
            run(mock_base_client)
