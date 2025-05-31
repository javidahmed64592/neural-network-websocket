import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from nn_websocket.protobuf.proto_types import ActionData, NeuralNetworkConfigData, ObservationData
from nn_websocket.tools.mock_client import mock_client, run


class TestMockClient:
    # Minimum expected message count: config + at least one observation
    MIN_EXPECTED_MESSAGES = 2

    @pytest.mark.asyncio
    async def test_mock_client_config_send(self, action_data: ActionData) -> None:
        """Test that the mock client sends the configuration data correctly."""
        mock_ws = AsyncMock()

        # Create a mock response for the websocket recv
        mock_ws.recv.return_value = ActionData.to_bytes(action_data)

        # Fix: Use a context manager mock that returns our mock_ws
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_ws

        with patch("websockets.connect", return_value=mock_cm):
            # Run the mock client with a controlled number of iterations
            with patch("asyncio.sleep", side_effect=[None, None, asyncio.CancelledError]):
                with pytest.raises(asyncio.CancelledError):
                    await mock_client()

                # Check that config was sent
                assert mock_ws.send.call_count >= 1
                # The first call should be the config data
                config_bytes = mock_ws.send.call_args_list[0][0][0]
                assert isinstance(config_bytes, bytes)
                # Verify we can decode it as a NeuralNetworkConfigData
                decoded_config = NeuralNetworkConfigData.from_bytes(config_bytes)
                assert decoded_config.num_inputs > 0
                assert decoded_config.num_outputs > 0

    @pytest.mark.asyncio
    async def test_mock_client_observation_send(self, action_data: ActionData) -> None:
        """Test that the mock client sends observation data and processes responses."""
        mock_ws = AsyncMock()

        # Create a mock response for the websocket recv
        mock_ws.recv.return_value = ActionData.to_bytes(action_data)

        # Fix: Use a context manager mock that returns our mock_ws
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_ws

        with patch("websockets.connect", return_value=mock_cm):
            # Run the mock client with a controlled number of iterations
            with patch("asyncio.sleep", side_effect=[None, None, asyncio.CancelledError]):
                with pytest.raises(asyncio.CancelledError):
                    await mock_client()

                # Check that at least MIN_EXPECTED_MESSAGES messages were sent (config + observation)
                assert mock_ws.send.call_count >= self.MIN_EXPECTED_MESSAGES

                # Verify the second call was an observation
                if mock_ws.send.call_count >= self.MIN_EXPECTED_MESSAGES:
                    obs_bytes = mock_ws.send.call_args_list[1][0][0]
                    assert isinstance(obs_bytes, bytes)
                    # Verify we can decode it as ObservationData
                    observation = ObservationData.from_bytes(obs_bytes)
                    assert len(observation.inputs) > 0

    @pytest.mark.asyncio
    async def test_mock_client_timeout_handling(self) -> None:
        """Test that the mock client properly handles timeouts."""
        mock_ws = AsyncMock()

        # Make recv throw a TimeoutError
        mock_ws.recv.side_effect = asyncio.TimeoutError

        # Fix: Use a context manager mock that returns our mock_ws
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_ws

        with patch("websockets.connect", return_value=mock_cm):
            # Run the mock client with a controlled number of iterations
            with patch("asyncio.sleep", side_effect=[None, None, asyncio.CancelledError]):
                with pytest.raises(asyncio.CancelledError):
                    await mock_client()

                # Should continue even after timeout
                assert mock_ws.send.call_count >= self.MIN_EXPECTED_MESSAGES

    def test_run_function(self) -> None:
        """Test that the run function properly calls asyncio.run with mock_client."""
        with patch("asyncio.run") as mock_run:
            with patch("nn_websocket.tools.mock_client.mock_client"):
                run()
                # Check that asyncio.run was called once
                mock_run.assert_called_once()

    def test_run_keyboard_interrupt(self) -> None:
        """Test that KeyboardInterrupt is properly handled."""
        with patch("asyncio.run", side_effect=KeyboardInterrupt):
            # Should not raise an exception
            run()

    def test_run_system_exit(self) -> None:
        """Test that SystemExit is properly handled."""
        with patch("asyncio.run", side_effect=SystemExit):
            # Should not raise an exception
            run()
