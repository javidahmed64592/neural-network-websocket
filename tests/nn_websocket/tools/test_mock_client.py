from unittest.mock import AsyncMock, patch

import pytest

from nn_websocket.protobuf.proto_types import (
    FitnessData,
    ObservationData,
)
from nn_websocket.tools.mock_client import SERVER_URI, MockClient, run


class TestMockClient:
    def test_get_random_observation(self) -> None:
        """Test that get_random_observation generates a valid observation."""
        observation = MockClient.get_random_observation()
        assert isinstance(observation.observation, ObservationData)

    def test_get_random_population_fitness(self) -> None:
        """Test that get_random_population_fitness generates valid population fitness data."""
        fitness = MockClient.get_random_population_fitness()
        assert isinstance(fitness.fitness, FitnessData)

    # Unit test for start method
    @pytest.mark.asyncio
    async def test_start(self) -> None:
        """Test that the start method correctly communicates with the WebSocket server."""
        # Mock websocket connection
        mock_ws = AsyncMock()
        mock_ws.recv.return_value = b"mock_response"

        # Create a proper async context manager mock
        cm_mock = AsyncMock()
        cm_mock.__aenter__.return_value = mock_ws
        cm_mock.__aexit__.return_value = None

        with patch("websockets.connect", return_value=cm_mock) as mock_connect:
            # Mock sleep to avoid waiting during test
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                # Exit after a few iterations
                mock_sleep.side_effect = [None, None, KeyboardInterrupt]

                # Run the start method and handle the expected exception
                with pytest.raises(KeyboardInterrupt):
                    await MockClient.start()

                # Verify websockets.connect was called with the correct URI
                mock_connect.assert_called_once_with(SERVER_URI)

                # Verify that configuration data was sent first
                assert mock_ws.send.call_count >= 1
                first_call_args = mock_ws.send.call_args_list[0][0][0]
                assert isinstance(first_call_args, bytes | str)

                # Verify observation data was sent
                assert mock_ws.send.call_count > 1

                # Verify recv was called to get response
                assert mock_ws.recv.called


class TestRunFunction:
    def test_run_success(self) -> None:
        """Test that the run function executes without exceptions."""
        with patch("asyncio.run") as mock_run:
            run()
            mock_run.assert_called_once()

    def test_run_keyboard_interrupt(self) -> None:
        """Test that KeyboardInterrupt is properly handled."""
        with patch("asyncio.run", side_effect=KeyboardInterrupt):
            run()

    def test_run_system_exit(self) -> None:
        """Test that SystemExit is properly handled."""
        with patch("asyncio.run", side_effect=SystemExit):
            run()
