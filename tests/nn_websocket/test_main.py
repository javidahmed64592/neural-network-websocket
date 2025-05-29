import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from nn_websocket.main import (
    configure_neural_networks,
    handle_connection,
    load_config,
    main,
    process_observations,
    run,
)
from nn_websocket.models.config import Config
from nn_websocket.protobuf.proto_types import ActionData, NeuralNetworkConfigData, ObservationData


class TestLoadConfig:
    def test_load_config(self) -> None:
        """Test that the configuration is loaded correctly from a file."""
        mock_host = "test_host"
        mock_port = 9999
        mock_config = {"host": mock_host, "port": mock_port}

        # Use builtins.open rather than trying to patch Path.open
        with patch("pathlib.Path.open", mock_open(read_data=json.dumps(mock_config))):
            config = load_config()

            assert isinstance(config, Config)
            assert config.host == mock_host
            assert config.port == mock_port


class TestConfigureNeuralNetworks:
    def test_configure_neural_networks(self, nn_config_data: NeuralNetworkConfigData) -> None:
        """Test that the neural networks are configured correctly."""
        config_bytes = NeuralNetworkConfigData.to_bytes(nn_config_data)

        with patch("nn_websocket.main.neural_network_suite") as mock_suite:
            configure_neural_networks(config_bytes)
            mock_suite.set_networks_from_bytes.assert_called_once_with(config_bytes)


class TestProcessObservations:
    @pytest.mark.asyncio
    async def test_process_observations(self, observation_data: ObservationData, action_data: ActionData) -> None:
        """Test that observations are processed correctly and actions are sent back."""
        observation_bytes = ObservationData.to_bytes(observation_data)

        mock_websocket = AsyncMock()

        with patch("nn_websocket.main.neural_network_suite") as mock_suite:
            mock_suite.feedforward_through_networks_from_bytes.return_value = action_data

            await process_observations(mock_websocket, observation_bytes)

            mock_suite.feedforward_through_networks_from_bytes.assert_called_once_with(observation_bytes)
            mock_websocket.send.assert_awaited_once()
            # Check the argument is bytes (can't directly compare the actual bytes due to message wrapping)
            assert isinstance(mock_websocket.send.call_args[0][0], bytes)


class TestHandleConnection:
    @pytest.mark.asyncio
    async def test_handle_connection_first_message(self, nn_config_data: NeuralNetworkConfigData) -> None:
        """Test that the first message configures the neural networks."""
        config_bytes = NeuralNetworkConfigData.to_bytes(nn_config_data)

        mock_websocket = AsyncMock()
        mock_websocket.__aiter__.return_value = [config_bytes]

        with patch("nn_websocket.main.configure_neural_networks") as mock_configure:
            with patch("nn_websocket.main.process_observations"):
                await handle_connection(mock_websocket)

                mock_configure.assert_called_once_with(config_bytes)

    @pytest.mark.asyncio
    async def test_handle_connection_subsequent_messages(
        self, nn_config_data: NeuralNetworkConfigData, observation_data: ObservationData
    ) -> None:
        """Test that subsequent messages process observations."""
        config_bytes = NeuralNetworkConfigData.to_bytes(nn_config_data)
        observation_bytes = ObservationData.to_bytes(observation_data)

        mock_websocket = AsyncMock()
        mock_websocket.__aiter__.return_value = [config_bytes, observation_bytes]

        with patch("nn_websocket.main.configure_neural_networks"):
            with patch("nn_websocket.main.process_observations") as mock_process:
                await handle_connection(mock_websocket)

                mock_process.assert_called_once_with(mock_websocket, observation_bytes)

    @pytest.mark.asyncio
    async def test_handle_connection_string_message(self) -> None:
        """Test that string messages are properly encoded to bytes."""
        string_message = "test_message"

        mock_websocket = AsyncMock()
        mock_websocket.__aiter__.return_value = [string_message]

        with patch("nn_websocket.main.configure_neural_networks") as mock_configure:
            with patch("nn_websocket.main.process_observations"):
                await handle_connection(mock_websocket)

                mock_configure.assert_called_once_with(string_message.encode("utf-8"))


class TestMain:
    @pytest.mark.asyncio
    async def test_main(self, mock_config: Config, mock_load_config: MagicMock) -> None:
        """Test that the main function sets up the websocket server correctly."""
        with patch("websockets.serve") as mock_serve:
            mock_serve.return_value.__aenter__.return_value = None
            mock_serve.return_value.__aexit__.return_value = None

            # Create a real Future object that can be awaited
            future = asyncio.Future()
            future.set_result(None)

            with patch("asyncio.Future", return_value=future):
                await main()

                mock_serve.assert_called_once_with(handle_connection, mock_config.host, mock_config.port)

    def test_run(self) -> None:
        """Test that the run function properly calls asyncio.run with main."""
        with patch("asyncio.run") as mock_run:
            with patch("nn_websocket.main.main"):
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
