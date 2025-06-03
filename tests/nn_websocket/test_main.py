import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nn_websocket.main import NeuralNetworkWebsocketServer, run
from nn_websocket.models.config import Config
from nn_websocket.models.nn_suite import NeuralNetworkSuite
from nn_websocket.protobuf.proto_types import (
    ActionData,
    FitnessData,
    FrameRequestData,
    NeuralNetworkConfigData,
    ObservationData,
)


class TestNeuralNetworkWebsocketServer:
    def test_init(
        self, mock_neural_network_websocket_server: NeuralNetworkWebsocketServer, mock_config: Config
    ) -> None:
        assert mock_neural_network_websocket_server.config == mock_config

    def test_configure_neural_networks(self, nn_config_data: NeuralNetworkConfigData) -> None:
        """Test that the neural networks are configured correctly."""
        nn_suite = NeuralNetworkWebsocketServer.configure_neural_networks(
            NeuralNetworkConfigData.to_bytes(nn_config_data)
        )
        assert isinstance(nn_suite, NeuralNetworkSuite)

    def test_crossover_neural_networks(
        self, mock_neural_network_suite: NeuralNetworkSuite, population_fitness_data: FitnessData
    ) -> None:
        """Test that the crossover of neural networks works correctly."""
        with patch.object(mock_neural_network_suite, "crossover_networks", autospec=True) as mock_crossover:
            NeuralNetworkWebsocketServer.crossover_neural_networks(mock_neural_network_suite, population_fitness_data)
            assert mock_crossover.called

    def test_process_observations(
        self, mock_neural_network_suite: NeuralNetworkSuite, observation_data: ObservationData
    ) -> None:
        """Test that the processing of observations works correctly."""
        action_data = NeuralNetworkWebsocketServer.process_observations(mock_neural_network_suite, observation_data)
        assert isinstance(action_data, ActionData)

    @pytest.mark.asyncio
    async def test_handle_connection(
        self,
        mock_websocket: AsyncMock,
        nn_config_data: NeuralNetworkConfigData,
        frame_request_data_observation: FrameRequestData,
        frame_request_data_population: FrameRequestData,
        mock_configure_neural_networks: MagicMock,
        mock_crossover_neural_networks: MagicMock,
        mock_process_observations: MagicMock,
    ) -> None:
        """Test that the first message configures the neural networks."""
        config_bytes = NeuralNetworkConfigData.to_bytes(nn_config_data)

        await NeuralNetworkWebsocketServer.handle_connection(mock_websocket)
        mock_configure_neural_networks.assert_called_once_with(config_bytes)
        mock_process_observations.assert_called_once_with(
            mock_configure_neural_networks.return_value, frame_request_data_observation.observation
        )
        mock_crossover_neural_networks.assert_called_once_with(
            mock_configure_neural_networks.return_value, frame_request_data_population.population_fitness
        )

    @pytest.mark.asyncio
    async def test_start(
        self, mock_neural_network_websocket_server: NeuralNetworkWebsocketServer, mock_config: Config
    ) -> None:
        """Test that the websocket server starts correctly."""
        with patch("websockets.serve") as mock_serve:
            mock_serve.return_value.__aenter__.return_value = None
            mock_serve.return_value.__aexit__.return_value = None

            future: asyncio.Future = asyncio.Future()
            future.set_result(None)

            with patch("asyncio.Future", return_value=future):
                await mock_neural_network_websocket_server.start()

                mock_serve.assert_called_once_with(
                    NeuralNetworkWebsocketServer.handle_connection, mock_config.host, mock_config.port
                )

    def test_run(self, mock_neural_network_websocket_server: NeuralNetworkWebsocketServer) -> None:
        """Test that the run method calls asyncio.run with start."""
        with patch("asyncio.run") as mock_run:
            with patch.object(mock_neural_network_websocket_server, "start", new_callable=AsyncMock) as mock_start:
                mock_neural_network_websocket_server.run()

                mock_run.assert_called_once()
                mock_start.assert_called_once()


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
