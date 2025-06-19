"""Unit tests for the src/nn_websocket/main.py module."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nn_websocket.main import NeuralNetworkWebsocketServer, run
from nn_websocket.models.config import Config
from nn_websocket.models.nn_suites import FitnessSuite, NeuroevolutionSuite
from nn_websocket.protobuf.frame_data_types import (
    ActionData,
    FitnessData,
    FrameRequestData,
    ObservationData,
    TrainRequestData,
)
from nn_websocket.protobuf.nn_websocket_data_types import ConfigurationData


class TestNeuralNetworkWebsocketServer:
    """Test suite for NeuralNetworkWebsocketServer methods and behaviors."""

    def test_init(
        self, mock_neural_network_websocket_server: NeuralNetworkWebsocketServer, mock_config: Config
    ) -> None:
        """Test that the NeuralNetworkWebsocketServer initializes with the correct configuration."""
        assert mock_neural_network_websocket_server.config == mock_config

    def test_configure_nn_suite_neuroevolution(self, configuration_data_neuroevolution: ConfigurationData) -> None:
        """Test that the neural network suite is configured for neuroevolution."""
        nn_suite = NeuralNetworkWebsocketServer.configure_neural_network_suite(configuration_data_neuroevolution)
        assert isinstance(nn_suite, NeuroevolutionSuite)

    def test_configure_nn_suite_fitness_approach(self, configuration_data_fitness: ConfigurationData) -> None:
        """Test that the neural network suite is configured for fitness approach."""
        nn_suite = NeuralNetworkWebsocketServer.configure_neural_network_suite(configuration_data_fitness)
        assert isinstance(nn_suite, FitnessSuite)

    def test_process_observations_neuroevolution(
        self, mock_neuroevolution_suite: NeuroevolutionSuite, observation_data: ObservationData
    ) -> None:
        """Test that the processing of observations works correctly for neuroevolution."""
        observation_data.inputs *= mock_neuroevolution_suite.nn_ga.population_size
        action_data = NeuralNetworkWebsocketServer.process_observations(mock_neuroevolution_suite, observation_data)
        assert isinstance(action_data, ActionData)
        assert (
            len(action_data.outputs)
            == mock_neuroevolution_suite.nn_ga.population_size
            * mock_neuroevolution_suite.nn_ga.nn_members[0]._num_outputs
        )

    def test_process_observations_fitness_approach(
        self, mock_fitness_suite: FitnessSuite, observation_data: ObservationData
    ) -> None:
        """Test that the processing of observations works correctly for fitness approach."""
        action_data = NeuralNetworkWebsocketServer.process_observations(mock_fitness_suite, observation_data)
        assert isinstance(action_data, ActionData)
        assert len(action_data.outputs) == mock_fitness_suite.nn_member._num_outputs

    def test_crossover_neural_networks(
        self, mock_neuroevolution_suite: NeuroevolutionSuite, fitness_data: FitnessData
    ) -> None:
        """Test that the crossover of neural networks works correctly."""
        with patch.object(mock_neuroevolution_suite, "crossover_networks", autospec=True) as mock_crossover:
            NeuralNetworkWebsocketServer.crossover_neural_networks(mock_neuroevolution_suite, fitness_data)
            assert mock_crossover.called

    def test_train_neural_network(self, mock_fitness_suite: FitnessSuite, train_request_data: TrainRequestData) -> None:
        """Test that the training of neural networks works correctly."""
        with patch.object(mock_fitness_suite, "train", autospec=True) as mock_train:
            NeuralNetworkWebsocketServer.train(mock_fitness_suite, train_request_data)
            assert mock_train.called

    @pytest.mark.asyncio
    async def test_handle_connection_neuroevolution(
        self,
        mock_websocket_neuroevolution: AsyncMock,
        configuration_data_neuroevolution: ConfigurationData,
        frame_request_data_observation: FrameRequestData,
        frame_request_data_fitness: FrameRequestData,
        mock_configure_neural_networks_neuroevolution: MagicMock,
        mock_process_observations: MagicMock,
        mock_crossover_neural_networks: MagicMock,
        action_data: ActionData,
    ) -> None:
        """Test that the handle_connection method works correctly for neuroevolution."""
        await NeuralNetworkWebsocketServer.handle_connection(mock_websocket_neuroevolution)

        mock_configure_neural_networks_neuroevolution.assert_called_once_with(
            ConfigurationData.from_bytes(ConfigurationData.to_bytes(configuration_data_neuroevolution))
        )

        mock_process_observations.assert_called_once_with(
            mock_configure_neural_networks_neuroevolution.return_value, frame_request_data_observation.observation
        )

        mock_crossover_neural_networks.assert_called_once_with(
            mock_configure_neural_networks_neuroevolution.return_value, frame_request_data_fitness.fitness
        )

        mock_websocket_neuroevolution.send.assert_called_once_with(ActionData.to_bytes(action_data))

    @pytest.mark.asyncio
    async def test_handle_connection_fitness_approach(
        self,
        mock_websocket_fitness: AsyncMock,
        configuration_data_fitness: ConfigurationData,
        frame_request_data_observation: FrameRequestData,
        frame_request_data_train: FrameRequestData,
        mock_configure_neural_networks_fitness: MagicMock,
        mock_process_observations: MagicMock,
        mock_train_neural_network: MagicMock,
        action_data: ActionData,
    ) -> None:
        """Test that the handle_connection method works correctly for fitness approach."""
        await NeuralNetworkWebsocketServer.handle_connection(mock_websocket_fitness)

        mock_configure_neural_networks_fitness.assert_called_once_with(
            ConfigurationData.from_bytes(ConfigurationData.to_bytes(configuration_data_fitness))
        )

        mock_process_observations.assert_called_once_with(
            mock_configure_neural_networks_fitness.return_value, frame_request_data_observation.observation
        )

        mock_train_neural_network.assert_called_once_with(
            mock_configure_neural_networks_fitness.return_value, frame_request_data_train.train_request
        )

        mock_websocket_fitness.send.assert_called_once_with(ActionData.to_bytes(action_data))

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
    """Test cases for the run function in main.py."""

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
