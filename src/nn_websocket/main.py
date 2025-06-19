"""Websocket server for neural network operations and training.

This module implements a Websocket server for handling neural network operations,
including processing observations, training networks, and performing crossover operations.
It uses the `websockets` library to manage connections and messages.
It is designed to work with neural network configurations defined in Protobuf format,
and it supports both neuroevolution and fitness-based approaches to neural networks.
This script serves as the entry point for running the server.
"""

import asyncio
import logging
import os
from pathlib import Path

import websockets

from nn_websocket.models.config import Config
from nn_websocket.models.nn_suites import FitnessSuite, NeuroevolutionSuite
from nn_websocket.protobuf.frame_data_types import (
    ActionType,
    FitnessType,
    FrameRequestDataType,
    ObservationType,
    TrainRequestType,
)
from nn_websocket.protobuf.nn_websocket_data_types import (
    ConfigDataType,
    FitnessApproachConfigType,
    NeuroevolutionConfigType,
)

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="[%d-%m-%Y|%I:%M:%S]", level=logging.DEBUG)
logger = logging.getLogger(__name__)

CONFIG_FILEPATH = Path("config") / "websocket_config.json"


class NeuralNetworkWebsocketServer:
    """A Websocket server for handling neural network operations."""

    def __init__(self, config_filepath: os.PathLike = CONFIG_FILEPATH) -> None:
        """Initialize the Websocket server with the given configuration file path.

        :param os.PathLike config_filepath:
            The path to the configuration file containing server settings and neural network configurations.
        """
        self.config_filepath = config_filepath
        self.config = Config.load_config(self.config_filepath)

    @staticmethod
    def configure_neural_network_suite(config: ConfigDataType) -> NeuroevolutionSuite | FitnessSuite:
        """Configure the neural network suite based on the provided configuration data.

        :param ConfigDataType config:
            The configuration data containing settings for the neural networks.
        :return NeuroevolutionSuite | FitnessSuite:
            An instance of NeuroevolutionSuite or FitnessSuite based on the configuration.
        :raises ValueError:
            If no fitness approach is configured in the provided ConfigDataType.
        """
        logger.info("Configuring neural networks...")
        if neuroevolution := config.neuroevolution:
            return NeuroevolutionSuite.from_bytes(NeuroevolutionConfigType.to_bytes(neuroevolution))

        if not (fitness_approach := config.fitness_approach):
            msg = "No fitness approach configured in the provided ConfigDataType."
            raise ValueError(msg)
        return FitnessSuite.from_bytes(FitnessApproachConfigType.to_bytes(fitness_approach))

    @staticmethod
    def process_observations(
        neural_network_suite: NeuroevolutionSuite | FitnessSuite, observation: ObservationType
    ) -> ActionType:
        """Process an observation through the neural network suite and return the resulting actions.

        :param NeuroevolutionSuite | FitnessSuite neural_network_suite:
            The neural network suite to process the observation.
        :param ObservationType observation:
            The observation data to process.
        :return ActionType:
            The resulting action data from the neural network(s).
        """
        if isinstance(neural_network_suite, NeuroevolutionSuite):
            return neural_network_suite.feedforward_through_networks(observation)
        return neural_network_suite.feedforward(observation)

    @staticmethod
    def crossover_neural_networks(neural_network_suite: NeuroevolutionSuite, fitness_data: FitnessType) -> None:
        """Perform crossover on the neural networks using the provided fitness data.

        :param NeuroevolutionSuite neural_network_suite:
            The neuroevolution suite containing the networks to crossover.
        :param FitnessType fitness_data:
            The fitness data to guide the crossover process.
        """
        logger.info("Crossover neural networks...")
        neural_network_suite.crossover_networks(fitness_data)

    @staticmethod
    def train(neural_network_suite: FitnessSuite, train_request: TrainRequestType) -> None:
        """Train the neural network using the provided training request.

        :param FitnessSuite neural_network_suite: The fitness suite containing the network to train.
        :param TrainRequestType train_request: The training request data.
        """
        logger.info("Training neural network...")
        neural_network_suite.train(train_request)

    @staticmethod
    async def handle_connection(websocket: websockets.ServerConnection) -> None:
        """Handle an incoming websocket connection, processing messages for neural network operations.

        :param websockets.ServerConnection websocket: The websocket connection to handle.
        """
        neural_network_suite: NeuroevolutionSuite | FitnessSuite | None = None
        async for message in websocket:
            message_bytes = message.encode("utf-8") if isinstance(message, str) else message

            # Initialise the neural network suite if not already done
            if neural_network_suite is None:
                neural_network_suite = NeuralNetworkWebsocketServer.configure_neural_network_suite(
                    ConfigDataType.from_bytes(message_bytes)
                )
            # Check if client requesting actions or training
            else:
                message_data = FrameRequestDataType.from_bytes(message_bytes)
                if observation := message_data.observation:
                    actions = NeuralNetworkWebsocketServer.process_observations(neural_network_suite, observation)
                    await websocket.send(actions.to_bytes(actions))
                elif (fitness := message_data.fitness) and isinstance(neural_network_suite, NeuroevolutionSuite):
                    NeuralNetworkWebsocketServer.crossover_neural_networks(neural_network_suite, fitness)
                elif (train_request := message_data.train_request) and isinstance(neural_network_suite, FitnessSuite):
                    NeuralNetworkWebsocketServer.train(neural_network_suite, train_request)

    async def start(self) -> None:
        """Start the websocket server and listen for incoming connections."""
        async with websockets.serve(NeuralNetworkWebsocketServer.handle_connection, self.config.host, self.config.port):
            logger.info("Neural network websocket server running on %s:%s", self.config.host, self.config.port)
            await asyncio.Future()

    def run(self) -> None:
        """Run the websocket server event loop."""
        asyncio.run(self.start())


def run() -> None:
    """Entry point for running the neural network websocket server."""
    try:
        server = NeuralNetworkWebsocketServer()
        server.run()
    except (SystemExit, KeyboardInterrupt):
        logger.info("Shutting down the neural network websocket server.")
