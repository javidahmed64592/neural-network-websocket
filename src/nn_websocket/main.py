import asyncio
import logging
from pathlib import Path

import websockets

from nn_websocket.models.config import Config
from nn_websocket.models.nn_suite import NeuralNetworkSuite
from nn_websocket.protobuf.proto_types import ActionData, FitnessData, FrameRequestData, ObservationData

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="[%d-%m-%Y|%I:%M:%S]", level=logging.DEBUG)
logger = logging.getLogger(__name__)

CONFIG_FILEPATH = Path("config") / "websocket_config.json"


class NeuralNetworkWebsocketServer:
    def __init__(self, config_filepath: Path = CONFIG_FILEPATH) -> None:
        self.config_filepath = config_filepath
        self.config = Config.load_config(self.config_filepath)

    @staticmethod
    def configure_neural_networks(config: bytes) -> NeuralNetworkSuite:
        logger.info("Configuring neural networks...")
        return NeuralNetworkSuite.from_bytes(config)

    @staticmethod
    def crossover_neural_networks(neural_network_suite: NeuralNetworkSuite, fitness_data: FitnessData) -> None:
        logger.info("Crossover neural networks...")
        neural_network_suite.crossover_networks(fitness_data)

    @staticmethod
    def process_observations(neural_network_suite: NeuralNetworkSuite, observation: ObservationData) -> ActionData:
        return neural_network_suite.feedforward_through_networks(observation)

    @staticmethod
    async def handle_connection(websocket: websockets.ServerConnection) -> None:
        neural_network_suite: NeuralNetworkSuite | None = None
        async for message in websocket:
            if isinstance(message, str):
                message_bytes = message.encode("utf-8")
            else:
                message_bytes = message
            if neural_network_suite is None:
                neural_network_suite = NeuralNetworkWebsocketServer.configure_neural_networks(message_bytes)
            else:
                message_data = FrameRequestData.from_bytes(message_bytes)
                if fitness := message_data.fitness:
                    NeuralNetworkWebsocketServer.crossover_neural_networks(neural_network_suite, fitness)
                elif observation := message_data.observation:
                    actions = NeuralNetworkWebsocketServer.process_observations(neural_network_suite, observation)
                    await websocket.send(actions.to_bytes(actions))

    async def start(self) -> None:
        async with websockets.serve(NeuralNetworkWebsocketServer.handle_connection, self.config.host, self.config.port):
            logger.info("Neural network websocket server running on %s:%s", self.config.host, self.config.port)
            await asyncio.Future()

    def run(self) -> None:
        asyncio.run(self.start())


def run() -> None:
    try:
        server = NeuralNetworkWebsocketServer()
        server.run()
    except (SystemExit, KeyboardInterrupt):
        logger.info("Shutting down the neural network websocket server.")
