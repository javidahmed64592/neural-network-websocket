import asyncio
import logging
from pathlib import Path

import websockets

from nn_websocket.models.config import Config
from nn_websocket.models.nn_suites import FitnessSuite, NeuroevolutionSuite
from nn_websocket.protobuf.frame_data_types import (
    ActionData,
    FitnessData,
    FrameRequestData,
    ObservationData,
    TrainRequestData,
)
from nn_websocket.protobuf.neural_network_types import (
    ConfigurationData,
    FitnessApproachConfigData,
    NeuroevolutionConfigData,
)

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="[%d-%m-%Y|%I:%M:%S]", level=logging.DEBUG)
logger = logging.getLogger(__name__)

CONFIG_FILEPATH = Path("config") / "websocket_config.json"


class NeuralNetworkWebsocketServer:
    def __init__(self, config_filepath: Path = CONFIG_FILEPATH) -> None:
        self.config_filepath = config_filepath
        self.config = Config.load_config(self.config_filepath)

    @staticmethod
    def configure_neural_network_suite(config: ConfigurationData) -> NeuroevolutionSuite | FitnessSuite:
        logger.info("Configuring neural networks...")
        if neuroevolution := config.neuroevolution:
            return NeuroevolutionSuite.from_bytes(NeuroevolutionConfigData.to_bytes(neuroevolution))

        fitness_approach = config.fitness_approach
        return FitnessSuite.from_bytes(FitnessApproachConfigData.to_bytes(fitness_approach))

    @staticmethod
    def process_observations(
        neural_network_suite: NeuroevolutionSuite | FitnessSuite, observation: ObservationData
    ) -> ActionData:
        if isinstance(neural_network_suite, NeuroevolutionSuite):
            return neural_network_suite.feedforward_through_networks(observation)
        return neural_network_suite.feedforward(observation)

    @staticmethod
    def crossover_neural_networks(neural_network_suite: NeuroevolutionSuite, fitness_data: FitnessData) -> None:
        logger.info("Crossover neural networks...")
        neural_network_suite.crossover_networks(fitness_data)

    @staticmethod
    def train(neural_network_suite: FitnessSuite, train_request: TrainRequestData) -> None:
        logger.info("Training neural network...")
        neural_network_suite.train(train_request)

    @staticmethod
    async def handle_connection(websocket: websockets.ServerConnection) -> None:
        def _ensure_bytes(message: str | bytes) -> bytes:
            return message.encode("utf-8") if isinstance(message, str) else message

        neural_network_suite: NeuroevolutionSuite | FitnessSuite | None = None
        async for message in websocket:
            message_bytes = _ensure_bytes(message)

            # Initialise the neural network suite if not already done
            if neural_network_suite is None:
                neural_network_suite = NeuralNetworkWebsocketServer.configure_neural_network_suite(
                    ConfigurationData.from_bytes(message_bytes)
                )
            # Check if client requesting actions or training
            else:
                message_data = FrameRequestData.from_bytes(message_bytes)
                if observation := message_data.observation:
                    actions = NeuralNetworkWebsocketServer.process_observations(neural_network_suite, observation)
                    await websocket.send(actions.to_bytes(actions))
                elif (fitness := message_data.fitness) and isinstance(neural_network_suite, NeuroevolutionSuite):
                    NeuralNetworkWebsocketServer.crossover_neural_networks(neural_network_suite, fitness)
                elif (train_request := message_data.train_request) and isinstance(neural_network_suite, FitnessSuite):
                    NeuralNetworkWebsocketServer.train(neural_network_suite, train_request)

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
