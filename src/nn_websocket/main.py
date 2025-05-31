import asyncio
import logging
from pathlib import Path

import websockets

from nn_websocket.models.config import Config
from nn_websocket.models.nn_suite import NeuralNetworkSuite
from nn_websocket.protobuf.proto_types import ActionData, FrameRequestData, ObservationData, PopulationFitnessData

CONFIG_FILEPATH = Path("config") / "websocket_config.json"

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="[%d-%m-%Y|%I:%M:%S]", level=logging.DEBUG)
logger = logging.getLogger(__name__)


async def handle_connection(websocket: websockets.ServerConnection) -> None:
    neural_network_suite: NeuralNetworkSuite | None = None
    async for message in websocket:
        if isinstance(message, str):
            message_bytes = message.encode("utf-8")
        else:
            message_bytes = message
        if neural_network_suite is None:
            neural_network_suite = configure_neural_networks(message_bytes)
        else:
            message_data = FrameRequestData.from_bytes(message_bytes)
            if population_fitness := message_data.population_fitness:
                crossover_neural_networks(neural_network_suite, population_fitness)
            elif observation := message_data.observation:
                actions = process_observations(neural_network_suite, observation)
                await websocket.send(actions.to_bytes(actions))


def configure_neural_networks(config: bytes) -> NeuralNetworkSuite:
    logger.info("Configuring neural networks...")
    return NeuralNetworkSuite.from_bytes(config)


def crossover_neural_networks(
    neural_network_suite: NeuralNetworkSuite, population_fitness: PopulationFitnessData
) -> None:
    logger.info("Crossover neural networks...")
    neural_network_suite.crossover_networks(population_fitness)


def process_observations(neural_network_suite: NeuralNetworkSuite, observation: ObservationData) -> ActionData:
    return neural_network_suite.feedforward_through_networks(observation)


async def main() -> None:
    config = Config.load_config(CONFIG_FILEPATH)

    async with websockets.serve(handle_connection, config.host, config.port):
        logger.info("Neural network websocket server running on %s:%s", config.host, config.port)
        await asyncio.Future()


def run() -> None:
    try:
        asyncio.run(main())
    except (SystemExit, KeyboardInterrupt):
        logger.info("Shutting down the neural network websocket server.")
