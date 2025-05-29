import asyncio
import logging
from pathlib import Path

import websockets

from nn_websocket.models.config import Config
from nn_websocket.models.nn_suite import NeuralNetworkSuite

CONFIG_FILEPATH = Path("config") / "websocket_config.json"

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="[%d-%m-%Y|%I:%M:%S]", level=logging.DEBUG)
logger = logging.getLogger(__name__)
neural_network_suite = NeuralNetworkSuite()


async def handle_connection(websocket: websockets.ServerConnection) -> None:
    num_messages = 0
    async for message in websocket:
        logger.info("Received message of length: %s", len(message))
        if isinstance(message, str):
            logger.info("Message is a string, encoding to bytes.")
            message_bytes = message.encode("utf-8")
        else:
            message_bytes = message
        if num_messages == 0:
            logger.info("Configuring neural networks with the first message.")
            configure_neural_networks(message_bytes)
        else:
            await process_observations(websocket, message_bytes)
        num_messages += 1


def configure_neural_networks(config: bytes) -> None:
    neural_network_suite.set_networks_from_bytes(config)
    logger.info("Configured %s neural networks with shared settings.", len(neural_network_suite.networks))


async def process_observations(websocket: websockets.ServerConnection, message: bytes) -> None:
    actions = neural_network_suite.feedforward_through_networks_from_bytes(message)
    logger.info("Calculated actions: %s", actions.outputs)
    await websocket.send(actions.to_bytes(actions))


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
