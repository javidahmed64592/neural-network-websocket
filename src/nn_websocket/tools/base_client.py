"""Base class and runner utility for neural network websocket clients."""

import asyncio
import logging

import websockets

from nn_websocket.protobuf.nn_websocket_data_types import (
    ConfigurationData,
)
from nn_websocket.tools.client_utils import get_config

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="[%d-%m-%Y|%I:%M:%S]", level=logging.DEBUG)
logger = logging.getLogger(__name__)

config = get_config()
EPISODE_LENGTH = 10


class BaseClient:
    """Base class for neural network websocket clients."""

    def __init__(self, config_data: ConfigurationData) -> None:
        """Initialize the base client with configuration data.

        :param ConfigurationData config_data:
            The configuration data for the client.
        """
        self.config_data = config_data

    async def send_configuration(self, ws: websockets.ClientConnection) -> None:
        """Send configuration data to the server.

        :param websockets.ClientConnection ws:
            The websocket connection to send data to.
        """
        logger.info("Sending ConfigData to server.")
        await ws.send(ConfigurationData.to_bytes(self.config_data))
        await asyncio.sleep(1)

    async def send_observation(self, ws: websockets.ClientConnection) -> None:
        """Send observation data to the server.

        :param websockets.ClientConnection ws:
            The websocket connection to send data to.
        """
        logger.info("Sending ObservationData to server.")

    async def send_training(self, ws: websockets.ClientConnection) -> None:
        """Send training data to the server.

        :param websockets.ClientConnection ws:
            The websocket connection to send data to.
        """
        logger.info("Sending training batch to server.")

    async def start(self) -> None:
        """Start the client and manage the websocket connection loop."""
        num_observations = 0

        async with websockets.connect(config.uri) as ws:
            # Send configuration data to the server
            await self.send_configuration(ws)

            # Send observations
            while True:
                await self.send_observation(ws)
                num_observations += 1

                # Every EPISODE_LENGTH observations, send training data
                if num_observations % EPISODE_LENGTH == 0:
                    await self.send_training(ws)

                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=2)
                    logger.info("Received action response of length: %d bytes", len(response))
                except TimeoutError:
                    logger.warning("No response received from server within timeout.")

                await asyncio.sleep(5)


def run(client: BaseClient) -> None:
    """Run the websocket client event loop.

    :param BaseClient client:
        The client instance to run.
    """
    try:
        asyncio.run(client.start())
    except (SystemExit, KeyboardInterrupt):
        logger.info("Shutting down client.")
