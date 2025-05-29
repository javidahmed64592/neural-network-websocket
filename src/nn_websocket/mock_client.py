import asyncio
import json
import logging
from pathlib import Path

import numpy as np
import websockets

from nn_websocket.models.config import Config
from nn_websocket.protobuf.proto_types import (
    ActivationFunctionEnum,
    NeuralNetworkConfigData,
    ObservationData,
)

rng = np.random.default_rng()

# Load websocket config from JSON file (same as main.py)
CONFIG_FILEPATH = Path("config") / "websocket_config.json"
with CONFIG_FILEPATH.open() as f:
    config_dict = json.load(f)
config = Config(**config_dict)
SERVER_URI = f"ws://{config.host}:{config.port}"

# Configuration for the mock client
NUM_NETWORKS = 10
NUM_INPUTS = 5
NUM_OUTPUTS = 2
HIDDEN_LAYER_SIZES = [4, 4]
WEIGHTS_MIN = -1.0
WEIGHTS_MAX = 1.0
BIAS_MIN = -0.2
BIAS_MAX = 0.2
INPUT_ACTIVATION = ActivationFunctionEnum.LINEAR
HIDDEN_ACTIVATION = ActivationFunctionEnum.RELU
OUTPUT_ACTIVATION = ActivationFunctionEnum.LINEAR

nn_config = NeuralNetworkConfigData(
    num_networks=NUM_NETWORKS,
    num_inputs=NUM_INPUTS,
    num_outputs=NUM_OUTPUTS,
    hidden_layer_sizes=HIDDEN_LAYER_SIZES,
    weights_min=WEIGHTS_MIN,
    weights_max=WEIGHTS_MAX,
    bias_min=BIAS_MIN,
    bias_max=BIAS_MAX,
    input_activation=INPUT_ACTIVATION,
    hidden_activation=HIDDEN_ACTIVATION,
    output_activation=OUTPUT_ACTIVATION,
)

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="[%d-%m-%Y|%I:%M:%S]", level=logging.DEBUG)
logger = logging.getLogger(__name__)


async def mock_client() -> None:
    async with websockets.connect(SERVER_URI) as ws:
        # Send NeuralNetworkConfigData as bytes
        config_bytes = NeuralNetworkConfigData.to_bytes(nn_config)
        logger.info("Sending NeuralNetworkConfigData to server.")
        await ws.send(config_bytes)
        await asyncio.sleep(5)

        # Periodically send ObservationData as bytes
        while True:
            inputs = rng.uniform(low=-1, high=1, size=NUM_INPUTS * NUM_NETWORKS).astype(np.float32).tolist()
            observation = ObservationData(inputs=inputs)
            obs_bytes = ObservationData.to_bytes(observation)
            logger.info("Sending ObservationData to server.")
            await ws.send(obs_bytes)
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=2)
                logger.info("Received response of length: %d", len(response))
            except TimeoutError:
                logger.warning("No response received from server.")
            await asyncio.sleep(5)


def run() -> None:
    try:
        asyncio.run(mock_client())
    except (SystemExit, KeyboardInterrupt):
        logger.info("Shutting down the mock client.")
