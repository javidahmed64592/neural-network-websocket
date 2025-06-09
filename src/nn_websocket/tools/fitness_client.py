import asyncio
import logging

import numpy as np
import websockets

from nn_websocket.protobuf.frame_data_types import (
    FitnessData,
    FrameRequestData,
    TrainRequestData,
)
from nn_websocket.protobuf.neural_network_types import (
    ActivationFunctionEnumData,
    ConfigurationData,
    FitnessApproachConfigData,
    NeuralNetworkConfigData,
)
from nn_websocket.tools.client_utils import get_config, get_random_observation

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="[%d-%m-%Y|%I:%M:%S]", level=logging.DEBUG)
logger = logging.getLogger(__name__)
rng = np.random.default_rng()

# Load websocket config from JSON file
config = get_config()
SERVER_URI = f"ws://{config.host}:{config.port}"

# Configuration for the fitness approach client
NUM_INPUTS = 5
NUM_OUTPUTS = 2
HIDDEN_LAYER_SIZES = [4, 4]
WEIGHTS_MIN = -1.0
WEIGHTS_MAX = 1.0
BIAS_MIN = -0.2
BIAS_MAX = 0.2
INPUT_ACTIVATION = ActivationFunctionEnumData.LINEAR
HIDDEN_ACTIVATION = ActivationFunctionEnumData.RELU
OUTPUT_ACTIVATION = ActivationFunctionEnumData.SIGMOID
LEARNING_RATE = 0.01

TRAINING_BATCH_SIZE = 5

nn_config = NeuralNetworkConfigData(
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
    learning_rate=LEARNING_RATE,
)

fitness_approach_config = FitnessApproachConfigData(neural_network=nn_config)
config_data = ConfigurationData(fitness_approach=fitness_approach_config)


class FitnessClient:
    @staticmethod
    def get_training_data() -> FrameRequestData:
        """Generate training data with observations and target fitness values."""
        # TODO: Move to client_utils.py
        observations = []
        fitness_values = []

        for _ in range(TRAINING_BATCH_SIZE):
            # Generate random observation
            observations.append(get_random_observation(NUM_INPUTS))

            # Generate target fitness (could be based on some reward function)
            fitness = rng.uniform(low=0, high=1, size=1).astype(np.float32).tolist()
            fitness_values.append(FitnessData(values=fitness))

        train_request = TrainRequestData(observation=observations, fitness=fitness_values)
        return FrameRequestData(train_request=train_request)

    @staticmethod
    async def start() -> None:
        observation_count = 0

        async with websockets.connect(SERVER_URI) as ws:
            # Send configuration data to the server
            logger.info("Sending FitnessApproachConfigData to server.")
            await ws.send(ConfigurationData.to_bytes(config_data))
            await asyncio.sleep(1)

            # Send observations and training data
            while True:
                logger.info("Sending ObservationData to server (observation #%d).", observation_count + 1)
                await ws.send(FrameRequestData.to_bytes(get_random_observation(NUM_INPUTS)))
                observation_count += 1

                # Every TRAINING_BATCH_SIZE observations, send training data
                if observation_count % TRAINING_BATCH_SIZE == 0:
                    logger.info("Sending training batch to server.")
                    await ws.send(FrameRequestData.to_bytes(FitnessClient.get_training_data()))

                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=2)
                    logger.info("Received action response of length: %d bytes", len(response))
                except TimeoutError:
                    logger.warning("No response received from server within timeout.")

                await asyncio.sleep(5)


def main() -> None:
    try:
        asyncio.run(FitnessClient.start())
    except (SystemExit, KeyboardInterrupt):
        logger.info("Shutting down the fitness approach client.")
