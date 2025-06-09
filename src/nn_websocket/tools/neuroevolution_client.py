import asyncio
import logging

import numpy as np
import websockets

from nn_websocket.protobuf.frame_data_types import (
    FrameRequestData,
)
from nn_websocket.protobuf.neural_network_types import (
    ActivationFunctionEnumData,
    ConfigurationData,
    GeneticAlgorithmConfigData,
    NeuralNetworkConfigData,
    NeuroevolutionConfigData,
)
from nn_websocket.tools.client_utils import get_config, get_random_fitness_frame, get_random_observation_frame

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="[%d-%m-%Y|%I:%M:%S]", level=logging.DEBUG)
logger = logging.getLogger(__name__)
rng = np.random.default_rng()

# Load websocket config from JSON file
config = get_config()

# Configuration for the neuroevolution client
NUM_AGENTS = 10
MUTATION_RATE = 0.1
EPISODE_LENGTH = 10

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

genetic_algorithm_config = GeneticAlgorithmConfigData(
    population_size=NUM_AGENTS,
    mutation_rate=MUTATION_RATE,
)

neuroevolution_config = NeuroevolutionConfigData(
    neural_network=nn_config,
    genetic_algorithm=genetic_algorithm_config,
)

config_data = ConfigurationData(neuroevolution=neuroevolution_config)


class NeuroevolutionClient:
    @staticmethod
    async def start() -> None:
        num_observations = 0

        async with websockets.connect(config.uri) as ws:
            # Send configuration data to the server
            logger.info("Sending NeuroevolutionConfigData to server.")
            await ws.send(ConfigurationData.to_bytes(config_data))
            await asyncio.sleep(1)

            # Send observations and fitness data in episodes
            while True:
                logger.info("Sending ObservationData to server (observation #%d).", num_observations + 1)
                await ws.send(FrameRequestData.to_bytes(get_random_observation_frame(NUM_INPUTS * NUM_AGENTS)))
                num_observations += 1

                # Every EPISODE_LENGTH observations, send fitness data for crossover
                if num_observations % EPISODE_LENGTH == 0:
                    logger.info("Episode complete. Sending FitnessData to server for crossover.")
                    await ws.send(FrameRequestData.to_bytes(get_random_fitness_frame(NUM_AGENTS)))

                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=2)
                    logger.info("Received action response of length: %d bytes", len(response))
                except TimeoutError:
                    logger.warning("No response received from server within timeout.")

                await asyncio.sleep(5)


def main() -> None:
    try:
        asyncio.run(NeuroevolutionClient.start())
    except (SystemExit, KeyboardInterrupt):
        logger.info("Shutting down the neuroevolution client.")
