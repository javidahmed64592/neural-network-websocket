import asyncio
import logging
from pathlib import Path

import numpy as np
import websockets

from nn_websocket.models.config import Config
from nn_websocket.protobuf.proto_types import (
    ActivationFunctionEnum,
    ConfigurationData,
    FrameRequestData,
    GeneticAlgorithmConfigData,
    NeuralNetworkConfigData,
    ObservationData,
    PopulationFitnessData,
)

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="[%d-%m-%Y|%I:%M:%S]", level=logging.DEBUG)
logger = logging.getLogger(__name__)
rng = np.random.default_rng()

# Load websocket config from JSON file (same as main.py)
CONFIG_FILEPATH = Path("config") / "websocket_config.json"
config = Config.load_config(CONFIG_FILEPATH)
SERVER_URI = f"ws://{config.host}:{config.port}"

# Configuration for the mock client
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
INPUT_ACTIVATION = ActivationFunctionEnum.LINEAR
HIDDEN_ACTIVATION = ActivationFunctionEnum.RELU
OUTPUT_ACTIVATION = ActivationFunctionEnum.LINEAR

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
)

genetic_algorithm_config = GeneticAlgorithmConfigData(
    population_size=NUM_AGENTS,
    mutation_rate=MUTATION_RATE,
)

config_data = ConfigurationData(
    genetic_algorithm=genetic_algorithm_config,
    neural_network=nn_config,
)


class MockClient:
    @staticmethod
    def get_random_observation() -> FrameRequestData:
        """Generate a random observation for testing."""
        inputs = rng.uniform(low=-1, high=1, size=NUM_INPUTS * NUM_AGENTS).astype(np.float32).tolist()
        observation = ObservationData(inputs=inputs)
        return FrameRequestData(observation=observation)

    @staticmethod
    def get_random_population_fitness() -> FrameRequestData:
        """Generate a random population fitness for testing."""
        fitness = rng.uniform(low=0, high=1, size=NUM_AGENTS).astype(np.float32).tolist()
        population_fitness = PopulationFitnessData(fitness=fitness)
        return FrameRequestData(population_fitness=population_fitness)

    @staticmethod
    async def start() -> None:
        num_observations = 0

        async with websockets.connect(SERVER_URI) as ws:
            # Send configuration data to the server
            logger.info("Sending ConfigurationData to server.")
            await ws.send(ConfigurationData.to_bytes(config_data))
            await asyncio.sleep(5)

            # Send observations and population fitness data
            while True:
                logger.info("Sending ObservationData to server.")
                await ws.send(FrameRequestData.to_bytes(MockClient.get_random_observation()))
                num_observations += 1

                if num_observations % EPISODE_LENGTH == 0:
                    logger.info("Sending PopulationFitnessData to server.")
                    await ws.send(FrameRequestData.to_bytes(MockClient.get_random_population_fitness()))
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=2)
                    logger.info("Received response of length: %d", len(response))
                except TimeoutError:
                    logger.warning("No response received from server.")
                await asyncio.sleep(5)


def run() -> None:
    try:
        asyncio.run(MockClient.start())
    except (SystemExit, KeyboardInterrupt):
        logger.info("Shutting down the mock client.")
