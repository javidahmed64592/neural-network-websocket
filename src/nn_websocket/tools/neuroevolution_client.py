"""Mock neuroevolution client for the neural network websocket server.

Simulates a population-based neuroevolution approach for testing and demonstration.
"""

import logging

import websockets
from neural_network.protobuf.neural_network_types import (
    ActivationFunctionEnum,
    AdamOptimizerDataType,
    LearningRateMethodEnum,
    LearningRateSchedulerDataType,
    OptimizerDataType,
)

from nn_websocket.protobuf.frame_data_types import (
    FrameRequestData,
)
from nn_websocket.protobuf.nn_websocket_data_types import (
    ConfigurationData,
    GeneticAlgorithmConfigData,
    NeuralNetworkConfigData,
    NeuroevolutionConfigData,
)
from nn_websocket.tools.base_client import BaseClient, run
from nn_websocket.tools.client_utils import get_random_fitness_frame, get_random_observation_frame

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="[%d-%m-%Y|%I:%M:%S]", level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configuration for the neuroevolution client
NUM_INPUTS = 5
NUM_OUTPUTS = 2
HIDDEN_LAYER_SIZES = [4, 4]
WEIGHTS_MIN = -1.0
WEIGHTS_MAX = 1.0
BIAS_MIN = -0.2
BIAS_MAX = 0.2
INPUT_ACTIVATION = ActivationFunctionEnum.LINEAR
HIDDEN_ACTIVATION = ActivationFunctionEnum.RELU
OUTPUT_ACTIVATION = ActivationFunctionEnum.SIGMOID
MOCK_LEARNING_RATE = 0.01
MOCK_BETA1 = 0.9
MOCK_BETA2 = 0.999
MOCK_EPSILON = 1e-08
MOCK_DECAY_RATE = 0.1
MOCK_DECAY_STEPS = 1000
MOCK_LEARNING_RATE_METHOD = LearningRateMethodEnum.EXPONENTIAL_DECAY
MOCK_OPTIMIZER = OptimizerDataType(
    adam=AdamOptimizerDataType(
        learning_rate=MOCK_LEARNING_RATE,
        beta1=MOCK_BETA1,
        beta2=MOCK_BETA2,
        epsilon=MOCK_EPSILON,
    ),
    sgd=None,
    learning_rate_scheduler=LearningRateSchedulerDataType(
        decay_rate=MOCK_DECAY_RATE, decay_steps=MOCK_DECAY_STEPS, method=MOCK_LEARNING_RATE_METHOD
    ),
)

NUM_AGENTS = 10
MUTATION_RATE = 0.1

NN_CONFIG = NeuralNetworkConfigData(
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
    optimizer=MOCK_OPTIMIZER,
)
GA_CONFIG = GeneticAlgorithmConfigData(population_size=NUM_AGENTS, mutation_rate=MUTATION_RATE)

NEUROEVOLUTION_CONFIG = NeuroevolutionConfigData(
    neural_network=NN_CONFIG,
    genetic_algorithm=GA_CONFIG,
)
CONFIG_DATA = ConfigurationData(neuroevolution=NEUROEVOLUTION_CONFIG)


class NeuroevolutionClient(BaseClient):
    """Client for simulating neuroevolution-based neural network training over websocket."""

    async def send_observation(self, ws: websockets.ClientConnection) -> None:
        """Send observation data to the server.

        :param websockets.ClientConnection ws:
            The websocket connection to send data to.
        """
        await super().send_observation(ws)
        await ws.send(
            FrameRequestData.to_bytes(
                get_random_observation_frame(
                    self.config_data.neuroevolution.neural_network.num_inputs  # type: ignore[union-attr]
                    * self.config_data.neuroevolution.genetic_algorithm.population_size  # type: ignore[union-attr]
                )
            )
        )

    async def send_training(self, ws: websockets.ClientConnection) -> None:
        """Send training data to the server.

        :param websockets.ClientConnection ws:
            The websocket connection to send data to.
        """
        await super().send_training(ws)
        await ws.send(
            FrameRequestData.to_bytes(
                get_random_fitness_frame(self.config_data.neuroevolution.genetic_algorithm.population_size)  # type: ignore[union-attr]
            )
        )


def main() -> None:
    """Entry point for running the neuroevolution client."""
    run(NeuroevolutionClient(CONFIG_DATA))
