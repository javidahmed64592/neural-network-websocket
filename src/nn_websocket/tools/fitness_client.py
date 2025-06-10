import logging

import websockets

from nn_websocket.protobuf.frame_data_types import FrameRequestData
from nn_websocket.protobuf.neural_network_types import (
    ActivationFunctionEnumData,
    ConfigurationData,
    FitnessApproachConfigData,
    NeuralNetworkConfigData,
)
from nn_websocket.tools.base_client import BaseClient, run
from nn_websocket.tools.client_utils import get_random_observation_frame, get_random_train_request_frame

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="[%d-%m-%Y|%I:%M:%S]", level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
    learning_rate=LEARNING_RATE,
)

FITNESS_CONFIG = FitnessApproachConfigData(neural_network=NN_CONFIG)
CONFIG_DATA = ConfigurationData(fitness_approach=FITNESS_CONFIG)


class FitnessClient(BaseClient):
    async def send_observation(self, ws: websockets.ClientConnection) -> None:
        """Send observation data to the server."""
        await super().send_observation(ws)
        await ws.send(
            FrameRequestData.to_bytes(
                get_random_observation_frame(self.config_data.fitness_approach.neural_network.num_inputs)  # type: ignore[union-attr]
            )
        )

    async def send_training(self, ws: websockets.ClientConnection) -> None:
        """Send training data to the server."""
        await super().send_training(ws)
        await ws.send(
            FrameRequestData.to_bytes(
                get_random_train_request_frame(
                    TRAINING_BATCH_SIZE,
                    self.config_data.fitness_approach.neural_network.num_inputs,  # type: ignore[union-attr]
                )
            )
        )


def main() -> None:
    run(FitnessClient(CONFIG_DATA))
