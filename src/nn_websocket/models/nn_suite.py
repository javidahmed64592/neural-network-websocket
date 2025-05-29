import numpy as np
from neural_network.layer import HiddenLayer, InputLayer, OutputLayer
from neural_network.neural_network import NeuralNetwork
from numpy.typing import NDArray

from nn_websocket.protobuf.proto_types import (
    ActionData,
    NeuralNetworkConfigData,
    ObservationData,
)


class NeuralNetworkSuite:
    """A suite of neural networks for handling multiple configurations."""

    def __init__(self) -> None:
        """Initialize the suite with a configuration."""
        self.networks: list[NeuralNetwork] = []

    def set_networks(self, config_data: NeuralNetworkConfigData) -> None:
        """Set the neural networks based on the provided configuration data."""
        self.networks.clear()

        input_layer = InputLayer(
            size=config_data.num_inputs,
            activation=config_data.input_activation.get_class(),
        )

        hidden_layers = [
            HiddenLayer(
                size=size,
                activation=config_data.hidden_activation.get_class(),
                weights_range=(config_data.weights_min, config_data.weights_max),
                bias_range=(config_data.bias_min, config_data.bias_max),
            )
            for size in config_data.hidden_layer_sizes
        ]

        output_layer = OutputLayer(
            size=config_data.num_outputs,
            activation=config_data.output_activation.get_class(),
            weights_range=(config_data.weights_min, config_data.weights_max),
            bias_range=(config_data.bias_min, config_data.bias_max),
        )

        for _ in range(config_data.num_networks):
            network = NeuralNetwork.from_layers(layers=[input_layer, *hidden_layers, output_layer])
            self.networks.append(network)

    def set_networks_from_bytes(self, config_data_bytes: bytes) -> None:
        """Set the neural networks from a bytes representation of the configuration data."""
        config_data = NeuralNetworkConfigData.from_bytes(config_data_bytes)
        self.set_networks(config_data)

    @staticmethod
    def feedforward_through_network(nn: NeuralNetwork, observation: NDArray) -> ActionData:
        """Feedforward through the neural network and return the action data."""
        outputs = nn.feedforward(observation)
        return ActionData(outputs=outputs)

    def feedforward_through_networks(self, observation_data: ObservationData) -> list[ActionData]:
        """Feedforward through all networks and return a list of action data."""
        observations = np.reshape(
            observation_data.inputs,
            (len(self.networks), -1),
        )

        action_data_list = []
        for i, network in enumerate(self.networks):
            action_data = NeuralNetworkSuite.feedforward_through_network(network, observations[i])
            action_data_list.append(action_data)
        return action_data_list

    def feedforward_through_networks_from_bytes(self, observation_data_bytes: bytes) -> list[ActionData]:
        """Feedforward through all networks from bytes representation of observation data."""
        observation_data = ObservationData.from_bytes(observation_data_bytes)
        return self.feedforward_through_networks(observation_data)
