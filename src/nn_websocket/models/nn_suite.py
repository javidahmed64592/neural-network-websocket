from neural_network.layer import HiddenLayer, InputLayer, OutputLayer
from neural_network.neural_network import NeuralNetwork

from nn_websocket.protobuf.proto_types import (
    ActionData,
    ActivationFunctionEnum,
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
