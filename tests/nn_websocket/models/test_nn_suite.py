from nn_websocket.models.nn_suite import NeuralNetworkSuite
from nn_websocket.protobuf.proto_types import NeuralNetworkConfigData


class TestNeuralNetworkSuite:
    def test_set_networks(self, nn_config_data: NeuralNetworkConfigData) -> None:
        suite = NeuralNetworkSuite()
        suite.set_networks(nn_config_data)

        assert len(suite.networks) == nn_config_data.num_networks
        for network in suite.networks:
            assert len(network.layers) == len(nn_config_data.hidden_layer_sizes) + 2
            assert network._num_inputs == nn_config_data.num_inputs
            assert network._input_layer._activation == nn_config_data.input_activation.get_class()

            assert all(
                layer._size == size
                for layer, size in zip(network._hidden_layers, nn_config_data.hidden_layer_sizes, strict=False)
            )
            assert all(
                layer._activation == nn_config_data.hidden_activation.get_class() for layer in network._hidden_layers
            )

            assert network._num_outputs == nn_config_data.num_outputs
            assert network._output_layer._activation == nn_config_data.output_activation.get_class()

            assert all(
                layer._weights_range == (nn_config_data.weights_min, nn_config_data.weights_max)
                and layer._bias_range == (nn_config_data.bias_min, nn_config_data.bias_max)
                for layer in network.layers[1:]
            )
