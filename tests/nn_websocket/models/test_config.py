from nn_websocket.models.config import Config


class TestConfig:
    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        custom_host = "custom_host"
        custom_port = 1234
        config = Config(host=custom_host, port=custom_port)
        assert config.host == custom_host
        assert config.port == custom_port
