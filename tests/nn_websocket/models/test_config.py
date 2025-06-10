import json
from pathlib import Path
from unittest.mock import mock_open, patch

from nn_websocket.models.config import Config


class TestConfig:
    def test_uri_property(self) -> None:
        """Test the URI property of the Config class."""
        config = Config(host="localhost", port=8080)
        assert config.uri == "ws://localhost:8080"

    def test_load_config(self, mock_config: Config) -> None:
        """Test loading configuration from a file."""
        mock_data = json.dumps({"host": mock_config.host, "port": mock_config.port})
        with patch("nn_websocket.models.config.Path.open", mock_open(read_data=mock_data)):
            config = Config.load_config(Path("dummy_path"))
            assert config.host == mock_config.host
            assert config.port == mock_config.port
