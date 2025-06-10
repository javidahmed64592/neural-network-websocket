"""Configuration dataclass and loader for the neural network websocket server."""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic.dataclasses import dataclass


@dataclass
class Config:
    """Configuration for the Websocket server."""

    host: str
    port: int

    @property
    def uri(self) -> str:
        """Return the Websocket URI based on the host and port.

        :return str:
            The websocket URI string.
        """
        return f"ws://{self.host}:{self.port}"

    @staticmethod
    def load_config(filepath: os.PathLike) -> Config:
        """Load configuration from a JSON file.

        :param os.PathLike filepath:
            Path to the configuration file.
        :return Config:
            The loaded configuration object.
        """
        with Path(filepath).open() as f:
            config_dict = json.load(f)
        return Config(**config_dict)
