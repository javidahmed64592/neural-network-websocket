from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic.dataclasses import dataclass


@dataclass
class Config:
    """
    Configuration for the Websocket server.
    """

    host: str
    port: int

    @property
    def uri(self) -> str:
        """
        Returns the WebSocket URI based on the host and port.
        """
        return f"ws://{self.host}:{self.port}"

    @staticmethod
    def load_config(filepath: os.PathLike) -> Config:
        with Path(filepath).open() as f:
            config_dict = json.load(f)
        return Config(**config_dict)
