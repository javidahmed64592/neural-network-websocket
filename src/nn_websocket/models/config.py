from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic.dataclasses import dataclass


@dataclass
class Config:
    """
    Configuration for the WebSocket server.
    """

    host: str
    port: int

    @staticmethod
    def load_config(filepath: os.PathLike) -> Config:
        with Path(filepath).open() as f:
            config_dict = json.load(f)
        return Config(**config_dict)
