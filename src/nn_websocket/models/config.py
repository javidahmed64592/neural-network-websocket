from pydantic.dataclasses import dataclass


@dataclass
class Config:
    """
    Configuration for the WebSocket server.
    """

    host: str = "localhost"
    port: int = 8765
