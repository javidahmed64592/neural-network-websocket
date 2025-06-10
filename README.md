[![python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=ffd343)](https://docs.python.org/3.12/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

<!-- omit from toc -->
# Neural Network Websocket

A Python websocket server and client toolkit for communicating with neural networks using Protobuf-encoded messages. This project enables real-time, efficient communication between clients and a neural network backend, supporting both neuroevolution and fitness-based training paradigms. It includes a server, mock clients for testing, and utilities for message encoding/decoding.

<!-- omit from toc -->
## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Websocket Server](#running-the-websocket-server)
- [Running the Mock Clients](#running-the-mock-clients)
  - [Neuroevolution Client](#neuroevolution-client)
  - [Fitness Client](#fitness-client)
- [Protobuf Integration](#protobuf-integration)
  - [Compiling Protobuf Files](#compiling-protobuf-files)
- [Development](#development)
- [Testing, Linting, and Type Checking](#testing-linting-and-type-checking)
- [License](#license)

## Overview

This application provides a websocket server that interfaces with neural networks for tasks such as inference, training, and neuroevolution. Communication between clients and the server is handled using Protocol Buffers (Protobuf) for efficient, strongly-typed message serialization.

Two mock clients are included:
- **Neuroevolution Client:** Simulates a population-based evolutionary approach.
- **Fitness Client:** Simulates a single-network fitness-based training approach.

## Features

- **Websocket Server:** Handles real-time neural network requests and training instructions.
- **Protobuf Messaging:** All data (observations, actions, fitness, configuration) is serialized using Protobuf for speed and interoperability.
- **Mock Clients:** Easily test server behavior with randomized data.
- **Configurable:** Server and client behavior can be customized via config files and code.

## Project Structure

```
config/
  websocket_config.json      # Server configuration file
protobuf/                    # Protobuf message definitions
src/
  nn_websocket/
    main.py                  # Websocket server entry point
    ga/                      # Genetic algorithm integration
    models/                  # Data models and configuration
    protobuf/                # Data classes for parsing Protobuf
    tools/
      neuroevolution_client.py  # Mock neuroevolution client
      fitness_client.py         # Mock fitness client
      base_client.py            # Base client logic
      client_utils.py           # Utilities for generating random frames
```

---

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/javidahmed64592/neural-network-websocket.git
    cd neural-network-websocket
    ```

2. **Install dependencies:**
    ```sh
    pip install -e .[dev]
    ```

---

## Configuration

The server reads its host and port from `config/websocket_config.json`:

```json
{
  "host": "localhost",
  "port": 8765
}
```

Edit this file to change the server's network settings.

## Running the Websocket Server

Start the server using the provided script:

```sh
nn-websocket
```

Or directly:

```sh
python -m nn_websocket.main
```

The server will listen for websocket connections and handle Protobuf-encoded messages for neural network inference and training.

## Running the Mock Clients

### Neuroevolution Client

Simulates a population of agents sending observations and fitness data.

```sh
neuroevolution-client
```

Or:

```sh
python -m nn_websocket.tools.neuroevolution_client
```

### Fitness Client

Simulates a single agent sending observations and training batches.

```sh
fitness-client
```

Or:

```sh
python -m nn_websocket.tools.fitness_client
```

Both clients connect to the server, send randomized data, and log responses for testing purposes.

## Protobuf Integration

All communication between clients and the server uses Protocol Buffers for message serialization. The `.proto` files are located in `protobuf/` and compiled Python classes are generated in `src/nn_websocket/protobuf/compiled/`.

### Compiling Protobuf Files

If you modify or add `.proto` files, recompile them with:

```sh
compile-websocket-protobuf
```

Or:

```sh
python -m nn_websocket.protobuf.compile_protobuf
```

This will regenerate the Python classes used for message encoding/decoding.

## Development

- The server and clients use the `websockets` library for async communication.
- Neural network and genetic algorithm logic is integrated via external packages (`neural-network`, `genetic-algorithm`).
- Utilities are provided for generating random test data and encoding/decoding Protobuf messages.

## Testing, Linting, and Type Checking

- **Run tests:** `python -m pytest tests`
- **Lint code:** `python -m ruff check .`
- **Format code:** `python -m ruff format .`
- **Type check:** `python -m mypy .`

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
