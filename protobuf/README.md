# Protobuf Message Definitions

This directory contains the Protocol Buffer (`.proto`) files used for defining the message formats exchanged between the neural network websocket server and its clients. These definitions ensure efficient, strongly-typed, and language-agnostic communication for neural network inference, training, and configuration.

## Overview

The `.proto` files in this folder describe the structure of messages for:

- **Neural network configuration and training** (see `NNWebsocketClasses.proto`)
- **Frame data exchange between client and server** (see `FrameRequestClasses.proto`)

These files are compiled into Python classes (and can be compiled for other languages) to enable serialization and deserialization of messages over the network.

## Files

- **NNWebsocketClasses.proto**
  Defines messages for configuring neural networks, specifying training methods (neuroevolution or fitness-based), and genetic algorithm parameters.

- **FrameRequestClasses.proto**
  Defines messages for exchanging observations, actions, fitness values, and training requests between clients and the server.

## Message Structure

### NNWebsocketClasses.proto

- **ActivationFunctionEnum**: Enumerates supported activation functions (LINEAR, RELU, SIGMOID, TANH).
- **NeuralNetworkConfig**: Specifies neural network architecture and hyperparameters.
- **Configuration**: Uses a `oneof` to select between neuroevolution and fitness-based training configurations.
- **GeneticAlgorithmConfig**: Parameters for population size and mutation rate.
- **NeuroevolutionConfig**: Configuration for neuroevolution training.
- **FitnessApproachConfig**: Configuration for fitness-based training.

### FrameRequestClasses.proto

- **FrameRequest**: Uses a `oneof` to encapsulate either an observation, fitness value, or training request.
- **Observation**: Input data for the neural network.
- **Action**: Output data from the neural network.
- **Fitness**: Fitness values for neuroevolution.
- **TrainRequest**: Batch of observations and fitness values for fitness-based training.

## Compilation

To use these Protobuf definitions in Python, compile them using the `protoc` compiler. The generated Python files are placed in `src/nn_websocket/protobuf/compiled/`.

### Compile All Protobuf Files

From the project root, run:

```sh
compile-websocket-protobuf
python -m nn_websocket.protobuf.compile_protobuf # Or run directly
```

Or, manually with `protoc`:

```sh
python -m grpc_tools.protoc -I=protobuf --python_out=src/nn_websocket/protobuf/compiled protobuf/NNWebsocketClasses.proto
python -m grpc_tools.protoc -I=protobuf --python_out=src/nn_websocket/protobuf/compiled protobuf/FrameRequestClasses.proto
```

_Note: If you modify any `.proto` files, you must recompile them for changes to take effect._

## Usage

- **Server and clients** import the generated Python classes to encode and decode messages.
- The message formats are designed to be extensible and compatible with other languages that support Protobuf.

## References

- [Protocol Buffers Documentation](https://developers.google.com/protocol-buffers)
- [grpc_tools Python Package](https://pypi.org/project/grpcio-tools/)

---
