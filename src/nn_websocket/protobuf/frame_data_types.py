"""Dataclasses for neural network websocket Protobuf frame messages."""

from __future__ import annotations

from pydantic.dataclasses import dataclass

from nn_websocket.protobuf.compiled.FrameData_pb2 import (
    Action,
    Fitness,
    FrameRequestData,
    Observation,
    TrainRequest,
)


@dataclass
class FrameRequestDataType:
    """Data class to hold frame request data."""

    observation: ObservationType | None = None
    fitness: FitnessType | None = None
    train_request: TrainRequestType | None = None

    @classmethod
    def from_protobuf(cls, frame_request: FrameRequestData) -> FrameRequestDataType:
        """Create a FrameRequestDataType instance from Protobuf.

        :param FrameRequestData frame_request:
            The Protobuf FrameRequestData message.
        :return FrameRequestDataType:
            The corresponding FrameRequestDataType instance.
        """
        result = cls()

        which_oneof = frame_request.WhichOneof("msg")
        match which_oneof:
            case "observation":
                result.observation = ObservationType.from_protobuf(frame_request.observation)
            case "fitness":
                result.fitness = FitnessType.from_protobuf(frame_request.fitness)
            case "train_request":
                result.train_request = TrainRequestType.from_protobuf(frame_request.train_request)
            case _:
                pass

        return result

    @staticmethod
    def to_protobuf(frame_request_data: FrameRequestDataType) -> FrameRequestData:
        """Convert FrameRequestDataType to Protobuf.

        :param FrameRequestDataType frame_request_data:
            The FrameRequestDataType instance.
        :return FrameRequestData:
            The Protobuf FrameRequestData message.
        """
        frame_request = FrameRequestData()

        if frame_request_data.observation is not None:
            frame_request.observation.CopyFrom(ObservationType.to_protobuf(frame_request_data.observation))
        elif frame_request_data.fitness is not None:
            frame_request.fitness.CopyFrom(FitnessType.to_protobuf(frame_request_data.fitness))
        elif frame_request_data.train_request is not None:
            frame_request.train_request.CopyFrom(TrainRequestType.to_protobuf(frame_request_data.train_request))

        return frame_request

    @classmethod
    def from_bytes(cls, data: bytes) -> FrameRequestDataType:
        """Create a FrameRequestDataType instance from Protobuf bytes.

        :param bytes data:
            The serialized Protobuf bytes.
        :return FrameRequestDataType:
            The corresponding FrameRequestDataType instance.
        """
        frame_request = FrameRequestData()
        frame_request.ParseFromString(data)
        return cls.from_protobuf(frame_request)

    @staticmethod
    def to_bytes(frame_request_data: FrameRequestDataType) -> bytes:
        """Convert FrameRequestDataType to Protobuf bytes.

        :param FrameRequestDataType frame_request_data:
            The FrameRequestDataType instance.
        :return bytes:
            The serialized Protobuf bytes.
        """
        frame_request = FrameRequestDataType.to_protobuf(frame_request_data)
        return frame_request.SerializeToString()


@dataclass
class ObservationType:
    """Data class to hold observation data."""

    inputs: list[float]

    @classmethod
    def from_protobuf(cls, observation: Observation) -> ObservationType:
        """Create an ObservationType instance from Protobuf.

        :param Observation observation:
            The Protobuf Observation message.
        :return ObservationType:
            The corresponding ObservationType instance.
        """
        return cls(inputs=list(observation.inputs))

    @staticmethod
    def to_protobuf(observation_data: ObservationType) -> Observation:
        """Convert ObservationType to Protobuf.

        :param ObservationType observation_data:
            The ObservationType instance.
        :return Observation:
            The Protobuf Observation message.
        """
        return Observation(inputs=observation_data.inputs)

    @classmethod
    def from_bytes(cls, data: bytes) -> ObservationType:
        """Create an ObservationType instance from Protobuf bytes.

        :param bytes data:
            The serialized Protobuf bytes.
        :return ObservationType:
            The corresponding ObservationType instance.
        """
        observation = Observation()
        observation.ParseFromString(data)
        return cls.from_protobuf(observation)

    @staticmethod
    def to_bytes(observation_data: ObservationType) -> bytes:
        """Convert ObservationType to Protobuf bytes.

        :param ObservationType observation_data:
            The ObservationType instance.
        :return bytes:
            The serialized Protobuf bytes.
        """
        observation = ObservationType.to_protobuf(observation_data)
        return observation.SerializeToString()


@dataclass
class ActionType:
    """Data class to hold action data."""

    outputs: list[float]

    @classmethod
    def from_protobuf(cls, action: Action) -> ActionType:
        """Create an ActionType instance from Protobuf.

        :param Action action:
            The Protobuf Action message.
        :return ActionType:
            The corresponding ActionType instance.
        """
        return cls(outputs=list(action.outputs))

    @staticmethod
    def to_protobuf(action_data: ActionType) -> Action:
        """Convert ActionType to Protobuf.

        :param ActionType action_data:
            The ActionType instance.
        :return Action:
            The Protobuf Action message.
        """
        return Action(outputs=action_data.outputs)

    @classmethod
    def from_bytes(cls, data: bytes) -> ActionType:
        """Create an ActionType instance from Protobuf bytes.

        :param bytes data:
            The serialized Protobuf bytes.
        :return ActionType:
            The corresponding ActionType instance.
        """
        action = Action()
        action.ParseFromString(data)
        return cls.from_protobuf(action)

    @staticmethod
    def to_bytes(action_data: ActionType) -> bytes:
        """Convert ActionType to Protobuf bytes.

        :param ActionType action_data:
            The ActionType instance.
        :return bytes:
            The serialized Protobuf bytes.
        """
        action = ActionType.to_protobuf(action_data)
        return action.SerializeToString()


@dataclass
class FitnessType:
    """Data class to hold population fitness data."""

    values: list[float]

    @classmethod
    def from_protobuf(cls, fitness: Fitness) -> FitnessType:
        """Create a FitnessType instance from Protobuf.

        :param Fitness fitness:
            The Protobuf Fitness message.
        :return FitnessType:
            The corresponding FitnessType instance.
        """
        return cls(values=list(fitness.values))

    @staticmethod
    def to_protobuf(fitness_data: FitnessType) -> Fitness:
        """Convert FitnessType to Protobuf.

        :param FitnessType fitness_data:
            The FitnessType instance.
        :return Fitness:
            The Protobuf Fitness message.
        """
        return Fitness(values=fitness_data.values)

    @classmethod
    def from_bytes(cls, data: bytes) -> FitnessType:
        """Create a FitnessType instance from Protobuf bytes.

        :param bytes data:
            The serialized Protobuf bytes.
        :return FitnessType:
            The corresponding FitnessType instance.
        """
        fitness = Fitness()
        fitness.ParseFromString(data)
        return cls.from_protobuf(fitness)

    @staticmethod
    def to_bytes(fitness_data: FitnessType) -> bytes:
        """Convert FitnessType to Protobuf bytes.

        :param FitnessType fitness_data:
            The FitnessType instance.
        :return bytes:
            The serialized Protobuf bytes.
        """
        fitness = FitnessType.to_protobuf(fitness_data)
        return fitness.SerializeToString()


@dataclass
class TrainRequestType:
    """Data class to hold training request data."""

    observation: list[ObservationType]
    fitness: list[FitnessType]

    @classmethod
    def from_protobuf(cls, train_request: TrainRequest) -> TrainRequestType:
        """Create a TrainRequestType instance from Protobuf.

        :param TrainRequest train_request:
            The Protobuf TrainRequest message.
        :return TrainRequestType:
            The corresponding TrainRequestType instance.
        """
        return cls(
            observation=[ObservationType.from_protobuf(obs) for obs in train_request.observation],
            fitness=[FitnessType.from_protobuf(fit) for fit in train_request.fitness],
        )

    @classmethod
    def to_protobuf(cls, train_request: TrainRequestType) -> TrainRequest:
        """Convert TrainRequestType to Protobuf.

        :param TrainRequestType train_request:
            The TrainRequestType instance.
        :return TrainRequest:
            The Protobuf TrainRequest message.
        """
        return TrainRequest(
            observation=[ObservationType.to_protobuf(observation) for observation in train_request.observation],
            fitness=[FitnessType.to_protobuf(fitness) for fitness in train_request.fitness],
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> TrainRequestType:
        """Create a TrainRequestType instance from Protobuf bytes.

        :param bytes data:
            The serialized Protobuf bytes.
        :return TrainRequestType:
            The corresponding TrainRequestType instance.
        """
        train_request = TrainRequest()
        train_request.ParseFromString(data)
        return cls.from_protobuf(train_request)

    @staticmethod
    def to_bytes(train_request: TrainRequestType) -> bytes:
        """Convert TrainRequestType to Protobuf bytes.

        :param TrainRequestType train_request:
            The TrainRequestType instance.
        :return bytes:
            The serialized Protobuf bytes.
        """
        train_request_proto = TrainRequestType.to_protobuf(train_request)
        return train_request_proto.SerializeToString()
