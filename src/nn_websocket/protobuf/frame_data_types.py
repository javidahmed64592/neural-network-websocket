"""Dataclasses for neural network websocket Protobuf frame messages."""

from __future__ import annotations

from pydantic.dataclasses import dataclass

from nn_websocket.protobuf.compiled.FrameRequestClasses_pb2 import (
    Action,
    Fitness,
    FrameRequest,
    Observation,
    TrainRequest,
)


@dataclass
class FrameRequestData:
    """Data class to hold frame request data."""

    observation: ObservationData | None = None
    fitness: FitnessData | None = None
    train_request: TrainRequestData | None = None

    @classmethod
    def from_protobuf(cls, frame_request: FrameRequest) -> FrameRequestData:
        """Create a FrameRequestData instance from Protobuf.

        :param FrameRequest frame_request:
            The Protobuf FrameRequest message.
        :return FrameRequestData:
            The corresponding FrameRequestData instance.
        """
        result = cls()

        which_oneof = frame_request.WhichOneof("msg")
        match which_oneof:
            case "observation":
                result.observation = ObservationData.from_protobuf(frame_request.observation)
            case "fitness":
                result.fitness = FitnessData.from_protobuf(frame_request.fitness)
            case "train_request":
                result.train_request = TrainRequestData.from_protobuf(frame_request.train_request)
            case _:
                pass

        return result

    @staticmethod
    def to_protobuf(frame_request_data: FrameRequestData) -> FrameRequest:
        """Convert FrameRequestData to Protobuf.

        :param FrameRequestData frame_request_data:
            The FrameRequestData instance.
        :return FrameRequest:
            The Protobuf FrameRequest message.
        """
        frame_request = FrameRequest()

        if frame_request_data.observation is not None:
            frame_request.observation.CopyFrom(ObservationData.to_protobuf(frame_request_data.observation))
        elif frame_request_data.fitness is not None:
            frame_request.fitness.CopyFrom(FitnessData.to_protobuf(frame_request_data.fitness))
        elif frame_request_data.train_request is not None:
            frame_request.train_request.CopyFrom(TrainRequestData.to_protobuf(frame_request_data.train_request))

        return frame_request

    @classmethod
    def from_bytes(cls, data: bytes) -> FrameRequestData:
        """Create a FrameRequestData instance from Protobuf bytes.

        :param bytes data:
            The serialized Protobuf bytes.
        :return FrameRequestData:
            The corresponding FrameRequestData instance.
        """
        frame_request = FrameRequest()
        frame_request.ParseFromString(data)
        return cls.from_protobuf(frame_request)

    @staticmethod
    def to_bytes(frame_request_data: FrameRequestData) -> bytes:
        """Convert FrameRequestData to Protobuf bytes.

        :param FrameRequestData frame_request_data:
            The FrameRequestData instance.
        :return bytes:
            The serialized Protobuf bytes.
        """
        frame_request = FrameRequestData.to_protobuf(frame_request_data)
        return frame_request.SerializeToString()


@dataclass
class ObservationData:
    """Data class to hold observation data."""

    inputs: list[float]

    @classmethod
    def from_protobuf(cls, observation: Observation) -> ObservationData:
        """Create an ObservationData instance from Protobuf.

        :param Observation observation:
            The Protobuf Observation message.
        :return ObservationData:
            The corresponding ObservationData instance.
        """
        return cls(inputs=list(observation.inputs))

    @staticmethod
    def to_protobuf(observation_data: ObservationData) -> Observation:
        """Convert ObservationData to Protobuf.

        :param ObservationData observation_data:
            The ObservationData instance.
        :return Observation:
            The Protobuf Observation message.
        """
        return Observation(inputs=observation_data.inputs)

    @classmethod
    def from_bytes(cls, data: bytes) -> ObservationData:
        """Create an ObservationData instance from Protobuf bytes.

        :param bytes data:
            The serialized Protobuf bytes.
        :return ObservationData:
            The corresponding ObservationData instance.
        """
        observation = Observation()
        observation.ParseFromString(data)
        return cls.from_protobuf(observation)

    @staticmethod
    def to_bytes(observation_data: ObservationData) -> bytes:
        """Convert ObservationData to Protobuf bytes.

        :param ObservationData observation_data:
            The ObservationData instance.
        :return bytes:
            The serialized Protobuf bytes.
        """
        observation = ObservationData.to_protobuf(observation_data)
        return observation.SerializeToString()


@dataclass
class ActionData:
    """Data class to hold action data."""

    outputs: list[float]

    @classmethod
    def from_protobuf(cls, action: Action) -> ActionData:
        """Create an ActionData instance from Protobuf.

        :param Action action:
            The Protobuf Action message.
        :return ActionData:
            The corresponding ActionData instance.
        """
        return cls(outputs=list(action.outputs))

    @staticmethod
    def to_protobuf(action_data: ActionData) -> Action:
        """Convert ActionData to Protobuf.

        :param ActionData action_data:
            The ActionData instance.
        :return Action:
            The Protobuf Action message.
        """
        return Action(outputs=action_data.outputs)

    @classmethod
    def from_bytes(cls, data: bytes) -> ActionData:
        """Create an ActionData instance from Protobuf bytes.

        :param bytes data:
            The serialized Protobuf bytes.
        :return ActionData:
            The corresponding ActionData instance.
        """
        action = Action()
        action.ParseFromString(data)
        return cls.from_protobuf(action)

    @staticmethod
    def to_bytes(action_data: ActionData) -> bytes:
        """Convert ActionData to Protobuf bytes.

        :param ActionData action_data:
            The ActionData instance.
        :return bytes:
            The serialized Protobuf bytes.
        """
        action = ActionData.to_protobuf(action_data)
        return action.SerializeToString()


@dataclass
class FitnessData:
    """Data class to hold population fitness data."""

    values: list[float]

    @classmethod
    def from_protobuf(cls, fitness: Fitness) -> FitnessData:
        """Create a FitnessData instance from Protobuf.

        :param Fitness fitness:
            The Protobuf Fitness message.
        :return FitnessData:
            The corresponding FitnessData instance.
        """
        return cls(values=list(fitness.values))

    @staticmethod
    def to_protobuf(fitness_data: FitnessData) -> Fitness:
        """Convert FitnessData to Protobuf.

        :param FitnessData fitness_data:
            The FitnessData instance.
        :return Fitness:
            The Protobuf Fitness message.
        """
        return Fitness(values=fitness_data.values)

    @classmethod
    def from_bytes(cls, data: bytes) -> FitnessData:
        """Create a FitnessData instance from Protobuf bytes.

        :param bytes data:
            The serialized Protobuf bytes.
        :return FitnessData:
            The corresponding FitnessData instance.
        """
        fitness = Fitness()
        fitness.ParseFromString(data)
        return cls.from_protobuf(fitness)

    @staticmethod
    def to_bytes(fitness_data: FitnessData) -> bytes:
        """Convert FitnessData to Protobuf bytes.

        :param FitnessData fitness_data:
            The FitnessData instance.
        :return bytes:
            The serialized Protobuf bytes.
        """
        fitness = FitnessData.to_protobuf(fitness_data)
        return fitness.SerializeToString()


@dataclass
class TrainRequestData:
    """Data class to hold training request data."""

    observation: list[ObservationData]
    fitness: list[FitnessData]

    @classmethod
    def from_protobuf(cls, train_request: TrainRequest) -> TrainRequestData:
        """Create a TrainRequestData instance from Protobuf.

        :param TrainRequest train_request:
            The Protobuf TrainRequest message.
        :return TrainRequestData:
            The corresponding TrainRequestData instance.
        """
        return cls(
            observation=[ObservationData.from_protobuf(obs) for obs in train_request.observation],
            fitness=[FitnessData.from_protobuf(fit) for fit in train_request.fitness],
        )

    @classmethod
    def to_protobuf(cls, train_request: TrainRequestData) -> TrainRequest:
        """Convert TrainRequestData to Protobuf.

        :param TrainRequestData train_request:
            The TrainRequestData instance.
        :return TrainRequest:
            The Protobuf TrainRequest message.
        """
        return TrainRequest(
            observation=[ObservationData.to_protobuf(observation) for observation in train_request.observation],
            fitness=[FitnessData.to_protobuf(fitness) for fitness in train_request.fitness],
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> TrainRequestData:
        """Create a TrainRequestData instance from Protobuf bytes.

        :param bytes data:
            The serialized Protobuf bytes.
        :return TrainRequestData:
            The corresponding TrainRequestData instance.
        """
        train_request = TrainRequest()
        train_request.ParseFromString(data)
        return cls.from_protobuf(train_request)

    @staticmethod
    def to_bytes(train_request: TrainRequestData) -> bytes:
        """Convert TrainRequestData to Protobuf bytes.

        :param TrainRequestData train_request:
            The TrainRequestData instance.
        :return bytes:
            The serialized Protobuf bytes.
        """
        train_request_proto = TrainRequestData.to_protobuf(train_request)
        return train_request_proto.SerializeToString()
