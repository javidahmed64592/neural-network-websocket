"""Unit tests for the src/nn_websocket/protobuf/frame_data_types.py module."""

import pytest

from nn_websocket.protobuf.compiled.FrameData_pb2 import (
    Action,
    Fitness,
    FrameRequestData,
    Observation,
    TrainRequest,
)
from nn_websocket.protobuf.frame_data_types import (
    ActionType,
    FitnessType,
    FrameRequestDataType,
    ObservationType,
    TrainRequestType,
)


class TestFrameRequestData:
    """Test cases for FrameRequestDataType conversions and serialization."""

    def test_from_protobuf(
        self,
        frame_request_observation: FrameRequestData,
        frame_request_fitness: FrameRequestData,
        frame_request_train: FrameRequestData,
    ) -> None:
        """Test FrameRequestDataType.from_protobuf for all frame types."""
        frame_request_data_observation = FrameRequestDataType.from_protobuf(frame_request_observation)
        frame_request_data_fitness = FrameRequestDataType.from_protobuf(frame_request_fitness)
        frame_request_data_train = FrameRequestDataType.from_protobuf(frame_request_train)

        assert isinstance(frame_request_data_observation.observation, ObservationType)
        assert frame_request_data_observation.observation.inputs == pytest.approx(
            frame_request_observation.observation.inputs
        )
        assert isinstance(frame_request_data_fitness.fitness, FitnessType)
        assert frame_request_data_fitness.fitness.values == pytest.approx(frame_request_fitness.fitness.values)
        assert isinstance(frame_request_data_train.train_request, TrainRequestType)
        assert isinstance(frame_request_data_train.train_request.observation, list)
        assert isinstance(frame_request_data_train.train_request.action, list)
        assert isinstance(frame_request_data_train.train_request.fitness, list)

    def test_to_bytes(
        self,
        frame_request_data_observation: FrameRequestDataType,
        frame_request_data_fitness: FrameRequestDataType,
        frame_request_data_train: FrameRequestDataType,
    ) -> None:
        """Test serializing FrameRequestDataType to bytes for all frame types."""
        assert isinstance(FrameRequestDataType.to_bytes(frame_request_data_observation), bytes)
        assert isinstance(FrameRequestDataType.to_bytes(frame_request_data_fitness), bytes)
        assert isinstance(FrameRequestDataType.to_bytes(frame_request_data_train), bytes)
        assert isinstance(FrameRequestDataType.to_bytes(frame_request_data_train), bytes)

    def test_from_bytes(
        self,
        frame_request_data_observation: FrameRequestDataType,
        frame_request_data_fitness: FrameRequestDataType,
        frame_request_data_train: FrameRequestDataType,
    ) -> None:
        """Test deserializing FrameRequestDataType from bytes for all frame types."""
        assert frame_request_data_observation.observation is not None
        assert frame_request_data_fitness.fitness is not None
        assert frame_request_data_train.train_request is not None

        frame_request_observation_bytes = FrameRequestDataType.to_bytes(frame_request_data_observation)
        new_frame_request_observation = FrameRequestDataType.from_bytes(frame_request_observation_bytes)
        frame_request_population_bytes = FrameRequestDataType.to_bytes(frame_request_data_fitness)
        new_frame_request_population = FrameRequestDataType.from_bytes(frame_request_population_bytes)
        frame_request_train_bytes = FrameRequestDataType.to_bytes(frame_request_data_train)
        new_frame_request_train = FrameRequestDataType.from_bytes(frame_request_train_bytes)

        assert isinstance(new_frame_request_observation.observation, ObservationType)
        assert new_frame_request_observation.observation.inputs == pytest.approx(
            frame_request_data_observation.observation.inputs
        )
        assert isinstance(new_frame_request_population.fitness, FitnessType)
        assert new_frame_request_population.fitness.values == pytest.approx(frame_request_data_fitness.fitness.values)
        assert isinstance(new_frame_request_train.train_request, TrainRequestType)
        assert isinstance(new_frame_request_train.train_request.observation, list)
        assert isinstance(new_frame_request_train.train_request.action, list)
        assert isinstance(new_frame_request_train.train_request.fitness, list)


class TestObservationData:
    """Test cases for ObservationType conversions and serialization."""

    def test_from_protobuf(self, observation: Observation) -> None:
        """Test converting from Protobuf to ObservationType."""
        observation_data = ObservationType.from_protobuf(observation)
        assert isinstance(observation_data, ObservationType)
        assert observation_data.inputs == pytest.approx(observation.inputs)

    def test_to_protobuf(self, observation_data: ObservationType) -> None:
        """Test converting ObservationType to Protobuf."""
        observation = ObservationType.to_protobuf(observation_data)
        assert isinstance(observation, Observation)
        assert observation.inputs == pytest.approx(observation_data.inputs)

    def test_to_bytes(self, observation_data: ObservationType) -> None:
        """Test serializing ObservationType to bytes."""
        assert isinstance(ObservationType.to_bytes(observation_data), bytes)

    def test_from_bytes(self, observation_data: ObservationType) -> None:
        """Test deserializing ObservationType from bytes."""
        msg_bytes = ObservationType.to_bytes(observation_data)
        result = ObservationType.from_bytes(msg_bytes)

        assert result.inputs == pytest.approx(observation_data.inputs)


class TestActionData:
    """Test cases for ActionType conversions and serialization."""

    def test_from_protobuf(self, action: Action) -> None:
        """Test converting from Protobuf to ActionType."""
        action_data = ActionType.from_protobuf(action)
        assert isinstance(action_data, ActionType)
        assert action_data.outputs == pytest.approx(action.outputs)

    def test_to_protobuf(self, action_data: ActionType) -> None:
        """Test converting ActionType to Protobuf."""
        action = ActionType.to_protobuf(action_data)
        assert isinstance(action, Action)
        assert action.outputs == pytest.approx(action_data.outputs)

    def test_to_bytes(self, action_data: ActionType) -> None:
        """Test serializing ActionType to bytes."""
        assert isinstance(ActionType.to_bytes(action_data), bytes)

    def test_from_bytes(self, action_data: ActionType) -> None:
        """Test deserializing ActionType from bytes."""
        msg_bytes = ActionType.to_bytes(action_data)
        result = ActionType.from_bytes(msg_bytes)

        assert result.outputs == pytest.approx(action_data.outputs)


class TestFitnessData:
    """Test cases for FitnessType conversions and serialization."""

    def test_from_protobuf(self, fitness: Fitness) -> None:
        """Test converting from Protobuf to FitnessType."""
        fitness_data = FitnessType.from_protobuf(fitness)
        assert isinstance(fitness_data, FitnessType)
        assert fitness_data.values == pytest.approx(fitness.values)

    def test_to_protobuf(self, fitness_data: FitnessType) -> None:
        """Test converting FitnessType to Protobuf."""
        fitness = FitnessType.to_protobuf(fitness_data)
        assert isinstance(fitness, Fitness)
        assert fitness.values == pytest.approx(fitness_data.values)

    def test_to_bytes(self, fitness_data: FitnessType) -> None:
        """Test serializing FitnessType to bytes."""
        assert isinstance(FitnessType.to_bytes(fitness_data), bytes)

    def test_from_bytes(self, fitness_data: FitnessType) -> None:
        """Test deserializing FitnessType from bytes."""
        msg_bytes = FitnessType.to_bytes(fitness_data)
        result = FitnessType.from_bytes(msg_bytes)

        assert result.values == pytest.approx(fitness_data.values)


class TestTrainRequestData:
    """Test cases for TrainRequestType conversions and serialization."""

    def test_from_protobuf(self, train_request: TrainRequest) -> None:
        """Test converting from Protobuf to TrainRequestType."""
        train_request_data = TrainRequestType.from_protobuf(train_request)
        assert isinstance(train_request_data, TrainRequestType)
        assert isinstance(train_request_data.observation, list)
        assert isinstance(train_request_data.action, list)
        assert isinstance(train_request_data.fitness, list)

    def test_to_protobuf(self, train_request_data: TrainRequestType) -> None:
        """Test converting TrainRequestType to Protobuf."""
        train_request = TrainRequestType.to_protobuf(train_request_data)
        assert isinstance(train_request, TrainRequest)
        assert len(train_request.observation) == len(train_request_data.observation)
        assert len(train_request.action) == len(train_request_data.action)
        assert len(train_request.fitness) == len(train_request_data.fitness)

    def test_to_bytes(self, train_request_data: TrainRequestType) -> None:
        """Test serializing TrainRequestType to bytes."""
        assert isinstance(TrainRequestType.to_bytes(train_request_data), bytes)

    def test_from_bytes(self, train_request_data: TrainRequestType) -> None:
        """Test deserializing TrainRequestType from bytes."""
        msg_bytes = TrainRequestType.to_bytes(train_request_data)
        result = TrainRequestType.from_bytes(msg_bytes)

        assert isinstance(result.observation, list)
        assert isinstance(result.action, list)
        assert isinstance(result.fitness, list)

        assert len(result.observation) == len(train_request_data.observation)
        assert len(result.action) == len(train_request_data.action)
        assert len(result.fitness) == len(train_request_data.fitness)

        assert isinstance(result.observation[0], ObservationType)
        assert isinstance(result.action[0], ActionType)
        assert isinstance(result.fitness[0], FitnessType)

        assert result.observation[0].inputs == pytest.approx(train_request_data.observation[0].inputs)
        assert result.action[0].outputs == pytest.approx(train_request_data.action[0].outputs)
        assert result.fitness[0].values == pytest.approx(train_request_data.fitness[0].values)
