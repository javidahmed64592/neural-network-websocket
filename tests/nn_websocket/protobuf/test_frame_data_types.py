import pytest

from nn_websocket.protobuf.compiled.FrameData_pb2 import Action, Fitness, FrameRequest, Observation, TrainRequest
from nn_websocket.protobuf.frame_data_types import (
    ActionData,
    FitnessData,
    FrameRequestData,
    ObservationData,
    TrainRequestData,
)


class TestFrameRequestData:
    def test_from_protobuf(
        self,
        frame_request_observation: FrameRequest,
        frame_request_fitness: FrameRequest,
        frame_request_train: FrameRequest,
    ) -> None:
        frame_request_data_observation = FrameRequestData.from_protobuf(frame_request_observation)
        frame_request_data_fitness = FrameRequestData.from_protobuf(frame_request_fitness)
        frame_request_data_train = FrameRequestData.from_protobuf(frame_request_train)

        assert isinstance(frame_request_data_observation.observation, ObservationData)
        assert frame_request_data_observation.observation.inputs == pytest.approx(
            frame_request_observation.observation.inputs
        )
        assert isinstance(frame_request_data_fitness.fitness, FitnessData)
        assert frame_request_data_fitness.fitness.values == pytest.approx(frame_request_fitness.fitness.values)
        assert isinstance(frame_request_data_train.train_request, TrainRequestData)
        assert isinstance(frame_request_data_train.train_request.observation, list)
        assert isinstance(frame_request_data_train.train_request.fitness, list)

    def test_to_bytes(
        self,
        frame_request_data_observation: FrameRequestData,
        frame_request_data_fitness: FrameRequestData,
        frame_request_data_train: FrameRequestData,
    ) -> None:
        """Test serializing a FrameRequestData with observation data."""
        assert isinstance(FrameRequestData.to_bytes(frame_request_data_observation), bytes)
        assert isinstance(FrameRequestData.to_bytes(frame_request_data_fitness), bytes)
        assert isinstance(FrameRequestData.to_bytes(frame_request_data_train), bytes)
        assert isinstance(FrameRequestData.to_bytes(frame_request_data_train), bytes)

    def test_from_bytes(
        self,
        frame_request_data_observation: FrameRequestData,
        frame_request_data_fitness: FrameRequestData,
        frame_request_data_train: FrameRequestData,
    ) -> None:
        """Test deserializing a FrameRequestData with observation data."""
        assert frame_request_data_observation.observation is not None
        assert frame_request_data_fitness.fitness is not None
        assert frame_request_data_train.train_request is not None

        frame_request_observation_bytes = FrameRequestData.to_bytes(frame_request_data_observation)
        new_frame_request_observation = FrameRequestData.from_bytes(frame_request_observation_bytes)
        frame_request_population_bytes = FrameRequestData.to_bytes(frame_request_data_fitness)
        new_frame_request_population = FrameRequestData.from_bytes(frame_request_population_bytes)
        frame_request_train_bytes = FrameRequestData.to_bytes(frame_request_data_train)
        new_frame_request_train = FrameRequestData.from_bytes(frame_request_train_bytes)

        assert isinstance(new_frame_request_observation.observation, ObservationData)
        assert new_frame_request_observation.observation.inputs == pytest.approx(
            frame_request_data_observation.observation.inputs
        )
        assert isinstance(new_frame_request_population.fitness, FitnessData)
        assert new_frame_request_population.fitness.values == pytest.approx(frame_request_data_fitness.fitness.values)
        assert isinstance(new_frame_request_train.train_request, TrainRequestData)
        assert isinstance(new_frame_request_train.train_request.observation, list)
        assert isinstance(new_frame_request_train.train_request.fitness, list)


class TestObservationData:
    def test_from_protobuf(self, observation: Observation) -> None:
        """Test converting from Protobuf to ObservationData."""
        observation_data = ObservationData.from_protobuf(observation)
        assert isinstance(observation_data, ObservationData)
        assert observation_data.inputs == pytest.approx(observation.inputs)

    def test_to_protobuf(self, observation_data: ObservationData) -> None:
        """Test converting ObservationData to Protobuf."""
        observation = ObservationData.to_protobuf(observation_data)
        assert isinstance(observation, Observation)
        assert observation.inputs == pytest.approx(observation_data.inputs)

    def test_to_bytes(self, observation_data: ObservationData) -> None:
        assert isinstance(ObservationData.to_bytes(observation_data), bytes)

    def test_from_bytes(self, observation_data: ObservationData) -> None:
        msg_bytes = ObservationData.to_bytes(observation_data)
        result = ObservationData.from_bytes(msg_bytes)

        assert result.inputs == pytest.approx(observation_data.inputs)


class TestActionData:
    def test_from_protobuf(self, action: Action) -> None:
        """Test converting from Protobuf to ActionData."""
        action_data = ActionData.from_protobuf(action)
        assert isinstance(action_data, ActionData)
        assert action_data.outputs == pytest.approx(action.outputs)

    def test_to_protobuf(self, action_data: ActionData) -> None:
        """Test converting ActionData to Protobuf."""
        action = ActionData.to_protobuf(action_data)
        assert isinstance(action, Action)
        assert action.outputs == pytest.approx(action_data.outputs)

    def test_to_bytes(self, action_data: ActionData) -> None:
        assert isinstance(ActionData.to_bytes(action_data), bytes)

    def test_from_bytes(self, action_data: ActionData) -> None:
        msg_bytes = ActionData.to_bytes(action_data)
        result = ActionData.from_bytes(msg_bytes)

        assert result.outputs == pytest.approx(action_data.outputs)


class TestFitnessData:
    def test_from_protobuf(self, fitness: Fitness) -> None:
        """Test converting from Protobuf to FitnessData."""
        fitness_data = FitnessData.from_protobuf(fitness)
        assert isinstance(fitness_data, FitnessData)
        assert fitness_data.values == pytest.approx(fitness.values)

    def test_to_protobuf(self, fitness_data: FitnessData) -> None:
        """Test converting FitnessData to Protobuf."""
        fitness = FitnessData.to_protobuf(fitness_data)
        assert isinstance(fitness, Fitness)
        assert fitness.values == pytest.approx(fitness_data.values)

    def test_to_bytes(self, fitness_data: FitnessData) -> None:
        assert isinstance(FitnessData.to_bytes(fitness_data), bytes)

    def test_from_bytes(self, fitness_data: FitnessData) -> None:
        msg_bytes = FitnessData.to_bytes(fitness_data)
        result = FitnessData.from_bytes(msg_bytes)

        assert result.values == pytest.approx(fitness_data.values)


class TestTrainRequestData:
    def test_from_protobuf(self, train_request: TrainRequest) -> None:
        """Test converting from Protobuf to TrainRequestData."""
        train_request_data = TrainRequestData.from_protobuf(train_request)
        assert isinstance(train_request_data, TrainRequestData)
        assert isinstance(train_request_data.observation, list)
        assert isinstance(train_request_data.fitness, list)

    def test_to_protobuf(self, train_request_data: TrainRequestData) -> None:
        """Test converting TrainRequestData to Protobuf."""
        train_request = TrainRequestData.to_protobuf(train_request_data)
        assert isinstance(train_request, TrainRequest)
        assert len(train_request.observation) == len(train_request_data.observation)
        assert len(train_request.fitness) == len(train_request_data.fitness)

    def test_to_bytes(self, train_request_data: TrainRequestData) -> None:
        assert isinstance(TrainRequestData.to_bytes(train_request_data), bytes)

    def test_from_bytes(self, train_request_data: TrainRequestData) -> None:
        msg_bytes = TrainRequestData.to_bytes(train_request_data)
        result = TrainRequestData.from_bytes(msg_bytes)

        assert isinstance(result.observation, list)
        assert isinstance(result.fitness, list)

        assert len(result.observation) == len(train_request_data.observation)
        assert len(result.fitness) == len(train_request_data.fitness)

        assert isinstance(result.observation[0], ObservationData)
        assert isinstance(result.fitness[0], FitnessData)

        assert result.observation[0].inputs == pytest.approx(train_request_data.observation[0].inputs)
        assert result.fitness[0].values == pytest.approx(train_request_data.fitness[0].values)
