from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class FrameRequestData(_message.Message):
    __slots__ = ("observation", "fitness", "train_request")
    OBSERVATION_FIELD_NUMBER: _ClassVar[int]
    FITNESS_FIELD_NUMBER: _ClassVar[int]
    TRAIN_REQUEST_FIELD_NUMBER: _ClassVar[int]
    observation: Observation
    fitness: Fitness
    train_request: TrainRequest
    def __init__(self, observation: _Optional[_Union[Observation, _Mapping]] = ..., fitness: _Optional[_Union[Fitness, _Mapping]] = ..., train_request: _Optional[_Union[TrainRequest, _Mapping]] = ...) -> None: ...

class Observation(_message.Message):
    __slots__ = ("inputs",)
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    inputs: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, inputs: _Optional[_Iterable[float]] = ...) -> None: ...

class Action(_message.Message):
    __slots__ = ("outputs",)
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    outputs: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, outputs: _Optional[_Iterable[float]] = ...) -> None: ...

class Fitness(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, values: _Optional[_Iterable[float]] = ...) -> None: ...

class TrainRequest(_message.Message):
    __slots__ = ("observation", "action", "fitness")
    OBSERVATION_FIELD_NUMBER: _ClassVar[int]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    FITNESS_FIELD_NUMBER: _ClassVar[int]
    observation: _containers.RepeatedCompositeFieldContainer[Observation]
    action: _containers.RepeatedCompositeFieldContainer[Action]
    fitness: _containers.RepeatedCompositeFieldContainer[Fitness]
    def __init__(self, observation: _Optional[_Iterable[_Union[Observation, _Mapping]]] = ..., action: _Optional[_Iterable[_Union[Action, _Mapping]]] = ..., fitness: _Optional[_Iterable[_Union[Fitness, _Mapping]]] = ...) -> None: ...
