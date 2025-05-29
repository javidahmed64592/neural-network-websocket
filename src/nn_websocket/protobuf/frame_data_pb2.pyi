from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ObservationBatch(_message.Message):
    __slots__ = ("observations",)
    OBSERVATIONS_FIELD_NUMBER: _ClassVar[int]
    observations: _containers.RepeatedCompositeFieldContainer[Observation]
    def __init__(self, observations: _Optional[_Iterable[_Union[Observation, _Mapping]]] = ...) -> None: ...

class Observation(_message.Message):
    __slots__ = ("agent_id", "inputs")
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    agent_id: int
    inputs: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, agent_id: _Optional[int] = ..., inputs: _Optional[_Iterable[float]] = ...) -> None: ...

class OutputBatch(_message.Message):
    __slots__ = ("outputs",)
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    outputs: _containers.RepeatedCompositeFieldContainer[Output]
    def __init__(self, outputs: _Optional[_Iterable[_Union[Output, _Mapping]]] = ...) -> None: ...

class Output(_message.Message):
    __slots__ = ("agent_id", "actions")
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    ACTIONS_FIELD_NUMBER: _ClassVar[int]
    agent_id: int
    actions: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, agent_id: _Optional[int] = ..., actions: _Optional[_Iterable[float]] = ...) -> None: ...
