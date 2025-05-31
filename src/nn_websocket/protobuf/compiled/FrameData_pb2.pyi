from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

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

class PopulationFitness(_message.Message):
    __slots__ = ("fitness",)
    FITNESS_FIELD_NUMBER: _ClassVar[int]
    fitness: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, fitness: _Optional[_Iterable[float]] = ...) -> None: ...
