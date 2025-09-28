"""Core type definitions for RAG systems."""

from collections.abc import Sequence
from typing import TypeVar

import numpy as np
from numpy import floating, integer, number
from numpy.typing import NDArray

T = TypeVar("T")

PyEmbedding = Sequence[float] | Sequence[int]
PyEmbeddings = list[PyEmbedding]
Embedding = NDArray[np.int32 | np.float32]
Embeddings = list[Embedding]

Documents = list[str]
Images = list[np.ndarray]
Embeddable = Documents | Images

ScalarType = TypeVar("ScalarType", bound=np.generic)
IntegerType = TypeVar("IntegerType", bound=integer)
FloatingType = TypeVar("FloatingType", bound=floating)
NumberType = TypeVar("NumberType", bound=number)

DType32 = TypeVar("DType32", np.int32, np.float32)
DType64 = TypeVar("DType64", np.int64, np.float64)
DTypeCommon = TypeVar("DTypeCommon", np.int32, np.int64, np.float32, np.float64)
