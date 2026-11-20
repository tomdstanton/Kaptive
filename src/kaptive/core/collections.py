r"""Protocol contracts for Structure-of-Arrays (SoA) and batched containers.

This module defines formal protocols ([`BatchedContainer`][kaptive.core.collections.BatchedContainer],
[`RecordContainer`][kaptive.core.collections.RecordContainer], and
[`RaggedArrayContainer`][kaptive.core.collections.RaggedArrayContainer]) that govern high-performance,
vectorized containers across Kaptive.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, Self, TypeAlias, TypeVar, overload

import numpy as np
import numpy.typing as npt

ScalarIndex: TypeAlias = int | np.integer[Any]
BatchIndex: TypeAlias = slice | npt.NDArray[np.integer[Any]] | npt.NDArray[np.bool_] | list[int]

T = TypeVar("T", covariant=True)


class BatchedContainer(Protocol):
    r"""A formal contract for sliceable Structure-of-Arrays (SoA) and batched containers in Kaptive.

    This protocol ensures that all high-performance vectorized collections provide a standard
    interface for instantiation, concatenation, and batch slicing/masking without relying on slow
    dynamic mixins or runtime type introspection.
    """

    def __len__(self) -> int:
        r"""Return the number of records in the batch.

        Returns:
            int: The total count of items in the container.
        """
        ...

    def __getitem__(self, item: BatchIndex) -> BatchedContainer:
        r"""Access a subset of records by slice or boolean/integer array mask.

        Args:
            item (BatchIndex): A slice, boolean mask array, or integer indices array/list.

        Returns:
            BatchedContainer: A new batched collection containing the selected records.
        """
        ...

    @classmethod
    def empty(cls) -> Self:
        r"""Create an empty, 0-length collection with correctly typed arrays.

        Returns:
            Self: An empty instance of the container.
        """
        ...

    @classmethod
    def concat(cls, batches: Iterable[Self]) -> Self:
        r"""Concatenate multiple collections into a single, larger collection.

        Args:
            batches (Iterable[Self]): An iterable of collections of the same type.

        Returns:
            Self: A new, combined collection.

        Raises:
            ValueError: If the input iterable is empty or contains incompatible batches.
        """
        ...


class RecordContainer(BatchedContainer, Protocol[T]):
    r"""A formal contract for batched containers that also yield scalar records on integer indexing.

    Type Parameters:
        T: The scalar record type returned when accessing a single integer index.
    """

    @overload
    def __getitem__(self, item: ScalarIndex) -> T: ...

    @overload
    def __getitem__(self, item: BatchIndex) -> RecordContainer[T]: ...

    def __getitem__(self, item: ScalarIndex | BatchIndex) -> T | RecordContainer[T]:
        r"""Access records by scalar integer index, slice, or boolean/integer array mask.

        Args:
            item (ScalarIndex | BatchIndex): An integer index, slice, or mask/indices array.

        Returns:
            T | RecordContainer[T]: A single scalar record (`T`) if an integer index is provided,
                or a new batched collection if a slice or mask is provided.

        Raises:
            IndexError: If an integer index is out of bounds.
        """
        ...


class RaggedArrayContainer(RecordContainer[T], Protocol[T]):
    r"""A formal contract for ragged Structure-of-Arrays (SoA) containers.

    Ragged containers store variable-length data (such as sequences or CIGAR operations) in a flat,
    contiguous memory layout, managing sequence boundaries using `offsets` and `lengths` arrays.

    Attributes:
        offsets (npt.NDArray[np.int32]): Starting indices into the flat data array for each record.
        lengths (npt.NDArray[np.int32]): Number of elements in the flat array for each record.
    """

    @property
    def offsets(self) -> npt.NDArray[np.int32]:
        r"""Starting indices into the flat data array for each record."""
        ...

    @property
    def lengths(self) -> npt.NDArray[np.int32]:
        r"""Number of elements in the flat array for each record."""
        ...
