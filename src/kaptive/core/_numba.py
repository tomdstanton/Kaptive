r"""Typed Numba JIT decorator and parallel loop re-exports.

Provides static type signatures for `njit` and `prange` under `TYPE_CHECKING`
while re-exporting the real Numba symbols untouched at runtime.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar, overload

_F = TypeVar("_F", bound=Callable[..., object])

if TYPE_CHECKING:
    prange = range

    @overload
    def njit(func: _F, /) -> _F: ...

    @overload
    def njit(
        *,
        parallel: bool = False,
        cache: bool = False,
        nogil: bool = False,
        inline: str = "never",
    ) -> Callable[[_F], _F]: ...

    def njit(
        func: _F | None = None,
        /,
        *,
        parallel: bool = False,
        cache: bool = False,
        nogil: bool = False,
        inline: str = "never",
    ) -> _F | Callable[[_F], _F]:
        r"""Compile a function in nopython mode via Numba.

        Args:
            func: Optional target function when used as bare `@njit`.
            parallel: Enable automatic parallelization (`prange`).
            cache: Cache compiled machine code on disk.
            nogil: Release the Python GIL during kernel execution.
            inline: Inlining behavior (`"never"` or `"always"`).

        Returns:
            Compiled function or decorator preserving the original callable signature.
        """
        if func is not None:
            return func
        return lambda f: f
else:
    from numba import njit, prange

__all__ = ["njit", "prange"]
