"""Inert legacy MNE epoch reader used by current MEG phase materialization."""
import pickle
from typing import Any
from collections.abc import Mapping
import numpy as np

class _PicklePlaceholder:
    """Non-executable stand-in for legacy MNE classes during smoke loading."""

    def __new__(cls, *args: Any, **kwargs: Any) -> "_PicklePlaceholder":
        instance = object.__new__(cls)
        instance._mapping = {}
        return instance

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, Mapping):
            self.__dict__.update(state)
        elif (
            isinstance(state, tuple)
            and len(state) == 2
            and isinstance(state[0], Mapping)
        ):
            self.__dict__.update(state[0])
            if isinstance(state[1], Mapping):
                self.__dict__.update(state[1])
        else:
            self.state = state

    def __setitem__(self, key: Any, value: Any) -> None:
        self._mapping[key] = value

    def __getitem__(self, key: Any) -> Any:
        return self._mapping[key]

    def get(self, key: Any, default: Any = None) -> Any:
        return self._mapping.get(key, default)

    def keys(self) -> Any:
        return self._mapping.keys()

class _EpochUnpickler(pickle.Unpickler):
    """Load only official MNE epoch containers and standard numeric values."""

    _SAFE_MODULES = frozenset(
        {
            "builtins",
            "collections",
            "copyreg",
            "datetime",
            "numpy",
            "numpy.core.multiarray",
            "numpy.core.numeric",
            "numpy.dtype",
        }
    )

    def find_class(self, module: str, name: str) -> Any:
        if module.startswith("mne.") or module.startswith("scipy."):
            # The release was serialized with an older MNE layout.  The smoke
            # only needs the serialized arrays/metadata, so no MNE/SciPy
            # constructor or method is executed.
            return _PicklePlaceholder
        if module not in self._SAFE_MODULES:
            raise pickle.UnpicklingError(f"pickle class is outside allowlist: {module}.{name}")
        return super().find_class(module, name)
