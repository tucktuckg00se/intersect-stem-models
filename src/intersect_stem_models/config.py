from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class SafeLoaderWithTuple(yaml.SafeLoader):
    """Treat !!python/tuple as a list and normalize it after load."""


def _tuple_constructor(loader: yaml.Loader, node: yaml.Node) -> list[Any]:
    return loader.construct_sequence(node)


SafeLoaderWithTuple.add_constructor("tag:yaml.org,2002:python/tuple", _tuple_constructor)


@dataclass(frozen=True)
class AttrDict:
    """Recursive attribute access wrapper around nested dictionaries."""

    _data: dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return self._data


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return AttrDict({key: _normalize(inner) for key, inner in value.items()})
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def load_yaml_config(path: str | Path) -> AttrDict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.load(handle, Loader=SafeLoaderWithTuple)
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML object at {path}, got {type(data)!r}")
    return _normalize(data)

