import datetime as dt
from abc import ABC, abstractmethod
from typing import ClassVar

import pandas as pd


class Index(ABC):
    """An index to use for calculating the energy cost."""

    indexes: ClassVar[dict[str, "Index"]] = {}

    @classmethod
    def register(cls, name: str, index: "Index") -> None:
        """Register an index instance by name."""
        cls.indexes[name] = index

    @abstractmethod
    def get_values(self, start: dt.datetime, end: dt.datetime, resolution: dt.timedelta) -> pd.DataFrame:
        """Get the index values for the given time range and resolution."""

    @staticmethod
    def from_name(name: str) -> "Index":
        """Get the index instance for the given name."""
        if name not in Index.indexes:
            raise ValueError(f"Unsupported index: {name}")
        return Index.indexes[name]
