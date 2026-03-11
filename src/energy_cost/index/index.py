import datetime as dt
from abc import ABC, abstractmethod

import narwhals as nw


class Index(ABC):
    """An index to use for calculating the energy cost."""

    name: str
    indexes: dict[str, "Index"] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "name"):
            cls.indexes[cls.name] = cls()

    @abstractmethod
    def get_values(self, start: dt.datetime, end: dt.datetime, resolution: dt.timedelta) -> nw.DataFrame:
        """Get the index values for the given time range and resolution."""

    @staticmethod
    def from_name(name: str) -> "Index":
        """Get the index instance for the given name."""
        if name not in Index.indexes:
            raise ValueError(f"Unsupported index: {name}")
        return Index.indexes[name]
