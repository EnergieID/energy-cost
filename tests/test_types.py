from __future__ import annotations

import pytest

from energy_cost.types import _validate_tzinfo


class TestValidateTzinfo:
    def test_invalid_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected timezone string or tzinfo"):
            _validate_tzinfo(42)
