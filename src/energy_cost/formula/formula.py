from typing import Annotated, cast

from pydantic import Discriminator, Tag

from .index import IndexFormula
from .periodic import PeriodicFormula
from .scheduled import ScheduledFormulas
from .tiered import TieredFormula


def _formula_discriminator(v: object) -> str | None:
    """Return the ``kind`` tag used for union routing.

    Explicit ``kind`` fields take priority; otherwise heuristics based on the
    presence of characteristic keys are used (for YAML that omits ``kind``).
    Model instances are routed via their ``.kind`` attribute.
    """
    if isinstance(v, dict):
        d = cast(dict[str, object], v)
        if "kind" in d:
            return str(d["kind"])
        if "bands" in d:
            return "tiered"
        if "period" in d:
            return "periodic"
        if "schedule" in d:
            return "scheduled"
        if "constant_cost" in d or "variable_costs" in d:
            return "index"
        return None
    return getattr(v, "kind", None)


# Discriminated union of all concrete formula types.
# A callable Discriminator handles both explicit ``kind`` fields and
# heuristic routing for YAML that omits ``kind``.
Formula = Annotated[
    Annotated[IndexFormula, Tag("index")]
    | Annotated[PeriodicFormula, Tag("periodic")]
    | Annotated[ScheduledFormulas, Tag("scheduled")]
    | Annotated[TieredFormula, Tag("tiered")],
    Discriminator(_formula_discriminator),
]
