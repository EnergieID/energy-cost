from typing import Annotated, cast

from pydantic import Discriminator, Tag

from .formula import FormulaBase
from .index import IndexAdder, IndexFormula
from .periodic import PeriodicFormula
from .scheduled import DayOfWeek, ScheduledFormula, ScheduledFormulas, WhenClause
from .tiered import TierBand, TieredFormula, TieringMode


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

# Rebuild models whose `formula: Formula` field is a forward reference,
# now that the Formula union is fully defined.
TierBand.model_rebuild(_types_namespace={"Formula": Formula})
TieredFormula.model_rebuild(_types_namespace={"Formula": Formula})
ScheduledFormula.model_rebuild(_types_namespace={"Formula": Formula})
ScheduledFormulas.model_rebuild(_types_namespace={"Formula": Formula})

__all__ = [
    "Formula",
    "FormulaBase",
    "IndexAdder",
    "IndexFormula",
    "PeriodicFormula",
    "DayOfWeek",
    "WhenClause",
    "ScheduledFormula",
    "ScheduledFormulas",
    "TierBand",
    "TieredFormula",
    "TieringMode",
]
