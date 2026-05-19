from .base import FormulaBase
from .formula import Formula
from .index import IndexAdder, IndexFormula
from .minmax import MaximumFormula, MinimumFormula
from .periodic import PeriodicFormula
from .scheduled import DayOfWeek, ScheduledFormula, ScheduledFormulas, WhenClause
from .tiered import TierBand, TieredFormula, TieringMode

# Rebuild models whose `formula: Formula` field is a forward reference,
# now that the Formula union is fully defined.
TierBand.model_rebuild(_types_namespace={"Formula": Formula})
TieredFormula.model_rebuild(_types_namespace={"Formula": Formula})
ScheduledFormula.model_rebuild(_types_namespace={"Formula": Formula})
ScheduledFormulas.model_rebuild(_types_namespace={"Formula": Formula})
MinimumFormula.model_rebuild(_types_namespace={"Formula": Formula})
MaximumFormula.model_rebuild(_types_namespace={"Formula": Formula})

__all__ = [
    "Formula",
    "FormulaBase",
    "IndexAdder",
    "IndexFormula",
    "MinimumFormula",
    "MaximumFormula",
    "PeriodicFormula",
    "DayOfWeek",
    "WhenClause",
    "ScheduledFormula",
    "ScheduledFormulas",
    "TierBand",
    "TieredFormula",
    "TieringMode",
]
