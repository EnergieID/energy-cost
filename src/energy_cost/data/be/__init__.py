from pathlib import Path

from energy_cost.tariff import Tariff

_DISTRIBUTORS_DIR = Path(__file__).parent / "distributors"

distributors: dict[str, Tariff] = {
    path.stem: Tariff.from_yaml(path) for path in sorted(_DISTRIBUTORS_DIR.glob("*.yml"))
}

_FEES_DIR = Path(__file__).parent / "fees"

fees: dict[str, Tariff] = {path.stem: Tariff.from_yaml(path) for path in sorted(_FEES_DIR.glob("*.yml"))}

tax_rate = 0.06
