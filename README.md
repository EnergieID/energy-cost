# Energy cost

Python package to model your energy bill based on your energy consumption.

To use this library, you first specify the `Tariff` applicable to you in a yaml file. See the `examples/tariffs` directory for inspiration, with `notebooks/tariffs.ipynb` for a detailed walkthrough.

A lot of tariffs are based on an `Index`, which is a price that changes over time based on the market price of energy. We have built in support for fetching these prices from ENTSOE or defining them in a data file, see `notebooks/index.ipynb` for more info.

Then, you can use the `Contract` class to calculate your costs based on your consumption data. See `notebooks/contract.ipynb` for an example.

We already have some built in tariffs for the Belgian market, which you can find in `src/energy_cost/data/be/`. These are the distributor tariffs for the main Belgian distributors, as well as the government fees and taxes. We plan to expand this to other European countries in the future, feel free to contribute if you want to see your country's tariffs in the library!

> Note on units: all consumption based costs are in €/MWh, all energy values are in MWh. All monetary values are in €.

> Note on timezones: every method that takes timestamps as input also takes a timezone as input. All timestamps will be aligned to this timezone, and all outputs will be in this timezone as well. This means that you can use the library with any timezone, regardless of the timezone of the input data or the tariff definitions. By default, the timezone is set to UTC.

## Example usage
First define the tariff from your distributor in a yaml file, for example:

```yaml
- start: 2024-01-01T00:00:00+01:00
  consumption:
    constant_cost: 100.0
  injection:
    constant_cost: -20.0
```

Then, you can use the `Contract` class to calculate your costs based on your consumption data:

```python
from energy_cost import Contract, Meter, Tariff
from energy_cost.data.be import distributors, fees, tax_rate

contract = Contract(
    supplier=Tariff.from_yaml("../examples/tariffs/fixed.yml"),
    distributor=distributors["fluvius_imewo"],
    fees=[fees["flanders_residential"], fees["be_residential"]],
    tax_rate=tax_rate,
    timezone=ZoneInfo("Europe/Brussels"),
)

consumption = Meter(
    data=pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T00:00:00+01:00", "2024-03-01T00:00:00+01:00", freq="15min"),
            "value": 0.0002,
        }
    )
)

contract.calculate_cost([consumption])
```

For more detailed examples, see the notebooks in the `notebooks` directory.

## Development

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (fast Python package manager)

### 1. Install dependencies
```bash
uv sync
uv tool install poethepoet
```

### 2. Notebooks
You can easilly test the project using our notebooks. They are located in the `notebooks` directory and can be run with:

```bash
poe notebooks
```

Some notebooks make use of external APIs and require API keys. You can set these in a `.env` file in the root of the project, using the `.env.example` file as a template.

If you add new features, please add a new notebook to demonstrate them.

### 3. Run tests
```bash
poe test
```

### 4. Code quality (pre-commit)
Pre-commit hooks are configured out of the box. Install them once:

```bash
pre-commit install
```

Every commit will automatically run:

| Tool | Task | Command |
|------|------|---------|
| **Ruff** | Linting + auto-fix | `poe lint` |
| **Ruff** | Formatting | `poe format` |
| **ty** | Type checking | `poe check` |

You can also run them manually at any time.

## License

[MIT](LICENSE)
