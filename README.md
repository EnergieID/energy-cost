# Energy cost

Python package to model your energy bill based on your energy consumption.

This energy bill is calculated based on a `Contract` or even a `ContractHistory`, which is a collection of `Contract`s that can change over time.
A contract consists of 5 parts:
- `supplier`: the `Tariff` of your energy supplier
- `distributor`: the `Tariff` of your energy distributor
- `fees`: the government fees applicable to you, als defined as a `Tariff`
- `taxes`: the government `Taxes` applicable to each cost component of your bill
- `timezone`: the timezone in which all timestamps will be aligned and all outputs will be in

You can also specify a `region`, `connection_type`, `customer_type` and `distributor_key` in your contract, which will automatically fetch the applicable distributor tariffs, fees, taxes and timezone for you from our built in data. This is optional, but it can save you a lot of time if your region is supported.

Contracts can be defined in a yaml file, which makes it easy to manage and update your contract over time. See [`notebooks/contract.ipynb`](notebooks/contract.ipynb) for more info on how to define your contract in a yaml file.

We also have more detailed documentation on the different components of a contract, see [`notebooks`](notebooks/) for all available notebooks.
You can find example yaml files for tariffs and taxes in the [`examples`](examples) directory.

A lot of tariffs are based on an `Index`, which is a price that changes over time based on the market price of energy. We have built in support for fetching these prices from ENTSOE or defining them in a data file, see [`notebooks/index.ipynb`](notebooks/index.ipynb) for more info.


> Note on units: all consumption based costs are in €/MWh, all energy values are in MWh. All monetary values are in €.

> Note on timezones: every method that takes timestamps as input also takes a timezone as input. All timestamps will be aligned to this timezone, and all outputs will be in this timezone as well. By default, the timezone is set by the regional data, but you can override it in your contract if needed.

## Example usage
You can define your contract in a yaml file like this:

```yaml
supplier:
- start: 2025-01-01T00:00:00+01:00
  consumption:
    constant_cost: 90.0
region: be_flanders
connection_type: electricity
customer_type: residential
distributor_key: fluvius_imewo
```

Then, you can use the `Contract` class to calculate your costs based on your consumption data:

```python
from energy_cost import Contract, Meter
import pandas as pd

Contract.from_yaml("../examples/contracts/inline.yml")

consumption = Meter(
    data=pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01T00:00:00+01:00", "2024-03-01T00:00:00+01:00", freq="15min"),
            "value": 0.0002,
        }
    )
)

contract.apply([consumption])
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
