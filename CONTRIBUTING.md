# Contributing

Thanks for your interest in contributing! This is a small project, so the process is intentionally lightweight.

## Getting started

Before starting work on something non-trivial, please open an issue first or comment on an existing one.
This avoids duplicate effort and lets us discuss the best approach. Questions are always welcome.

See the [Development](README.md#development) section in the README for how to set up the project locally.

## How we work with Git

- Work in a **feature branch** off `main`, named something descriptive like `fix/capacity-calculation` or `data/brussels-region`.
- Open a **pull request** when your work is ready (or as a draft if you want early feedback).
- Every PR must have:
  - **All CI checks passing** (tests, linting, type checks — see [GitHub Actions](https://github.com/EnergieID/energy-cost/actions))
  - **One approval** from a maintainer before merging
- PRs do not need to be large — small, focused changes are easier to review and merge faster.

## Areas where help is welcome

### Features and bugs

See the [open issues](https://github.com/EnergieID/energy-cost/issues) on GitHub.
Feel free to comment on one to claim it, or open a new issue for anything you'd like to work on.

### Data maintenance

This is the area where contributions are most needed.

The library includes **built-in regional data** (distributor tariffs, government fees, and taxes) that power the automatic contract resolution when you specify a `region` in your contract. This data needs to be kept up to date annually, and more regions need to be added.

Currently supported regions can be found in the [data directory](src/energy_cost/data/).

If your region is not listed, contributions are very welcome.

#### How to add a new region

The Flemish data (`src/energy_cost/data/be/flanders/`) is the reference implementation. Here is how it is structured:

```
src/energy_cost/data/be/flanders/
├── __init__.py                  # registry: merges the electricity and gas registries
├── electricity/
│   ├── __init__.py              # Builds RegionalData as `data`; registry = {("be_flanders", ELECTRICITY): data}
│   ├── parse_distributors.py   # Script to parse distributor tariffs from source files
│   ├── distributors/           # One YAML file per distributor (e.g. fluvius_imewo.yml)
│   └── fees/                   # Government fees per customer type
└── gas/
    ├── __init__.py
    ├── parse_distributors.py
    ├── distributors/
    └── fees/
```

Every level exports a `registry` dict that maps `(region, connection type)` to `RegionalData`:

- `electricity/__init__.py` builds its `RegionalData` as `data` and exports `registry = {("be_flanders", ConnectionType.ELECTRICITY): data}`.
- `flanders/__init__.py` merges the electricity and gas registries.
- `be/__init__.py` merges the registries of the Belgian regions, and `src/energy_cost/data/__init__.py` merges those of all countries.

The default catalog is filled from `energy_cost.data.registry`, so contracts can refer to any bundled region without setup. Users can add just the regions they need to their own `Catalog`, e.g. `Catalog(regions=flanders.registry)`.

**Steps to add a new region (e.g. `be_brussels`):**

1. Create the folder structure under `src/energy_cost/data/` following the Flanders example.
2. Add **YAML data files** for distributor tariffs and government fees. See the [example tariff files](examples/tariffs/) for the YAML format.
3. Add an **`__init__.py`** per connection type that builds a `RegionalData` object and exports it as `registry` (e.g. `registry = {("be_brussels", ConnectionType.ELECTRICITY): data}`), and one for the region that merges them.
4. Add a **`README.md`** in your region folder documenting:
   - Where the data comes from (URLs, document names)
   - Any choices or assumptions made (e.g. which connection types or meter types are supported)
   - Notes that will help the next person update the data
5. Optionally add a **`parse_distributors.py`** script if the source data comes in a machine-readable format. This makes future yearly updates much easier.
6. Merge your region's `registry` into the parent `registry` (e.g. in `src/energy_cost/data/be/__init__.py`).

If you are unsure about the data format or regional structure, open an issue and we can help you get started.
