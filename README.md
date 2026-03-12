# Energy cost

Python package to model your energy bill based on your energy consumption.

This package is currently under construction, with many features still incomplete or missing.
See [issues](https://github.com/EnergieID/energy-cost/issues) for future planned work and feel free to contribute!

Currently implemented features are documented using notebooks in the `notebooks/` directory.

> Note on units: all consumption based costs are in €/MWh, all energy values are in MWh. All monetary values are in €.

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
