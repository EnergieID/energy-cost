# Energy cost calculation



## Development

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (fast Python package manager)

### 1. Install dependencies

```bash
uv sync
```

### 2. Notebooks

You can easilly test the project using our [marimo](marimo.io) notebooks. Just open the `notebooks` folder and run the notebook of your choice.

```bash
marimo run notebooks/quick_start.py
```

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
