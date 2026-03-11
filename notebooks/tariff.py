import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from energy_cost.tariff import Tariff

    tariff = Tariff.from_yaml("src/energy_cost/tariffs/EBEM_Groen_Dynamic.yml")
    tariff
    return (tariff,)


@app.cell
def _(tariff):
    import datetime as dt

    from energy_cost.price_component import ComponentType

    start = dt.datetime.fromisoformat("2026-03-08 00:00:00+01:00")
    end = dt.datetime.fromisoformat("2026-03-10 00:00:00+01:00")
    resolution = dt.timedelta(minutes=15)
    tariff.get_cost(ComponentType.INJECTION, start, end, resolution)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
