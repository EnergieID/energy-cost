import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from os import environ

    from energy_cost.index import EntsoeDayAheadIndex, Index

    Index.register("Belpex15min", EntsoeDayAheadIndex(country_code="BE", api_key=environ["ENTSOE_API_KEY"]))
    return


@app.cell
def _():
    from energy_cost.tariff import Tariff

    tariff = Tariff.from_yaml("data/tariffs/EBEM/Groen_Dynamic.yml")
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


if __name__ == "__main__":
    app.run()
