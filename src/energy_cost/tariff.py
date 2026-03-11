import bisect
import datetime as dt
from collections import defaultdict

import narwhals as nw

from .price_component import ComponentType, PriceComponent


class Tariff:
    supplier: str
    product_name: str
    components: dict[ComponentType, list[PriceComponent]] = defaultdict(list)

    def add_component(self, component: PriceComponent):
        """Add a price component to the tariff."""
        items = self.components[component.type]
        index = bisect.bisect_left(items, component.start, key=lambda c: c.start)
        items.insert(index, component)

    def get_components(
        self,
        component_type: ComponentType,
        start: dt.datetime,
        end: dt.datetime,
    ) -> list[PriceComponent]:
        """Get the price components of the given type that are active during the given time range."""
        components = self.components[component_type]
        start_index = bisect.bisect_right(components, start, key=lambda c: c.start) - 1
        end_index = bisect.bisect_right(components, end, key=lambda c: c.start)
        return components[start_index:end_index]

    def get_cost(
        self, component_type: ComponentType, start: dt.datetime, end: dt.datetime, resolution: dt.timedelta
    ) -> nw.DataFrame:
        """Get the cost values for the given component type and time range."""
        components = self.get_components(component_type, start, end)
        if not components:
            raise ValueError(f"No components of type {component_type} found in tariff.")

        ends = [component.start for component in components[1:]] + [end]
        df = components[0].get_values(start, ends[0], resolution)

        for component, end_time in zip(components[1:], ends[1:], strict=True):
            component_values = component.get_values(component.start, end_time, resolution)
            df = nw.concat([df, component_values], how="vertical")

        return df
