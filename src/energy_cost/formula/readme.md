# design notes

We have four distinct types of cost components:

1. consumption based cost components:
You pay for the energy consume, or get payed for the energy you produce, this is expressed in €/MWh
This price is typically defined by:
- a formula a*A + b*B + c*C + ... + z where A, B, C are indexes (e.g. day-ahead price, imbalance price, etc.) and a, b, c are coefficients, with z being a constant term.
- there can be different formulas for different time slots, e.g. one formula for weekdays and another for weekends, or even more granularly, e.g. one formula for each hour of the day.

The exact result of the formula is different at different timestamps, depending on the values of the indexes at those timestamps.
The consumption data needs at least the timestamps and the amount of energy consumed/produced at those timestamps, and the cost is calculated by multiplying the amount of energy with the price at those timestamps.

2. capacity based cost components:
This is a cost component based on the capacity used (e.g. the maximum power drawn or fed into the grid during a billing period), expressed in €/kW.
Just like consumption based cost components, the price can be defined by a formula:
- a formula a*A + b*B + c*C + ... + z where A, B, C are indexes (e.g. day-ahead price, imbalance price, etc.) and a, b, c are coefficients, with z being a constant term.
- there can be different formulas for different time slots, e.g. one formula for weekdays and another for weekends, or even more granularly, e.g. one formula for each hour of the day.

But in this case we also have some extra formula structures:
- we can have thresholds, e.g. the first 10 kW are charged with one formula, and the next 10 kW are charged with another formula, etc.
- sometimes this can even be fixed costs for certain capacity levels, e.g. below 10 kW it's a fixed cost of 100 €/month (so no €/MW)
- Flanders has a rolling window of 12 months, averaging the maximum power drawn/fed into the grid over the last 12 months, and charging based on that average.

Capacity prices make use of two 'period' concepts:
- the measurement period, which is the period over which the capacity is measured (e.g. 15 minutes, 1 hour, 1 day, etc.)
- the billing period, which is the period over which the cost is calculated (e.g. 1 month, 1 year, etc.)
The cost is calculated by applying the price formula to the maximum capacity used in any measurement period during the billing period.

3. periodic cost components:
This is a cost component that is fixed for a certain period, e.g. a fixed cost of 10 €/month, or a fixed cost of 100 €/year, etc.
In theory yopu could also have these costs defined by a formula, e.g. a formula that depends on the day-ahead price at the first day of the month, but in practice this is not common, so we can just have a fixed cost for a certain period.

4. taxes and fees:
This cost component actualluly consists of two sub-components:
- taxes, which are typically a percentage of the total cost (e.g. VAT), that is applied on top of the total cost calculated from the other cost components.
- fees, which can be calculated in any of the three above ways, with the only key difference that they don't have VAT applied on top of them, so they are not included in the total cost that is used to calculate the VAT.



# IO
Usecases:
- I want to calculate the total cost of my electricity consumption for a certain period, e.g. a month, so I can see how much I have to pay for my electricity bill.
    - inputs:
        - consumption data (timestamps and amount of energy consumed/produced at those timestamps)
        - start ISO timestamp (optional, if not provided, we can use the minimum timestamp from the consumption data)
        - end ISO timestamp (optional, if not provided, we can use the maximum timestamp from the consumption data)
        - resolution ISO duration (only suport P1M and P1Y for now, so we calculate the cost for each month or each year, if not provided we only calulate the total cost for the whole period)
    - outputs:
        - a detailed breakdown of the cost, showing the contribution of each cost component, and the total cost.
          this includes the cost from consumption based cost components, capacity based cost components, periodic cost components, and taxes and fees.

A user should be able construct this himeself using the following, while providing way less data:
- I want a graph displaying the price of electricity for each hour of the day, for a certain period, e.g. a month, so I can see when it's cheapest to consume or produce electricity.
    - inputs:
        - start ISO timestamp
        - end ISO timestamp
        - resolution ISO duration (e.g. PT1H for hourly, P1D for daily, etc.)
    - outputs:
        - pandas DataFrame with timestamps as index and price as values, with the specified resolution (e.g. hourly, daily, etc.), each cost component can have its price as a separate column, and there can be an additional column for the total price (sum of all cost components).
        A tiered pricing can be represented by having multiple columns for the same cost component, e.g. 'capacity_cost_tier_1', 'capacity_cost_tier_2', etc.
        This should only include consumption/injection based cost components expressed in €/MWh.
 - I want to calculate fixed costs for a given period
    - inputs:
        - start ISO timestamp
        - end ISO timestamp
    - outputs:
        - a detailed breakdown of the fixed costs, showing the contribution of each cost component, and the total fixed cost.
- I want to calculate capacity costs for a given period, based on my maximum power drawn/fed into the grid per billing period, so I can see how much I have to pay for the capacity I use.
    - inputs:
        - start ISO timestamp
        - end ISO timestamp
        - the maximum power drawn/fed into the grid per billing period (e.g. per month, per year, etc.), provided as a pandas DataFrame with timestamps column and maximum power as values, with the specified resolution (e.g. monthly, yearly, etc.)
    - outputs:
        - a pandas DataFrame with timestamps as index and capacity cost as values, with the specified resolution (e.g. monthly, yearly, etc.), each cost component can have its cost as a separate column, and there can be an additional column for the total capacity cost (sum of all capacity cost components).


# Formula structure

Formula = Scheduled | Tiered | Periodic | Index

Index:
    constant_cost: float
    variable_costs: list[VariableCost]

Periodic:
    cost: float
    period: Resolution (e.g. monthly, yearly, etc.)

Scheduled:
    schedule: list[ScheduleEntry]

ScheduleEntry:
    mask: Time based mask (e.g. "weekdays", "weekends", "hours 0-6", etc.)
    formula: Formula

Tiered:
    bands: list[TierBand]

TierBand:
    mask: Value based mask (e.g. "up to 10 kW", "up to 20 kW", etc.)
    formula: Formula