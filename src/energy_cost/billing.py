"""
An energy billing is defined based on 4 parts:
- the provider's tariff
- the distributor's tariff
- government fees (also a tariff, but taxes don't apply to this tariff as they are assumed to be included in the fee)
- taxes

To then calculate the bill for a given time range, we'll need:
The users injection and consumption data for the given time range
We need to know the meter type (single rate, multi rate, etc.)

For flemish capacity cost we need one year of data before the start of the billing period
So we should support the dataframe extending beyond the requested start and end timestamps, but only use the relevant part for the cost calculation.

The response should be a dataframe with the following grouped columns:
timestamp | <provider_name>                                                              | <distributor_name>                                                           | fees                                                                         | taxes       | total_cost
timestamp | consumption cost   | injection cost     | capacity cost | fixed cost | total | consumption cost   | injection cost     | capacity cost | fixed cost | total | consumption fees   | injection fees     | capacity fees | fixed fees | total | total taxes | total cost
timestamp | energy | renewable | energy | renewable | capacity cost | fixed cost | total | energy | renewable | energy | renewable | capacity cost | fixed cost | total | energy | renewable | energy | renewable | capacity cost | fixed cost | total | total taxes | total cost

A tariff should alread have a method that given injection and consumption data, and the meter type, a start and end dat returns a dataframe like this:
timestamp | consumption cost   | injection cost     | capacity cost | fixed cost | total
timestamp | energy | renewable | energy | renewable | capacity cost | fixed cost | total

Which can then be reused 3 times for the provider, distributor and fees. The taxes can then be calculated based on the total cost of the provider and distributor. (taxes ignore fees as they are assumed to already include taxes)

"""
