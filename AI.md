  

## 1) Human-readable overview

### Goal

Build an **Open Source Python library** that calculates **end-consumer energy costs** (electricity + gas) for **Flanders (BE-VLG)** first, while being designed to expand to other regions later.

### What “cost” includes

For each time step in the energy time series, compute an itemized cost breakdown including:

- **Energy supply component** (contract-dependent: fixed/variable/dynamic)

- **Network costs**: distribution + transmission/system charges + data management fees (region/DSO-dependent; not supplier-specific)

- **Taxes/levies**: e.g. special excise tiers, Flemish energy fund, public service obligations (region- and customer-type dependent)

- **Capacity costs (electricity)**: monthly peak-based; spread over time steps within each month

- **VAT handling**: not all components are taxable; VAT rate depends on customer type and rules

### Supported scope (v1)

- **Electricity**

- **Offtake** and **Injection** (calculated separately in v1)

- **Contract types**

- **Dynamic**: requires **15-min (quarter-hourly)** consumption/injection + **15-min day-ahead price series**

- **Fixed / Variable**: supports **single-register** and **dual-register (day/night)** meters and pricing

- Day/night support applies only to fixed/variable electricity

- **Gas**

- **Offtake**, primarily **monthly volumes** (v1), but allow higher resolution inputs and aggregate as needed

### Inputs (public API)

1. **Energy time series**

- Electricity fixed/variable:

- single register: one channel `kwh`

- dual register: two channels `kwh_day`, `kwh_night`

- Electricity dynamic:

- 15-min series, channel `kwh` (day/night not applicable)

- Gas:

- monthly series, channel `kwh` (preferred; caller may supply in m³ but library should normalize to kWh if needed)

2. **Connection / customer metadata**

- Residential vs professional

- Region & DSO (e.g., Fluvius Antwerpen)

- Meter register: single vs dual (day/night)

- Meter type: analog vs digital

- Voltage level (LV/MV)

- Gas tariff class (T1..T6), telemetering, etc.

- Optional flags like “separate injection access point” (affects some grid fees)

3. **Tariff definition (supplier-agnostic)**

A generic tariff product definition describing:

- contract type (dynamic/variable/fixed)

- required register type (single/dual)

- time-of-use mapping (single vs day/night) where applicable

- energy pricing formula(s) and index references

- surcharges (e.g. green/WKK), fixed fees

- injection remuneration rules

- validity window

4. **Regulated datasets (internal to library)**

- Distribution tariffs and other regulated charges for the region/DSO

- Taxes/levies where appropriate (depending on chosen “grid vs tax” split)

- Data stored in a **normalized internal dataset format** (JSON/Parquet) shipped with the library

- Provide extraction scripts to regenerate datasets from published Excel files

5. **Market prices / indices**

- Dynamic electricity: **15-min** day-ahead price series (Belpex/Epex Spot 15’)

- Fixed/variable (gas & electricity): v1 assumes caller provides **monthly energy cost series** (no index-weighting with synthetic profiles in v1)

6. **Monthly capacity peak (electricity)**

- Provided by caller OR computed via a pluggable helper (caller may use OpenEnergyID algorithm externally)

- Capacity cost must be spread across all timesteps in that month (including quarter-hours for dynamic)

### Outputs

- A time series of total cost OR

- A dataframe with itemized components

- Some columns are variable €/kWh (applied per timestep)

- Some are fixed (€/year, €/month) prorated over timesteps

- Capacity (€/kW/month) allocated over month timesteps

- Taxes and VAT applied with correct applicability flags

- Standardized column naming and deterministic totals (sum of itemized parts)

### Expansion (v2+)

- Ability to calculate electricity offtake and injection together (dual input, separate tariff definitions, combined reporting/netting logic where legal/required)

## 2) Architecture (supplier-agnostic)

### Key principles

- **Supplier-agnostic core**: no supplier-specific modules/folders in library code

- **Tariff products are data**: definitions expressed as JSON/YAML (or loaded from external configs)

- **Region plug-ins**: datasets + rules per region/DSO

- **Deterministic calculation graph**: compute components in a stable order, output itemized breakdown with standardized names

- **Normalized datasets**: regulator/DSO excel -> normalized schema -> packaged data files

### Suggested package layout

```text

energy_costs/

__init__.py

core/

models.py # Pydantic objects

time.py # billing periods, proration, resampling

engine.py # orchestration

breakdown.py # standardized columns, aggregation helpers

validation.py

tariffs/

definitions.py # generic tariff schema & loaders (YAML/JSON)

formulas.py # formula evaluation: fixed, indexed-linear, pass-through

tou.py # time-of-use mapping and channel handling

regions/

be_flanders/

datasets/

fluvius_antwerpen_2026_elec_offtake.json

fluvius_antwerpen_2026_elec_injection.json

fluvius_antwerpen_2026_gas_offtake.json

fluvius_antwerpen_2026_gas_injection.json

grid.py # select applicable grid tariff set based on ConnectionInfo

taxes.py # Flemish taxes & VAT rules

normalization.py # normalized schema definitions

io/

parsers/

dso_excel.py # DSO/regulator excel extraction into normalized format

tariffcard_pdf.py # optional/best-effort: parse tariff cards into generic definitions

cli/

extract_dso_tariffs.py # regenerate normalized datasets

examples/

be_flanders_2026/

tariff_definitions.yaml

run_example.py

tests/

fixtures/

test_engine.py

```

### Engine flow

1. Validate inputs:

- Ensure required resolution per contract type

- Ensure required channels (kwh_day/kwh_night) if TOU pricing is used

1. Build TariffContext:

- connection metadata + billing period + region rules (VAT, tax applicability)

1. Resolve:

- grid tariff set from region datasets using ConnectionInfo

- taxes rules for period + customer type

1. Compute itemized components:

- supplier energy (per channel if day/night)

- supplier surcharges (per kWh)

- supplier fixed fees (prorated)

- grid variable (per kWh, typically on sum of channels)

- grid fixed (prorated)

- capacity charges (monthly kW peak -> €/month -> allocate over timesteps)

- taxes (tiered, fixed, variable)

- VAT (only on VAT-applicable components)

1. Return CostResult:

- breakdown dataframe + totals + assumptions

## 3) Core data objects (Pydantic)

### 3.1 Energy input (multi-channel)

EnergySeries

- carrier: Literal["electricity","gas"]

- direction: Literal["offtake","injection"]

- timezone: str = "Europe/Brussels"

- unit: Literal["kWh"] # recommend normalizing gas to kWh internally

- data: pandas.DataFrame

- index:

- electricity dynamic: DatetimeIndex at 15-min

- electricity fixed/variable: any DatetimeIndex; can be 15-min/hour/day, aggregated as needed

- gas v1: monthly (PeriodIndex or month-start timestamps)

- columns:

- electricity fixed/variable single: ["kwh"]

- electricity fixed/variable dual day/night: ["kwh_day","kwh_night"]

- electricity dynamic: ["kwh"]

- gas: ["kwh"]

- validation:

- monotonic index, no duplicates

- dynamic: enforce 15-min frequency

- fixed/variable with day/night tariff: require kwh_day and kwh_night

- non-negative by default (allow negatives only explicitly)

### 3.2 Market prices

MarketPriceSeries

- market: Literal["EPEX_DA_BE_15MIN","CUSTOM_MONTHLY","CUSTOM_OTHER"]

- unit: Literal["EUR/MWh","EUR/kWh"]

- data: pd.Series (DatetimeIndex)

- helper: to_eur_per_kwh()

### 3.3 Connection/customer metadata

ConnectionInfo

- region: Literal["BE-VLG"]

- dso: str # e.g. "Fluvius Antwerpen"

- customer_type: Literal["residential","professional"]

- vat_rate: Optional[float] # if None, infer from customer_type + rules

- electricity:

- meter_register: Literal["single","dual_day_night"]

- meter_type: Literal["digital","analog"]

- voltage_level: Literal["LV","MV_1_26kV","MV_26_36kV"]

- has_separate_injection_point: bool = False

- gas:

- gas_tariff_class: Optional[Literal["T1","T2","T3","T4","T5","T6"]]

- telemetered: Optional[bool]

### 3.4 Supplier-agnostic tariff definition (product config)

TariffDefinition

- name: str

- carrier: Literal["electricity","gas"]

- direction: Literal["offtake","injection"]

- contract_type: Literal["dynamic","variable","fixed"]

- meter_register_required: Optional[Literal["single","dual_day_night"]]

- time_of_use: Optional[Literal["single","day_night"]]

- components: list[TariffComponent]

- valid_from: date

- valid_to: date

- notes: dict (optional)

TariffComponent

- component_type: Literal[

"energy",

"surcharge_green","surcharge_wkk",

"fixed_fee",

"injection_remuneration",

"other"

]

- channel: Optional[Literal["kwh","kwh_day","kwh_night"]]

- Use channel to support day/night pricing (two energy components)

- pricing: PricingRule

- vat_applicable: bool

PricingRule

- kind: Literal["fixed_per_kwh","indexed_linear","fixed_per_year","fixed_per_month"]

- if kind == "indexed_linear":

- index_name: str # e.g. BelpexRLP0, Belpex15’, ZTPRLP0

- coef: float

- add_cents_per_kwh: float

- if kind == "fixed_per_kwh":

- eur_per_kwh: float

- if kind == "fixed_per_year":

- eur_per_year: float

- if kind == "fixed_per_month":

- eur_per_month: float

### 3.5 Region datasets (internal normalized)

GridTariffComponent

- component_type: Literal[

"distribution_variable","distribution_fixed",

"databeheer","public_service","system_management",

"capacity_tariff",

"other"

]

- unit: Literal["EUR/kWh","EUR/year","EUR/kW/month","EUR/kW/year","EUR/month"]

- value: float

- conditions: dict # meter_type, voltage_level, gas_class, etc.

- vat_applicable: bool (default False, set by region rules if needed)

GridTariffSet

- region: str

- dso: str

- carrier: str

- direction: str

- valid_from: date

- valid_to: date

- components: list[GridTariffComponent]

TaxRule

- name: str

- carrier/direction/customer_type applicability

- kind: Literal["variable","fixed_monthly","fixed_yearly","tiered"]

- unit: EUR/kWh or EUR/month or EUR/year

- tiers: optional list[{from_kwh,to_kwh,value_eur_per_kwh}]

- vat_applicable: bool

### 3.6 Output

CostResult

- breakdown: pd.DataFrame (same index as energy input after alignment)

- totals: dict[str,float] # sums per column

- assumptions: dict # VAT rate used, proration method, conversions, etc.

  

Column naming (standard)

- supplier_energy_eur

- if day/night: supplier_energy_day_eur, supplier_energy_night_eur

- supplier_surcharges_eur

- supplier_fixed_eur

- grid_variable_eur

- grid_fixed_eur

- capacity_eur

- taxes_eur (or itemized: tax_excise_eur, tax_energy_fund_eur, ...)

- vat_eur

- total_eur

## 4) Scraped/extracted example data (from provided attachments)

NOTE: This library remains supplier-agnostic; these are example tariff definitions and extracted regulated grid tariff values for Flanders (Fluvius Antwerpen 2026).

### 4.1 Example supplier tariff formulas (EBEM tariff cards, March 2026)

Electricity — Variable (example)

- Offtake, single register:

- c€/kWh = 0.110 * BelpexRLP0 + 2.2

- Offtake, dual register:

- Day (kwh_day): c€/kWh = 0.120 * BelpexRLP0 + 2.2

- Night (kwh_night): c€/kWh = 0.099 * BelpexRLP0 + 2.2

- Injection:

- c€/kWh = 0.0925 * BelpexSPP0 - 1.25

- Fixed fee:

- 84.91 EUR/year

- Exclusive night fixed fee:

- 33.06 EUR/year

Electricity — Fixed/Variable product with surcharges (example)

- Offtake:

- c€/kWh = 0.110 * BelpexRLP0 + 2.0

- Fixed fee:

- 70.75 EUR/year

- Surcharges (excl VAT):

- green: 1.100 c€/kWh

- WKK: 0.420 c€/kWh

Electricity — Dynamic (example)

- Offtake (15-min):

- c€/kWh = 0.108 * Belpex15’ + 1.625

- Injection (15-min):

- c€/kWh = 0.0925 * Belpex15’ - 1.10

- Fixed fee:

- 70.75 EUR/year

Gas — Variable (example)

- c€/kWh = 0.105 * ZTPRLP0 + 0.675 (one product)

- c€/kWh = 0.105 * ZTPRLP0 + 0.625 (other product)

- Fixed fee:

- 70.75 EUR/year or 61.32 EUR/year

### 4.2 Example taxes/levies shown on tariff cards (extract to TaxRule)

Electricity (household excise tiers, values as printed)

- 0–3 MWh: 5.0329 c€/kWh

- 3–20 MWh: 5.0329 c€/kWh

- 20–50 MWh: 4.8188 c€/kWh

- 50–1000 MWh: 4.7569 c€/kWh

Flanders energy fund (examples)

- Residential LV: fixed 0 EUR/month; variable 0.20417 c€/kWh

- Non-residential LV: fixed 10.07 EUR/month; variable 0.19261 c€/kWh

- MV: fixed 192.11 EUR/month

Gas excise tiers (examples)

- 0–12 MWh: 0.87238 c€/kWh

- 12–20,000 MWh: 0.96229 c€/kWh

Implementation note: Internally store everything preferably excl VAT and apply VAT based on vat_applicable flags and ConnectionInfo.vat_rate. If a source provides incl VAT values, either store with a flag or convert back.

### 4.3 Fluvius Antwerpen 2026 — regulated grid tariffs (from provided Excel)

These are the key values to normalize into GridTariffSet records. Ingest ALL relevant rows/columns via parser scripts; below are representative extracted values.

Electricity offtake — LV digital (capacity tariff)

- capacity: 4.1169714 EUR/kW/month (49.4036563 EUR/kW/year)

- variable netgebruik: 0.0234492 EUR/kWh

- databeheer LV: 17.85 EUR/year

- ODV kWh-tarief normaal (LV): 0.0254845 EUR/kWh

Electricity offtake — LV analog (retrograde)

- fixed term: 123.51 EUR/year

- variable netgebruik: 0.0492427 EUR/kWh

- databeheer LV: 17.85 EUR/year

- ODV kWh-tarief normaal (LV): 0.0254845 EUR/kWh

Electricity injection

- variable netgebruik: 0.001751 EUR/kWh

- databeheer: 17.85 EUR/year (LV) and 57.65 EUR/year (MV/distributiecabine)

- note/condition: databeheer for injection applies only if there is a separate injection access point; model via has_separate_injection_point

Gas offtake — non-telemetered classes

- T1 (0–5,000 kWh): fixed 14.79 EUR/year; variable 0.020533 EUR/kWh

- T2 (5,001–150,000 kWh): fixed 78.51 EUR/year; variable 0.0077903 EUR/kWh

- T3 (150,001–1,000,000 kWh): fixed 530.78 EUR/year; variable 0.0047752 EUR/kWh

- T4 (>1,000,000 kWh): fixed 4927.69 EUR/year; variable 0.0003783 EUR/kWh

Gas offtake — telemetered

- T5 (<10,000,000 kWh): variable 0.0003783 EUR/kWh; capacity 1.9710771 EUR/maxcap/year

- T6 (>10,000,000 kWh): variable 0.0003097 EUR/kWh; capacity 0.4743207 EUR/maxcap/year

Gas databeheer

- “Jaaropname”: 17.85 EUR/year

- MMR: 57.65 EUR/year

- AMR: 57.65 EUR/year

Gas ODV

- 0.0003667 EUR/kWh

Gas injection

- system management: 0.000963 EUR/kWh

- databeheer AMR: 57.65 EUR/year

## 5) Explicit development requirements (acceptance criteria)

### Functional requirements

- Support electricity and gas billing for a specified billing period.

- Support electricity contract types:

- dynamic: 15-min energy input + 15-min market price

- fixed/variable: single and dual day/night registers

- Day/night:

- Input supports two channels (kwh_day, kwh_night)

- TariffDefinition supports channel-specific components (energy day vs energy night)

- Support injection as separate calculation in v1.

- Apply monthly capacity peak:

- accept precomputed monthly peak series OR call pluggable peak calculator

- convert €/kW/month to a time-step allocation within the month

- Proration:

- annual fixed fees prorated over timesteps in billing period

- monthly fixed fees prorated over timesteps in each month

- Taxes:

- implement variable, fixed, tiered rules (tiered by annual/billing-period energy where applicable)

- VAT:

- enforce vat_applicable flags per component

- apply VAT rate determined from ConnectionInfo or region rules

### Data requirements

- Provide normalized internal datasets for:

- grid tariffs per DSO/region/carrier/direction/period

- tax rules per region/period/customer type

- Provide extraction scripts:

- parse DSO Excel inputs to normalized format

- allow regenerating datasets for new years/regions

### Engineering requirements

- Supplier-agnostic codebase:

- no supplier-specific modules/folders in library logic

- examples and fixtures allowed under examples/ or tests/fixtures/

- Deterministic outputs with stable column names

- Clear unit handling (c€/kWh vs €/kWh, €/MWh; conversion utilities)

- Extensive validation and meaningful errors (missing channels, resolution mismatch, missing metadata)

- Tests:

- dataset load tests

- end-to-end cost calculation with expected component sums

- proration and VAT applicability tests

- Documentation:

- quickstart with minimal example

- schemas for TariffDefinition and normalized grid/tax datasets

### v2 backlog (non-blocking)

- Joint offtake + injection calculation in one run (dual input, combined reporting)

- Index-weighted variable tariff calculation with synthetic profiles for BE-VLG

- Multi-region plugin support beyond Flanders