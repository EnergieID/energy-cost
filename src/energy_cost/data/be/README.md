# About this data

This directory contains info on the Belgian energy market.
This includes distributor tariffs, government fees and taxes.

## Choices
currently we only support residential and small business (laagspanningsnet en niet telegemetered gas).
We also only support the digital meter (slimme meter).

## Sources
More info on how the Belgian energy market is structured: https://www.creg.be/nl/consumenten/energiemarkt/hoe-de-energieprijs-opgebouwd

Distributor tariffs are parsed from the yearly updated Excel sheets at https://www.vlaamsenutsregulator.be/elektriciteit-en-aardgas/nettarieven/hoeveel-bedragen-de-distributienettarieven
Regulations on how these tariffs work exactly are explained here: https://assets.vlaamsenutsregulator.be/2025-11/Tariefmethodologie%20reguleringsperiode%202025-2028%20-%20BESL-2024-41.pdf
Fluxys transport tariffs (gas) are listed here: https://www.fluxys.com/nl/natural-gas-and-biomethane/empowering-you/tariffs/tariff_fluxys-belgium-domestic-2026

Flemish fees are listed here: https://www.vlaanderen.be/belastingen-en-begroting/vlaamse-belastingen/energieheffingen
Federal fees are listed here: https://www.minfin.fgov.be/myminfin-web/pages/public/fisconet/document/b91925cd-fbba-4da5-8b9a-06974300ff1e

## Refresh Synergrid profile data
Synergrid load/solar profiles are preprocessed into CSV files and committed in the repository.
Runtime calculations read those CSV files directly (no live Synergrid scraping during index evaluation).

Append/update year(s) for both profiles:

```bash
poe synergrid --year 2027
```

You can also specify a profile to update, e.g. `RLP0N` or `SPP`, and multiple years:

```bash
poe synergrid --profile SPP --year 2026 --year 2027
```
