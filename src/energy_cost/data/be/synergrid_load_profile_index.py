import datetime as dt
import re

import pandas as pd
import pytz
import pyxlsb
import requests
from bs4 import BeautifulSoup

from energy_cost.index import Index
from energy_cost.resolution import Resolution

ROOT_URL = "https://www.synergrid.be"
ROOT_HTML = f"{ROOT_URL}/nl/documentencentrum/statistieken-gegevens/profielen-slp-spp-rlp"
FILE_NAMES = {
    "RLP0N": "RLP0N ?{year} Electricity all DSOs",
    "SPP": "SPP {year} ex-ante and ex-post",
}


class SynergridLoadProfileIndex(Index):
    """
    Index for Synergrid load profiles.

    This index fetches load profile data from the Synergrid Excel files and provides it in a standardized format.
    """

    def __init__(self, profile: str, resolution: Resolution) -> None:
        super().__init__(resolution)
        self.profile = profile

    def _get_values(self, start: dt.datetime, end: dt.datetime, timezone: dt.tzinfo) -> pd.DataFrame:
        years = range(start.year, end.year + 1)
        dfs = []

        # fetch root HTMl to get the download links for the specified years
        response = requests.get(ROOT_HTML, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for year in years:
            file_name = FILE_NAMES.get(self.profile, "").format(year=year)
            link = soup.find(name="a", string=re.compile(file_name))  # ty: ignore[no-matching-overload]
            if not link:
                raise ValueError(f"Could not find download link for {file_name} in year {year}.")
            href = link.get("href")
            if not href:
                raise ValueError(f"No href found for {file_name} in year {year}.")
            href = href if href.startswith("http") else f"{ROOT_URL}{href}"

            excel = requests.get(href, headers={"User-Agent": "Mozilla/5.0"})
            excel.raise_for_status()

            # save the Excel file to a temporary location
            temp_file_path = f"temp_{year}.xlsb"
            with open(temp_file_path, "wb") as f:
                f.write(excel.content)

            # detect header row (with CET), and skip rows until we find it
            content = pyxlsb.open_workbook(temp_file_path)
            sheet = content.get_sheet(1)
            ln = 0
            for row in sheet.rows():
                if row[0].v == "CET":
                    break
                ln += 1
            else:
                raise ValueError(f"Could not find header row with 'CET' in {file_name} for year {year}.")

            # Download the Excel file
            rlp = (
                pd.read_excel(
                    temp_file_path,
                    skiprows=ln,
                    usecols="A,H:AF",
                    converters={
                        "CET": lambda x: pd.to_datetime(x, origin="1899-12-30", unit="D").round(
                            freq="15min"
                        ),  # weird xlsb time format
                    },
                )
                .set_index("CET")
                .tz_localize(tz=pytz.FixedOffset(60))
                .tz_convert(timezone)
            )
            rlp.index.name = None
            rlp.dropna(axis=1, how="all", inplace=True)
            rlp = rlp.mean(axis=1).round(10)

            df = pd.DataFrame({"timestamp": pd.to_datetime(rlp.index).as_unit("us"), "value": rlp.values})
            dfs.append(df)

        return pd.concat(dfs).reset_index(drop=True)
