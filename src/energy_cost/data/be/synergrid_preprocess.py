import argparse
import datetime as dt
import re
import tempfile
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT_URL = "https://www.synergrid.be"
ROOT_HTML = f"{ROOT_URL}/nl/documentencentrum/statistieken-gegevens/profielen-slp-spp-rlp"
TZ = "Europe/Brussels"

PROFILE_LINK_PATTERNS = {
    "RLP0N": r"RLP0N\s*{year}\s*Electricity all DSOs",
    "SPP": r"SPP\s*{year}",
}

META_COLUMNS = {"cet", "year", "month", "day", "h", "min", "date"}
REGIONS = ("belgium", "flanders", "wallonia", "brussels")

FLANDERS_MARKERS = (
    "fluvius",
    "gaselwest",
    "imewo",
    "intergem",
    "iveka",
    "iverlek",
    "pbe",
    "sibelgas",
)
WALLONIA_MARKERS = ("ores", "resa", "aieg", "aiesh", "wavre")

# Static SPP GLN-to-region mapping based on Synergrid "Read Me First" perimeter table.
SPP_GLN_TO_REGION = {
    # Flanders
    "5414488000301": "flanders",  # IMEA (historical)
    "5414488000509": "flanders",  # IVEG (historical)
    "5414488000608": "flanders",  # IVERLEK 1
    "5414488000707": "flanders",  # IMEWO
    "5414488000806": "flanders",  # INTERGEM
    "5414488000905": "flanders",  # GASELWEST
    "5414488001001": "flanders",  # SIBELGAS
    "5414488001100": "flanders",  # IVERLEK 2
    "5414488001209": "flanders",  # WVEM (historical)
    "5414488001704": "flanders",  # Fluvius Antwerpen
    "5414488001803": "flanders",  # Fluvius Kempen/Iveka
    "5414492999998": "flanders",  # Fluvius Limburg
    "5414494999995": "flanders",  # PBE (historical)
    "5414494999996": "flanders",  # PBE Vlaanderen
    "5414496999987": "flanders",  # Fluvius West
    # Brussels
    "5414489000102": "brussels",  # SIBELGA-IE
    "5414489000409": "brussels",  # SIBELGA-SE
    # Wallonia
    "5414490000108": "wallonia",  # ORES Verviers
    "5414490000207": "wallonia",  # ORES Hainaut
    "5414490000504": "wallonia",  # ORES Brabant Wallon
    "5414490000603": "wallonia",  # ORES Hainaut
    "5414490000801": "wallonia",  # ORES Mouscron
    "5414490000900": "wallonia",  # ORES Namur
    "5414490001006": "wallonia",  # ORES Luxembourg
    "5414490001105": "wallonia",  # ORES Est
    "5414490001204": "wallonia",  # ORES (historical)
    "5414557999994": "wallonia",  # Regie de Wavre
    "5414567999991": "wallonia",  # RESA
    "5499842496501": "wallonia",  # AIEG
    "5499982193704": "wallonia",  # AIESH
}
SPP_BELGIUM_TOTAL_COLUMNS = {"SPPEXANTEBE", "MODELBE"}


class SynergridPreprocessError(RuntimeError):
    pass


def _normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("_", " ").replace("-", " ").strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text)
    return text


def _to_header_name(value: object) -> str:
    """Convert Excel header cell values to stable column names.

    GLN headers are often read as floats (e.g. 5414488000608.0). We normalize
    those to digit-only strings so downstream mapping can match reliably.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""

    if isinstance(value, (int, float)):
        as_float = float(value)
        if as_float.is_integer():
            return str(int(as_float))

    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text


def _profile_output_path(profile: str, output_dir: Path) -> Path:
    return output_dir / f"synergrid_{profile.lower()}.csv"


def _extract_version_parts(value: str) -> tuple[int, ...]:
    match = re.search(r"\bv(\d+(?:\.\d+)*)\b", value, flags=re.IGNORECASE)
    if not match:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))


def _find_download_url(profile: str, year: int, html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    label_pattern = PROFILE_LINK_PATTERNS[profile].format(year=year)
    excel_candidates: list[tuple[tuple[int, ...], int, str]] = []
    for index, link in enumerate(soup.find_all("a")):
        text = link.get_text(" ", strip=True)
        if not text:
            continue
        if re.search(label_pattern, text, flags=re.IGNORECASE):
            href_attr = link.get("href")
            if not isinstance(href_attr, str) or not href_attr:
                continue
            absolute = href_attr if href_attr.startswith("http") else f"{ROOT_URL}{href_attr}"
            if Path(urlparse(absolute).path).suffix.lower() in {".xlsx", ".xls", ".xlsb"}:
                version_parts = _extract_version_parts(text) or _extract_version_parts(absolute)
                excel_candidates.append((version_parts, index, absolute))

    if excel_candidates:
        # Prefer the highest explicit version when available (e.g. v1.1 over v1.0).
        # For ties, keep page order stability.
        selected = max(excel_candidates, key=lambda item: (item[0], -item[1]))
        return selected[2]
    raise SynergridPreprocessError(f"Could not find {profile} download link for year {year}.")


def _read_profile_sheet(file_path: Path, profile: str) -> pd.DataFrame:
    engine = "pyxlsb" if file_path.suffix.lower() == ".xlsb" else None
    if profile == "SPP" and file_path.suffix.lower() in {".xlsx", ".xls"}:
        workbook = pd.ExcelFile(file_path, engine=engine)
        ex_ante_sheets = [name for name in workbook.sheet_names if "ex ante" in _normalize_text(name)]
        sheet_name = ex_ante_sheets[0] if ex_ante_sheets else workbook.sheet_names[0]
        raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None, engine=engine)
    else:
        raw = pd.read_excel(file_path, header=None, engine=engine)
    first_col = raw.iloc[:, 0].astype(str).str.strip()
    header_candidates = raw.index[first_col.isin(["CET", "UTC"])]
    if header_candidates.empty:
        raise SynergridPreprocessError(f"Could not locate a row starting with CET/UTC in {file_path}.")

    header_row = int(header_candidates[0])
    base_header = raw.iloc[header_row].tolist()
    dgo_header: list[object] | None = None
    if header_row > 0:
        possible_dgo = raw.iloc[header_row - 1].tolist()
        if any(_normalize_text(v) == "dgo" for v in possible_dgo):
            dgo_header = possible_dgo

    columns: list[str] = []
    for idx, value in enumerate(base_header):
        name = _to_header_name(value)
        if dgo_header is not None and idx < len(dgo_header):
            dgo_name = _to_header_name(dgo_header[idx])
            if dgo_name and _normalize_text(dgo_name) != "dgo" and _normalize_text(name) not in META_COLUMNS:
                name = dgo_name
        if not name:
            name = f"column_{idx}"
        columns.append(name)

    data = raw.iloc[header_row + 1 :].copy()
    data.columns = columns

    timestamp_col = "CET" if "CET" in data.columns else "UTC" if "UTC" in data.columns else None
    if timestamp_col is None:
        raise SynergridPreprocessError(f"Missing CET/UTC column in parsed sheet for {file_path}.")

    if timestamp_col == "CET":
        numeric = pd.to_numeric(data["CET"], errors="coerce")
        data = data.assign(CET=numeric).dropna(subset=["CET"]).copy()
        data["timestamp"] = pd.to_datetime(data["CET"], origin="1899-12-30", unit="D").dt.round("15min")
        data["timestamp"] = data["timestamp"].dt.tz_localize(dt.timezone(dt.timedelta(hours=1))).dt.tz_convert(TZ)
    else:
        data = data.dropna(subset=["UTC"]).copy()
        data["timestamp"] = pd.to_datetime(data["UTC"], utc=True, errors="coerce").dt.tz_convert(TZ)
        data = data.dropna(subset=["timestamp"]).copy()

    return data


def _is_flanders_column(normalized: str) -> bool:
    return any(marker in normalized for marker in FLANDERS_MARKERS)


def _is_brussels_column(normalized: str) -> bool:
    return "sibelga" in normalized and "sibelgas" not in normalized


def _is_wallonia_column(normalized: str) -> bool:
    return any(marker in normalized for marker in WALLONIA_MARKERS)


def _extract_region_columns(data: pd.DataFrame, profile: str) -> dict[str, list[str]]:
    dso_candidates = [
        column
        for column in data.columns
        if _normalize_text(column) not in META_COLUMNS and column != "timestamp" and column != "CET"
    ]

    unique_candidates: list[str] = []
    seen_normalized: set[str] = set()
    for column in dso_candidates:
        normalized = _normalize_text(column)
        if not normalized or normalized.startswith("column "):
            continue
        if normalized in seen_normalized:
            continue
        seen_normalized.add(normalized)
        unique_candidates.append(column)

    regions: dict[str, list[str]] = {region: [] for region in REGIONS}
    for column in unique_candidates:
        normalized = _normalize_text(column)
        if _is_brussels_column(normalized):
            regions["brussels"].append(column)
        if _is_flanders_column(normalized):
            regions["flanders"].append(column)
        if _is_wallonia_column(normalized):
            regions["wallonia"].append(column)

    regions["belgium"] = unique_candidates

    for region in ("flanders", "wallonia", "brussels"):
        if not regions[region]:
            if profile == "SPP":
                # Some SPP exports expose DSO codes only, without labels.
                # In that case keep deterministic behavior by using all DSOs.
                regions[region] = unique_candidates.copy()
            else:
                raise SynergridPreprocessError(f"Could not infer any '{region}' DSO columns from sheet headers.")

    return regions


def _compute_region_dataframe(data: pd.DataFrame, profile: str) -> pd.DataFrame:
    if profile == "SPP":
        return _compute_spp_region_dataframe(data)

    region_columns = _extract_region_columns(data, profile)
    result = pd.DataFrame({"timestamp": data["timestamp"]})

    for region, columns in region_columns.items():
        numeric_values = data[columns].apply(pd.to_numeric, errors="coerce")
        result[region] = numeric_values.mean(axis=1)

    result = result.dropna(subset=["belgium"]).drop_duplicates(subset=["timestamp"], keep="last")
    return result.sort_values("timestamp").reset_index(drop=True)


def _compute_spp_region_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame({"timestamp": data["timestamp"]})

    dso_columns = [column for column in data.columns if re.fullmatch(r"\d{10,}", str(column))]
    total_columns = [
        column
        for column in data.columns
        if _normalize_text(column).replace(" ", "").upper() in SPP_BELGIUM_TOTAL_COLUMNS
    ]

    if dso_columns:
        mapped_dso_columns = [column for column in dso_columns if column in SPP_GLN_TO_REGION]
        if not mapped_dso_columns:
            raise SynergridPreprocessError("No mapped SPP GLN columns found in ex-ante sheet.")

        dso_numeric = data[mapped_dso_columns].apply(pd.to_numeric, errors="coerce")
        result["belgium"] = dso_numeric.mean(axis=1)

        for region in ("flanders", "wallonia", "brussels"):
            region_columns = [column for column in mapped_dso_columns if SPP_GLN_TO_REGION[column] == region]
            if not region_columns:
                raise SynergridPreprocessError(f"No SPP GLN columns mapped for region '{region}'.")
            result[region] = dso_numeric[region_columns].mean(axis=1)
    elif total_columns:
        # 2026 ex-ante currently exposes only Belgian total. Reuse it for all regions.
        total = pd.to_numeric(data[total_columns[0]], errors="coerce")
        result["belgium"] = total
        result["flanders"] = total
        result["wallonia"] = total
        result["brussels"] = total
    else:
        raise SynergridPreprocessError("SPP ex-ante sheet does not contain GLN columns or Belgian total column.")

    result = result.dropna(subset=["belgium"]).drop_duplicates(subset=["timestamp"], keep="last")
    return result.sort_values("timestamp").reset_index(drop=True)


def _download_profile_file(profile: str, year: int) -> Path:
    response = requests.get(ROOT_HTML, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    response.raise_for_status()
    href = _find_download_url(profile, year, response.text)

    excel = requests.get(href, headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
    excel.raise_for_status()

    parsed = urlparse(href)
    suffix = Path(parsed.path).suffix or ".xlsx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{profile}_{year}{suffix}") as temporary_file:
        temporary_file.write(excel.content)
        return Path(temporary_file.name)


def _parse_profile_year(profile: str, year: int) -> pd.DataFrame:
    file_path = _download_profile_file(profile, year)
    try:
        data = _read_profile_sheet(file_path, profile)
        data = data[data["timestamp"].dt.year == year].copy()
        if data.empty:
            raise SynergridPreprocessError(f"No rows found for year {year} in downloaded {profile} file.")
        return _compute_region_dataframe(data, profile)
    finally:
        file_path.unlink(missing_ok=True)


def _read_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", *REGIONS])
    data = pd.read_csv(path, float_precision="round_trip")
    if "timestamp" not in data.columns:
        raise SynergridPreprocessError(f"Missing timestamp column in existing CSV {path}.")

    # Existing files contain timezone-aware strings with mixed offsets (+01:00/+02:00).
    # Parse to UTC first, then normalize to Europe/Brussels.
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    if data["timestamp"].isna().any():
        invalid_rows = int(data["timestamp"].isna().sum())
        raise SynergridPreprocessError(f"Found {invalid_rows} invalid timestamp values in existing CSV {path}.")
    data["timestamp"] = data["timestamp"].dt.tz_convert(TZ)
    return data


def update_profile_csv(profile: str, years: list[int], output_dir: Path) -> tuple[Path, list[int]]:
    output_path = _profile_output_path(profile, output_dir)
    existing = _read_existing(output_path)

    frames = [existing] if not existing.empty else []
    appended: list[int] = []

    for year in years:
        year_df = _parse_profile_year(profile, year)
        frames.append(year_df)
        appended.append(year)

    if not frames:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["timestamp", *REGIONS]).to_csv(output_path, index=False)
        return output_path, appended

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return output_path, appended


def _parse_years(args: argparse.Namespace) -> list[int]:
    if args.year:
        return sorted(set(args.year))
    current_year = dt.datetime.now().year
    return [current_year]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and append Synergrid profile data into CSV files.")
    parser.add_argument("--profile", choices=["RLP0N", "SPP", "all"], default="all")
    parser.add_argument("--year", type=int, action="append", help="Year to fetch (can be passed multiple times).")
    args = parser.parse_args()

    years = _parse_years(args)
    profiles = ["RLP0N", "SPP"] if args.profile == "all" else [args.profile]
    output_dir = Path(__file__).parent

    for profile in profiles:
        output_path, appended_years = update_profile_csv(profile, years, output_dir)
        if appended_years:
            print(f"{profile}: appended years {appended_years} -> {output_path}")
        else:
            print(f"{profile}: no changes -> {output_path}")


if __name__ == "__main__":
    main()
