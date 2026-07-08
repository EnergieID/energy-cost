import pandas as pd

from energy_cost.data.be import synergrid_preprocess


def _year_frame(year: int, value: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start=f"{year}-01-01T00:00:00",
                periods=2,
                freq="15min",
                tz="Europe/Brussels",
            ),
            "belgium": [value, value],
            "flanders": [value, value],
            "wallonia": [value, value],
            "brussels": [value, value],
        }
    )


def test_update_profile_csv_refetches_existing_years(tmp_path, monkeypatch) -> None:
    out_dir = tmp_path
    existing = _year_frame(2025, 1.0)
    existing.to_csv(out_dir / "synergrid_rlp0n.csv", index=False)

    def fake_parse(profile: str, year: int) -> pd.DataFrame:
        return _year_frame(year, 9.0)

    monkeypatch.setattr(synergrid_preprocess, "_parse_profile_year", fake_parse)

    output_path, appended = synergrid_preprocess.update_profile_csv("RLP0N", [2025], out_dir)

    assert output_path.exists()
    assert appended == [2025]

    merged = pd.read_csv(output_path)
    assert set(merged["belgium"].round(3)) == {9.0}


def test_update_profile_csv_keeps_non_overlapping_existing_rows(tmp_path, monkeypatch) -> None:
    out_dir = tmp_path
    existing = pd.concat([_year_frame(2025, 1.0), _year_frame(2026, 2.0)], ignore_index=True)
    existing.to_csv(out_dir / "synergrid_rlp0n.csv", index=False)

    def fake_parse(profile: str, year: int) -> pd.DataFrame:
        # Return only one row for 2025 to verify dedup replaces overlap,
        # while non-overlapping existing rows are retained.
        return _year_frame(year, 9.0).iloc[:1].copy()

    monkeypatch.setattr(synergrid_preprocess, "_parse_profile_year", fake_parse)

    output_path, appended = synergrid_preprocess.update_profile_csv("RLP0N", [2025], out_dir)

    assert output_path.exists()
    assert appended == [2025]

    merged = pd.read_csv(output_path, parse_dates=["timestamp"])
    merged["year"] = pd.to_datetime(merged["timestamp"]).dt.year

    # For 2025 we keep one old non-overlapping row + one new overlapping row replaced.
    assert len(merged[merged["year"] == 2025]) == 2
    assert set(merged.loc[merged["year"] == 2025, "belgium"].round(3)) == {1.0, 9.0}

    # Non-requested years stay untouched.
    assert len(merged[merged["year"] == 2026]) == 2
    assert set(merged.loc[merged["year"] == 2026, "belgium"].round(3)) == {2.0}


def test_compute_spp_region_dataframe_uses_gln_columns_only() -> None:
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01T00:00:00+01:00", periods=2, freq="15min"),
            "UTC": ["2024-12-31 23:00:00", "2024-12-31 23:15:00"],
            "Year": [2025, 2025],
            "Hour": [0, 0],
            "5414488001704": [2.0, 4.0],  # Flanders
            "5414490000900": [6.0, 8.0],  # Wallonia
            "5414489000102": [10.0, 12.0],  # Brussels
        }
    )

    result = synergrid_preprocess._compute_spp_region_dataframe(data)

    assert result["flanders"].tolist() == [2.0, 4.0]
    assert result["wallonia"].tolist() == [6.0, 8.0]
    assert result["brussels"].tolist() == [10.0, 12.0]
    assert result["belgium"].tolist() == [6.0, 8.0]


def test_compute_spp_region_dataframe_uses_belgium_total_for_all_regions() -> None:
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01T00:00:00+01:00", periods=2, freq="15min"),
            "SPPExanteBE": [0.0, 0.5],
        }
    )

    result = synergrid_preprocess._compute_spp_region_dataframe(data)

    assert result["belgium"].tolist() == [0.0, 0.5]
    assert result["flanders"].tolist() == [0.0, 0.5]
    assert result["wallonia"].tolist() == [0.0, 0.5]
    assert result["brussels"].tolist() == [0.0, 0.5]


def test_compute_spp_region_dataframe_accepts_float_gln_headers() -> None:
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01T00:00:00+01:00", periods=1, freq="15min"),
            5414488001704.0: [2.0],  # Flanders GLN as float header
            5414490000900.0: [6.0],  # Wallonia GLN as float header
            5414489000102.0: [10.0],  # Brussels GLN as float header
        }
    )

    data.columns = [synergrid_preprocess._to_header_name(c) for c in data.columns]
    result = synergrid_preprocess._compute_spp_region_dataframe(data)

    assert result["belgium"].tolist() == [6.0]
    assert result["flanders"].tolist() == [2.0]
    assert result["wallonia"].tolist() == [6.0]
    assert result["brussels"].tolist() == [10.0]
