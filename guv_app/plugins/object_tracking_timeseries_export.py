import argparse
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


REQUIRED_COLUMNS = {"frame_index", "track_id", "area", "mean_intensity"}


def export_tracking_timeseries_csvs(
    input_csv: str,
    output_dir: Optional[str] = None,
    intensity_column: str = "mean_intensity",
    area_column: str = "area",
    overwrite: bool = False,
) -> List[str]:
    """
    Convert Object Tracking plugin output into one wide CSV per time series.

    Output columns are arranged as adjacent intensity/area pairs for each track:
    frame_index, track_1_mean_intensity, track_1_area, track_2_mean_intensity, ...
    """
    input_path = Path(input_csv)
    df = pd.read_csv(input_path)
    _validate_tracking_table(df, intensity_column=intensity_column, area_column=area_column)

    out_dir = Path(output_dir) if output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for series_key, series_df in _iter_timeseries(df, fallback_name=input_path.stem):
        wide = reshape_tracking_timeseries(
            series_df,
            intensity_column=intensity_column,
            area_column=area_column,
        )
        out_path = out_dir / f"{_safe_filename(series_key)}_object_tracking_timeseries.csv"
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"{out_path} already exists; pass --overwrite to replace it")
        wide.to_csv(out_path, index=False)
        written.append(str(out_path))
    return written


def reshape_tracking_timeseries(
    df: pd.DataFrame,
    intensity_column: str = "mean_intensity",
    area_column: str = "area",
) -> pd.DataFrame:
    track_ids = sorted(df["track_id"].dropna().astype(int).unique())
    frames = sorted(df["frame_index"].dropna().astype(int).unique())
    wide = pd.DataFrame({"frame_index": frames})

    for track_id in track_ids:
        track_df = df[df["track_id"].astype(int) == track_id]
        intensity_by_frame = _series_by_frame(track_df, intensity_column)
        area_by_frame = _series_by_frame(track_df, area_column)
        title = f"track_{track_id}"
        wide[f"{title}_{intensity_column}"] = wide["frame_index"].map(intensity_by_frame)
        wide[f"{title}_{area_column}"] = wide["frame_index"].map(area_by_frame)
    return wide


def _validate_tracking_table(
    df: pd.DataFrame,
    intensity_column: str = "mean_intensity",
    area_column: str = "area",
) -> None:
    required = {"frame_index", "track_id", intensity_column, area_column}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            "Input CSV does not look like Object Tracking output. "
            f"Missing columns: {', '.join(missing)}"
        )


def _iter_timeseries(df: pd.DataFrame, fallback_name: str) -> Iterable[Tuple[str, pd.DataFrame]]:
    if "filename" not in df.columns:
        yield fallback_name, df
        return

    keyed = df.copy()
    keyed["_timeseries_key"] = keyed["filename"].map(lambda value: _timeseries_key(value, fallback_name))
    for key in sorted(keyed["_timeseries_key"].dropna().unique()):
        yield key, keyed[keyed["_timeseries_key"] == key].drop(columns=["_timeseries_key"])


def _timeseries_key(filename, fallback_name: str) -> str:
    if pd.isna(filename) or not str(filename).strip():
        return fallback_name
    text = str(filename)
    base, frame_id = _split_frame_reference(text)
    stem = Path(base).stem
    if not frame_id:
        return stem
    tokens = re.findall(r"([A-Za-z])(\d+)", frame_id)
    if not any(axis.upper() == "T" for axis, _ in tokens):
        return f"{stem}_{frame_id}"
    non_time = [f"{axis.upper()}{value}" for axis, value in tokens if axis.upper() != "T"]
    return f"{stem}_{'_'.join(non_time)}" if non_time else stem


def _split_frame_reference(filename: str) -> Tuple[str, Optional[str]]:
    if "::" not in filename:
        return filename, None
    base, frame_id = filename.split("::", 1)
    return base, frame_id


def _series_by_frame(df: pd.DataFrame, value_column: str) -> Dict[int, float]:
    values = (
        df[["frame_index", value_column]]
        .dropna(subset=["frame_index"])
        .groupby("frame_index", sort=True)[value_column]
        .first()
    )
    return {int(frame): value for frame, value in values.items()}


def _safe_filename(value: str) -> str:
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(value).strip())
    safe = re.sub(r"_+", "_", safe).strip("._")
    return safe or "object_tracking_timeseries"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reshape Object Tracking plugin CSV output into one wide time-series CSV "
            "with adjacent intensity and area columns for each track."
        )
    )
    parser.add_argument("input_csv", help="CSV produced by the Object Tracking analysis plugin.")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Directory for generated CSV files. Defaults to the input CSV directory.",
    )
    parser.add_argument(
        "--intensity-column",
        default="mean_intensity",
        help="Tracking output column to use for intensity values.",
    )
    parser.add_argument(
        "--area-column",
        default="area",
        help="Tracking output column to use for area values.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output CSVs.")
    args = parser.parse_args(argv)

    written = export_tracking_timeseries_csvs(
        args.input_csv,
        output_dir=args.output_dir,
        intensity_column=args.intensity_column,
        area_column=args.area_column,
        overwrite=args.overwrite,
    )
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
