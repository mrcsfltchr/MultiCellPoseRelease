import pandas as pd

from guv_app.plugins.object_tracking_timeseries_export import (
    export_tracking_timeseries_csvs,
    reshape_tracking_timeseries,
)


def test_reshape_tracking_timeseries_pairs_intensity_and_area_by_track():
    df = pd.DataFrame(
        {
            "frame_index": [0, 0, 1, 2],
            "track_id": [1, 2, 1, 2],
            "mean_intensity": [10.0, 20.0, 11.0, 22.0],
            "area": [100, 200, 101, 202],
        }
    )

    wide = reshape_tracking_timeseries(df)

    assert wide.columns.tolist() == [
        "frame_index",
        "track_1_mean_intensity",
        "track_1_area",
        "track_2_mean_intensity",
        "track_2_area",
    ]
    assert wide["track_1_mean_intensity"].iloc[:2].tolist() == [10.0, 11.0]
    assert pd.isna(wide["track_1_mean_intensity"].iloc[2])
    assert wide["track_2_area"].iloc[[0, 2]].tolist() == [200.0, 202.0]
    assert pd.isna(wide["track_2_area"].iloc[1])


def test_reshape_tracking_timeseries_auto_exports_measured_channel_columns():
    df = pd.DataFrame(
        {
            "frame_index": [0, 1],
            "track_id": [1, 1],
            "mean_intensity": [15.0, 16.0],
            "mean_intensity_ch1": [10.0, 11.0],
            "mean_intensity_ch3": [20.0, 21.0],
            "area": [100, 101],
        }
    )

    wide = reshape_tracking_timeseries(df, intensity_columns="auto")

    assert wide.columns.tolist() == [
        "frame_index",
        "track_1_mean_intensity_ch1",
        "track_1_mean_intensity_ch3",
        "track_1_area",
    ]


def test_export_tracking_timeseries_writes_one_file_per_position_series(tmp_path):
    input_csv = tmp_path / "statistics_results__Object_Tracking.csv"
    pd.DataFrame(
        {
            "filename": [
                "movie.nd2::P0_T0",
                "movie.nd2::P0_T1",
                "movie.nd2::P1_T0",
                "movie.nd2::P1_T1",
            ],
            "frame_index": [0, 1, 0, 1],
            "track_id": [1, 1, 1, 1],
            "mean_intensity": [10.0, 11.0, 30.0, 31.0],
            "area": [100, 101, 300, 301],
        }
    ).to_csv(input_csv, index=False)

    written = export_tracking_timeseries_csvs(str(input_csv), overwrite=True)

    assert [p.split("\\")[-1].split("/")[-1] for p in written] == [
        "movie_P0_object_tracking_timeseries.csv",
        "movie_P1_object_tracking_timeseries.csv",
    ]
    p0 = pd.read_csv(tmp_path / "movie_P0_object_tracking_timeseries.csv")
    p1 = pd.read_csv(tmp_path / "movie_P1_object_tracking_timeseries.csv")
    assert p0["track_1_area"].tolist() == [100, 101]
    assert p1["track_1_mean_intensity"].tolist() == [30.0, 31.0]
