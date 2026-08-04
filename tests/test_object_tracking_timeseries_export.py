import pandas as pd
import numpy as np

from guv_app.plugins.object_tracking import ObjectTrackingPlugin
from guv_app.plugins.object_tracking_timeseries_export import (
    export_tracking_timeseries_csvs,
    reshape_tracking_positions,
    reshape_tracking_timeseries,
    tracking_position_tables,
    tracking_timeseries_tables,
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
        "object_1_mean_intensity",
        "object_1_area",
        "object_2_mean_intensity",
        "object_2_area",
    ]
    assert wide["object_1_mean_intensity"].iloc[:2].tolist() == [10.0, 11.0]
    assert pd.isna(wide["object_1_mean_intensity"].iloc[2])
    assert wide["object_2_area"].iloc[[0, 2]].tolist() == [200.0, 202.0]
    assert pd.isna(wide["object_2_area"].iloc[1])


def test_reshape_tracking_timeseries_auto_exports_measured_channel_columns():
    df = pd.DataFrame(
        {
            "frame_index": [0, 1],
            "track_id": [1, 1],
            "mean_intensity": [15.0, 16.0],
            "mean_intensity_ch1": [10.0, 11.0],
            "background_intensity_ch1": [1.0, 1.5],
            "mean_intensity_bg_subtracted_ch1": [9.0, 9.5],
            "mean_intensity_ch3": [20.0, 21.0],
            "background_intensity_ch3": [2.0, 2.5],
            "mean_intensity_bg_subtracted_ch3": [18.0, 18.5],
            "area": [100, 101],
        }
    )

    wide = reshape_tracking_timeseries(df, intensity_columns="auto")

    assert wide.columns.tolist() == [
        "frame_index",
        "object_1_mean_intensity_ch1",
        "object_1_background_intensity_ch1",
        "object_1_mean_intensity_bg_subtracted_ch1",
        "object_1_mean_intensity_ch3",
        "object_1_background_intensity_ch3",
        "object_1_mean_intensity_bg_subtracted_ch3",
        "object_1_area",
    ]
    assert wide["object_1_background_intensity_ch1"].tolist() == [1.0, 1.5]


def test_tracking_timeseries_tables_keeps_time_down_rows_and_objects_grouped():
    df = pd.DataFrame(
        {
            "filename": ["movie.nd2::P0_T0", "movie.nd2::P0_T0", "movie.nd2::P0_T1"],
            "frame_index": [0, 0, 1],
            "track_id": [1, 2, 1],
            "mean_intensity_ch1": [10.0, 20.0, 11.0],
            "background_intensity_ch1": [1.0, 2.0, 1.1],
            "mean_intensity_bg_subtracted_ch1": [9.0, 18.0, 9.9],
            "mean_intensity_ch2": [100.0, 200.0, 101.0],
            "background_intensity_ch2": [10.0, 20.0, 10.1],
            "mean_intensity_bg_subtracted_ch2": [90.0, 180.0, 90.9],
            "area": [50, 60, 51],
        }
    )

    tables = tracking_timeseries_tables(df, intensity_columns="auto")

    assert len(tables) == 1
    _, wide = tables[0]
    assert wide.columns.tolist() == [
        "frame_index",
        "object_1_mean_intensity_ch1",
        "object_1_background_intensity_ch1",
        "object_1_mean_intensity_bg_subtracted_ch1",
        "object_1_mean_intensity_ch2",
        "object_1_background_intensity_ch2",
        "object_1_mean_intensity_bg_subtracted_ch2",
        "object_1_area",
        "object_2_mean_intensity_ch1",
        "object_2_background_intensity_ch1",
        "object_2_mean_intensity_bg_subtracted_ch1",
        "object_2_mean_intensity_ch2",
        "object_2_background_intensity_ch2",
        "object_2_mean_intensity_bg_subtracted_ch2",
        "object_2_area",
    ]
    assert wide["frame_index"].tolist() == [0, 1]


def test_reshape_tracking_positions_keeps_time_down_rows_and_xy_by_object():
    df = pd.DataFrame(
        {
            "frame_index": [0, 0, 1],
            "track_id": [1, 2, 1],
            "centroid_x": [5.0, 20.0, 6.0],
            "centroid_y": [7.0, 22.0, 8.0],
        }
    )

    wide = reshape_tracking_positions(df)

    assert wide.columns.tolist() == [
        "frame_index",
        "object_1_x",
        "object_1_y",
        "object_2_x",
        "object_2_y",
    ]
    assert wide["frame_index"].tolist() == [0, 1]
    assert wide["object_1_x"].tolist() == [5.0, 6.0]
    assert wide["object_1_y"].tolist() == [7.0, 8.0]
    assert wide["object_2_x"].iloc[0] == 20.0
    assert pd.isna(wide["object_2_x"].iloc[1])


def test_tracking_position_tables_split_by_position_series():
    df = pd.DataFrame(
        {
            "filename": ["movie.nd2::P0_T0", "movie.nd2::P0_T1", "movie.nd2::P1_T0"],
            "frame_index": [0, 1, 0],
            "track_id": [1, 1, 1],
            "centroid_x": [5.0, 6.0, 50.0],
            "centroid_y": [7.0, 8.0, 70.0],
        }
    )

    tables = tracking_position_tables(df)

    assert [name for name, _ in tables] == ["movie_P0", "movie_P1"]
    assert tables[0][1]["object_1_x"].tolist() == [5.0, 6.0]
    assert tables[1][1]["object_1_y"].tolist() == [70.0]


def test_object_tracking_visualization_uses_editable_track_ids():
    plugin = ObjectTrackingPlugin()
    image = np.zeros((32, 32), dtype=float)
    masks0 = image.astype("int32")
    masks0[5:9, 5:9] = 1
    masks1 = image.astype("int32")
    masks1[14:18, 14:18] = 1
    masks1[20:24, 20:24] = 2

    viz0 = plugin.visualize(image, masks0, filename="movie.tif::T0")
    viz1 = plugin.visualize(image, masks1, filename="movie.tif::T1")

    assert int(viz0.max()) == 1
    assert set(int(v) for v in pd.unique(pd.Series(viz1.ravel())) if int(v) > 0) == {1, 2}
    assert int(viz1[11, 11]) == 1
    assert np.count_nonzero(viz1[masks1 == 1]) < np.count_nonzero(masks1 == 1)


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
            "centroid_x": [5.0, 6.0, 50.0, 51.0],
            "centroid_y": [7.0, 8.0, 70.0, 71.0],
            "area": [100, 101, 300, 301],
        }
    ).to_csv(input_csv, index=False)

    written = export_tracking_timeseries_csvs(str(input_csv), overwrite=True)

    assert [p.split("\\")[-1].split("/")[-1] for p in written] == [
        "movie_P0_object_tracking_timeseries.csv",
        "movie_P1_object_tracking_timeseries.csv",
        "movie_P0_object_tracking_positions.csv",
        "movie_P1_object_tracking_positions.csv",
    ]
    p0 = pd.read_csv(tmp_path / "movie_P0_object_tracking_timeseries.csv")
    p1 = pd.read_csv(tmp_path / "movie_P1_object_tracking_timeseries.csv")
    p0_pos = pd.read_csv(tmp_path / "movie_P0_object_tracking_positions.csv")
    assert p0["object_1_area"].tolist() == [100, 101]
    assert p1["object_1_mean_intensity"].tolist() == [30.0, 31.0]
    assert p0_pos["object_1_x"].tolist() == [5.0, 6.0]
    assert p0_pos["object_1_y"].tolist() == [7.0, 8.0]
