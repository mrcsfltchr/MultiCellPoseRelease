import numpy as np

from guv_app.plugins.object_tracking import ObjectTrackingPlugin


def _frame(objects, shape=(32, 32)):
    masks = np.zeros(shape, dtype=np.int32)
    image = np.zeros(shape, dtype=np.float32)
    for mid, y0, x0, size, intensity in objects:
        masks[y0:y0 + size, x0:x0 + size] = mid
        image[y0:y0 + size, x0:x0 + size] = intensity
    return image, masks


def _multichannel_frame(objects, shape=(32, 32), channels=3):
    masks = np.zeros(shape, dtype=np.int32)
    image = np.zeros((*shape, channels), dtype=np.float32)
    for mid, y0, x0, size, intensities in objects:
        masks[y0:y0 + size, x0:x0 + size] = mid
        for channel, intensity in enumerate(intensities):
            image[y0:y0 + size, x0:x0 + size, channel] = intensity
    return image, masks


def test_tracking_matches_objects_and_creates_new_tracks_for_births():
    plugin = ObjectTrackingPlugin()
    img0, m0 = _frame([(1, 4, 4, 4, 10.0), (2, 20, 20, 4, 80.0)])
    img1, m1 = _frame([(1, 5, 5, 4, 11.0), (2, 21, 21, 4, 79.0), (3, 4, 22, 4, 30.0)])

    df0 = plugin.run(img0, m0, filename="movie.tif::T0")
    df1 = plugin.run(img1, m1, filename="movie.tif::T1")

    assert df0.sort_values("mask_id")["track_id"].tolist() == [1, 2]
    by_mask = df1.set_index("mask_id")
    assert int(by_mask.loc[1, "track_id"]) == 1
    assert int(by_mask.loc[2, "track_id"]) == 2
    assert int(by_mask.loc[3, "track_id"]) == 3
    assert by_mask.loc[3, "status"] == "new"


def test_tracking_recovers_after_short_gap_and_rejects_after_long_gap():
    plugin = ObjectTrackingPlugin()
    img0, m0 = _frame([(1, 4, 4, 4, 10.0)])
    img1, m1 = _frame([])
    img2, m2 = _frame([(1, 6, 6, 4, 11.0)])
    img5, m5 = _frame([(1, 8, 8, 4, 12.0)])

    plugin.run(img0, m0, filename="movie.tif::T0", max_frame_gap=1)
    empty = plugin.run(img1, m1, filename="movie.tif::T1", max_frame_gap=1)
    recovered = plugin.run(img2, m2, filename="movie.tif::T2", max_frame_gap=1)
    new_after_gap = plugin.run(img5, m5, filename="movie.tif::T5", max_frame_gap=1)

    assert empty.empty
    assert recovered.iloc[0]["status"] == "gap_closed"
    assert int(recovered.iloc[0]["track_id"]) == 1
    assert new_after_gap.iloc[0]["status"] == "new"
    assert int(new_after_gap.iloc[0]["track_id"]) == 2


def test_tracking_class_filter_treats_wrong_class_frame_as_gap():
    plugin = ObjectTrackingPlugin()
    img0, m0 = _frame([(1, 4, 4, 4, 10.0)])
    img1, m1 = _frame([(1, 5, 5, 4, 11.0)])
    img2, m2 = _frame([(1, 6, 6, 4, 12.0)])
    class1 = np.array([0, 1], dtype=np.int32)
    class2 = np.array([0, 2], dtype=np.int32)

    first = plugin.run(img0, m0, classes=class1, filename="movie.tif::T0", track_class_id="1", max_frame_gap=1)
    missed = plugin.run(img1, m1, classes=class2, filename="movie.tif::T1", track_class_id="1", max_frame_gap=1)
    recovered = plugin.run(img2, m2, classes=class1, filename="movie.tif::T2", track_class_id="1", max_frame_gap=1)

    assert int(first.iloc[0]["track_id"]) == 1
    assert missed.empty
    assert int(recovered.iloc[0]["track_id"]) == 1
    assert recovered.iloc[0]["status"] == "gap_closed"
    assert int(recovered.iloc[0]["gap_frames"]) == 1


def test_tracking_stack_input_is_pure_and_returns_all_frames():
    plugin = ObjectTrackingPlugin()
    img0, m0 = _frame([(1, 4, 4, 4, 10.0)])
    img1, m1 = _frame([(1, 5, 5, 4, 11.0)])
    images = np.stack([img0, img1], axis=0)
    masks = np.stack([m0, m1], axis=0)

    df = plugin.run(images, masks)

    assert df["frame_index"].tolist() == [0, 1]
    assert df["track_id"].tolist() == [1, 1]
    assert df["status"].tolist() == ["new", "matched"]


def test_tracking_uses_area_and_intensity_to_reduce_identity_swaps():
    plugin = ObjectTrackingPlugin()
    img0, m0 = _frame([(1, 5, 5, 3, 10.0), (2, 5, 20, 6, 80.0)])
    img1, m1 = _frame([(1, 5, 6, 3, 11.0), (2, 5, 19, 6, 79.0)])

    plugin.run(img0, m0, filename="movie.tif::T0")
    df1 = plugin.run(img1, m1, filename="movie.tif::T1")

    by_mask = df1.set_index("mask_id")
    assert int(by_mask.loc[1, "track_id"]) == 1
    assert int(by_mask.loc[2, "track_id"]) == 2


def test_tracking_channels_and_measurement_channels_are_independent():
    plugin = ObjectTrackingPlugin()
    img0, m0 = _multichannel_frame(
        [
            (1, 5, 5, 4, [10.0, 100.0, 1000.0]),
            (2, 5, 20, 4, [90.0, 20.0, 2000.0]),
        ]
    )
    img1, m1 = _multichannel_frame(
        [
            (1, 5, 5, 4, [91.0, 21.0, 2100.0]),
            (2, 5, 20, 4, [11.0, 99.0, 1100.0]),
        ]
    )

    plugin.run(
        img0,
        m0,
        filename="movie.tif::T0",
        tracking_channels="2",
        measurement_channels="1,3",
        distance_weight=0,
        area_weight=0,
        shape_weight=0,
        intensity_weight=1,
    )
    df1 = plugin.run(
        img1,
        m1,
        filename="movie.tif::T1",
        tracking_channels="2",
        measurement_channels="1,3",
        distance_weight=0,
        area_weight=0,
        shape_weight=0,
        intensity_weight=1,
    )

    by_mask = df1.set_index("mask_id")
    assert int(by_mask.loc[2, "track_id"]) == 1
    assert int(by_mask.loc[1, "track_id"]) == 2
    assert "tracking_mean_intensity" in df1.columns
    assert "mean_intensity_ch1" in df1.columns
    assert "mean_intensity_ch3" in df1.columns
    assert "mean_intensity_ch2" not in df1.columns
    assert float(by_mask.loc[2, "tracking_mean_intensity"]) == 99.0
    assert float(by_mask.loc[2, "mean_intensity_ch1"]) == 11.0
    assert float(by_mask.loc[2, "mean_intensity_ch3"]) == 1100.0


def test_tracking_reports_local_background_and_subtracted_intensities_by_default():
    plugin = ObjectTrackingPlugin()
    image = np.full((32, 32), 2.0, dtype=np.float32)
    masks = np.zeros((32, 32), dtype=np.int32)
    masks[12:16, 12:16] = 1
    image[masks == 1] = 10.0

    df = plugin.run(
        image,
        masks,
        filename="movie.tif::T0",
        background_inner_gap_px=0,
        background_outer_radius_px=4,
        background_percentile=50.0,
    )

    row = df.iloc[0]
    assert float(row["tracking_mean_intensity"]) == 10.0
    assert float(row["tracking_background_intensity"]) == 2.0
    assert float(row["tracking_mean_intensity_bg_subtracted"]) == 8.0
    assert float(row["mean_intensity"]) == 10.0
    assert float(row["mean_intensity_ch1"]) == 10.0
    assert float(row["background_intensity_ch1"]) == 2.0
    assert float(row["mean_intensity_bg_subtracted"]) == 8.0
    assert float(row["mean_intensity_bg_subtracted_ch1"]) == 8.0


def test_tracking_can_disable_local_background_measurement():
    plugin = ObjectTrackingPlugin()
    image = np.full((32, 32), 2.0, dtype=np.float32)
    masks = np.zeros((32, 32), dtype=np.int32)
    masks[12:16, 12:16] = 1
    image[masks == 1] = 10.0

    df = plugin.run(image, masks, filename="movie.tif::T0", local_background_subtraction=False)

    row = df.iloc[0]
    assert float(row["tracking_mean_intensity"]) == 10.0
    assert np.isnan(row["tracking_background_intensity"])
    assert np.isnan(row["tracking_mean_intensity_bg_subtracted"])
    assert np.isnan(row["background_intensity_ch1"])
    assert np.isnan(row["mean_intensity_bg_subtracted_ch1"])


def test_position_only_frames_do_not_track_as_timepoints():
    plugin = ObjectTrackingPlugin()
    img0, m0 = _frame([(1, 4, 4, 4, 10.0)])
    img1, m1 = _frame([(1, 5, 5, 4, 11.0)])

    df0 = plugin.run(img0, m0, filename="positions.nd2::P0")
    df1 = plugin.run(img1, m1, filename="positions.nd2::P1")

    assert int(df0.iloc[0]["track_id"]) == 1
    assert int(df1.iloc[0]["track_id"]) == 1
    assert df1.iloc[0]["status"] == "new"


def test_multi_position_time_series_tracks_within_position_only():
    plugin = ObjectTrackingPlugin()
    img0, m0 = _frame([(1, 4, 4, 4, 10.0)])
    img1, m1 = _frame([(1, 5, 5, 4, 11.0)])

    df0 = plugin.run(img0, m0, filename="movie.nd2::P0_T0")
    df1 = plugin.run(img1, m1, filename="movie.nd2::P0_T1")
    df_other_position = plugin.run(img1, m1, filename="movie.nd2::P1_T0")

    assert int(df0.iloc[0]["track_id"]) == 1
    assert int(df1.iloc[0]["track_id"]) == 1
    assert df1.iloc[0]["status"] == "matched"
    assert int(df_other_position.iloc[0]["track_id"]) == 1
    assert df_other_position.iloc[0]["status"] == "new"


def test_tracking_filters_to_approved_track_ids():
    plugin = ObjectTrackingPlugin()
    img0, m0 = _frame([(1, 4, 4, 4, 10.0), (2, 20, 20, 4, 80.0)])
    img1, m1 = _frame([(1, 5, 5, 4, 11.0), (2, 21, 21, 4, 79.0)])

    plugin.run(img0, m0, filename="movie.tif::T0")
    df1 = plugin.run(img1, m1, filename="movie.tif::T1", approved_track_ids=[2])

    assert df1["track_id"].tolist() == [2]
    assert df1["mask_id"].tolist() == [2]
