import numpy as np

from guv_app.plugins.object_tracking import ObjectTrackingPlugin


def _frame(objects, shape=(32, 32)):
    masks = np.zeros(shape, dtype=np.int32)
    image = np.zeros(shape, dtype=np.float32)
    for mid, y0, x0, size, intensity in objects:
        masks[y0:y0 + size, x0:x0 + size] = mid
        image[y0:y0 + size, x0:x0 + size] = intensity
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
