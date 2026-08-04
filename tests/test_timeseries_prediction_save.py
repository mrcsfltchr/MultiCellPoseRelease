import numpy as np

from cellpose import io
from guv_app.data_models.results import InferenceResult
from guv_app.models.app_state import ApplicationStateModel


def test_save_prediction_uses_result_masks_for_non_current_timeseries_frame(tmp_path):
    movie_path = tmp_path / "movie.tif"
    live_masks = np.zeros((1, 12, 12), dtype=np.uint16)
    live_masks[2:5, 2:5] = 1
    result_masks = np.zeros((12, 12), dtype=np.uint16)
    result_masks[7:10, 7:10] = 1

    model = ApplicationStateModel()
    model.filename = str(movie_path)
    model.frame_id = "T0"
    model.raw_image = np.zeros((12, 12), dtype=np.float32)
    model.NZ = 1
    model.Ly = 12
    model.Lx = 12
    model.channel_segmentations[0] = {
        "masks": live_masks,
        "outpix": np.zeros_like(live_masks),
        "mask_classes": np.array([0, 1], dtype=np.int16),
        "instance_colors": np.zeros((2, 3), dtype=np.uint8),
        "pred_classes_map": None,
        "flows": [],
    }

    result = InferenceResult(
        filename=str(movie_path),
        frame_id="T1",
        masks=result_masks,
        classes=np.array([0, 1], dtype=np.int16),
        channel_index=0,
    )

    model.save_prediction(result)

    saved_path = tmp_path / f"movie{io.frame_id_to_suffix('T1')}_pred.npy"
    saved = np.load(saved_path, allow_pickle=True).item()
    saved_channel_masks = saved["channel_segmentations"]["0"]["masks"]

    np.testing.assert_array_equal(saved["masks"], result_masks)
    np.testing.assert_array_equal(saved_channel_masks, result_masks[np.newaxis, ...])
    assert not np.array_equal(saved_channel_masks, live_masks)
