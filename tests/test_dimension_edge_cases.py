import numpy as np
import tifffile

from cellpose import io, transforms
from guv_app.services.image_service import ImageService
from guv_app.services.segmentation_service import SegmentationService


def image_service_without_thread():
    return ImageService.__new__(ImageService)


def test_prepare_image_array_drops_singleton_time_z_and_preserves_channels_last():
    svc = image_service_without_thread()
    arr = np.zeros((1, 1, 24, 32, 2), dtype=np.uint16)
    meta = io.ImageMeta(
        axes="TZYXC",
        shape=arr.shape,
        sizes={"T": 1, "Z": 1, "Y": 24, "X": 32, "C": 2},
        dtype=arr.dtype,
    )

    prepared = svc._prepare_image_array(arr, meta)

    assert prepared.shape == (24, 32, 2)


def test_prepare_image_array_handles_channels_first_singleton_time():
    svc = image_service_without_thread()
    arr = np.zeros((1, 2, 24, 32), dtype=np.uint16)
    meta = io.ImageMeta(
        axes="TCYX",
        shape=arr.shape,
        sizes={"T": 1, "C": 2, "Y": 24, "X": 32},
        dtype=arr.dtype,
    )

    prepared = svc._prepare_image_array(arr, meta)

    assert prepared.shape == (24, 32, 2)


def test_imagej_rgb_stack_with_timing_metadata_is_treated_as_time(tmp_path):
    path = tmp_path / "rgb_timelapse.tif"
    arr = np.zeros((4, 24, 32, 3), dtype=np.uint8)
    arr[2, ..., 0] = 123
    tifffile.imwrite(
        path,
        arr,
        imagej=True,
        metadata={
            "axes": "ZYXS",
            "fps": 0.2,
            "finterval": 5.0,
            "sampled_timestamps_seconds": [0.0, 5.0, 10.0, 15.0],
        },
    )

    assert io.get_series_time_info(str(path)) == (None, 1, 4)

    frame = io.read_image_frame(str(path), "T2")

    assert frame.array.shape == (24, 32, 3)
    assert frame.meta.axes == "YXC"
    assert int(frame.array[..., 0].max()) == 123


def test_image_service_iter_frames_keeps_sliced_tiff_frame_metadata_in_sync(tmp_path):
    path = tmp_path / "rgb_timelapse.tif"
    arr = np.zeros((4, 24, 32, 3), dtype=np.uint8)
    arr[1, ..., 1] = 77
    tifffile.imwrite(
        path,
        arr,
        imagej=True,
        metadata={
            "axes": "ZYXS",
            "fps": 0.2,
            "finterval": 5.0,
            "sampled_timestamps_seconds": [0.0, 5.0, 10.0, 15.0],
        },
    )
    svc = image_service_without_thread()
    svc._frame_cache = {}
    svc._stack_axis_overrides = {}

    refs = svc.build_frame_references(str(path))
    frames = svc.iter_image_frames(str(path))
    loaded = svc.load_frame(str(path), "T1")

    assert [ref.split("::")[-1] for ref in refs] == ["T0", "T1", "T2", "T3"]
    assert [frame.frame_id for frame in frames] == ["T0", "T1", "T2", "T3"]
    assert all(frame.array.shape == (24, 32, 3) for frame in frames)
    assert all(frame.meta.axes == "YXC" for frame in frames)
    assert loaded.shape == (24, 32, 3)
    assert int(loaded[..., 1].max()) == 77


def test_prepare_image_array_preserves_ambiguous_two_plane_channels_first_stack():
    svc = image_service_without_thread()
    arr = np.zeros((2, 24, 32), dtype=np.uint16)

    prepared = svc._prepare_image_array(arr)

    assert prepared.shape == (24, 32, 2)


def test_convert_image_channels_last_singleton_channel_pads_to_three_channels():
    arr = np.zeros((24, 32, 1), dtype=np.float32)

    converted = transforms.convert_image(arr, channel_axis=-1, z_axis=None, do_3D=False)

    assert converted.shape == (24, 32, 3)


def test_convert_image_batched_channels_last_input_is_preserved():
    arr = np.zeros((2, 24, 32, 1), dtype=np.float32)

    converted = transforms.convert_image(arr, channel_axis=-1, z_axis=None, do_3D=False)

    assert converted.shape == (2, 24, 32, 3)


def test_postprocess_classes_accepts_channels_last_semantic_styles_with_singleton_batch():
    svc = SegmentationService.__new__(SegmentationService)
    masks = np.zeros((24, 32), dtype=np.int32)
    masks[2:8, 2:8] = 1
    masks[12:18, 12:18] = 2
    styles = np.zeros((1, 24, 32, 4), dtype=np.float32)
    styles[..., 1] = 0.2
    styles[:, 2:8, 2:8, 2] = 0.9
    styles[:, 12:18, 12:18, 3] = 0.9

    classes, classes_map = svc.postprocess_classes(masks, styles)

    assert classes_map.shape == masks.shape
    assert classes.tolist() == [0, 2, 3]


def test_postprocess_classes_accepts_channels_first_semantic_styles_with_singleton_batch():
    svc = SegmentationService.__new__(SegmentationService)
    masks = np.zeros((24, 32), dtype=np.int32)
    masks[2:8, 2:8] = 1
    styles = np.zeros((1, 4, 24, 32), dtype=np.float32)
    styles[:, 1, :, :] = 0.2
    styles[:, 2, 2:8, 2:8] = 0.9

    classes, classes_map = svc.postprocess_classes(masks, styles)

    assert classes_map.shape == masks.shape
    assert classes.tolist() == [0, 2]
