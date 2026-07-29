import numpy as np

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
