import numpy as np

from cellpose.semantic_label_utils import discard_class_zero_instances


def test_discard_class_zero_instances_removes_and_compacts_ids():
    masks = np.array(
        [
            [0, 1, 1, 0],
            [2, 2, 0, 5],
            [0, 0, 5, 5],
        ],
        dtype=np.int32,
    )
    classes = np.array([0, 0, 2, 0, 0, 3], dtype=np.int16)

    filtered, filtered_classes = discard_class_zero_instances(masks, classes)

    np.testing.assert_array_equal(
        filtered,
        np.array(
            [
                [0, 0, 0, 0],
                [1, 1, 0, 2],
                [0, 0, 2, 2],
            ],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(filtered_classes, np.array([0, 2, 3], dtype=np.int16))


def test_discard_class_zero_instances_preserves_singleton_3d_shape():
    masks = np.array([[[0, 1], [2, 2]]], dtype=np.uint16)
    classes = np.array([0, 1, 0], dtype=np.int16)

    filtered, filtered_classes = discard_class_zero_instances(masks, classes)

    assert filtered.shape == masks.shape
    assert filtered.dtype == masks.dtype
    np.testing.assert_array_equal(filtered, np.array([[[0, 1], [0, 0]]], dtype=np.uint16))
    np.testing.assert_array_equal(filtered_classes, np.array([0, 1], dtype=np.int16))


def test_discard_class_zero_instances_removes_all_unclassified_masks():
    masks = np.array([[1, 1], [0, 2]], dtype=np.int32)
    classes = np.zeros(3, dtype=np.int16)

    filtered, filtered_classes = discard_class_zero_instances(masks, classes)

    assert not np.any(filtered)
    np.testing.assert_array_equal(filtered_classes, np.array([0], dtype=np.int16))


def test_discard_class_zero_instances_leaves_nonsemantic_result_unchanged():
    masks = np.array([[0, 4], [4, 0]], dtype=np.int32)

    filtered, filtered_classes = discard_class_zero_instances(masks, None)

    assert filtered is masks
    assert filtered_classes is None
