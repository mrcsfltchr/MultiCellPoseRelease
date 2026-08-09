import numpy as np
import tifffile

from cellpose import train
from tools.train_cpsam_finetune_balanced import build_lazy_semantic_cache


def test_get_batch_can_keep_full_semantic_label_channels(tmp_path):
    image_path = tmp_path / "image.npy"
    label_path = tmp_path / "semantic_flow.npy"
    image = np.zeros((16, 18, 3), dtype=np.float32)
    label = np.zeros((5, 16, 18), dtype=np.float32)
    label[1, 3:8, 4:9] = 2
    np.save(image_path, image)
    np.save(label_path, label)

    _imgs, labels = train._get_batch(
        [0],
        data=None,
        labels=None,
        files=[str(image_path)],
        labels_files=[str(label_path)],
        keep_label_first_channel=True,
    )

    assert labels[0].shape == (5, 16, 18)
    assert int(labels[0][1].max()) == 2


def test_get_batch_legacy_file_mode_still_drops_mask_channel(tmp_path):
    image_path = tmp_path / "image.npy"
    label_path = tmp_path / "flows.npy"
    np.save(image_path, np.zeros((16, 18, 3), dtype=np.float32))
    np.save(label_path, np.zeros((4, 16, 18), dtype=np.float32))

    _imgs, labels = train._get_batch(
        [0],
        data=None,
        labels=None,
        files=[str(image_path)],
        labels_files=[str(label_path)],
    )

    assert labels[0].shape == (3, 16, 18)


def test_lazy_semantic_cache_writes_image_views_and_full_labels(tmp_path):
    image_path = tmp_path / "sample.tif"
    label_path = tmp_path / "sample_masks.tif"
    class_path = tmp_path / "sample_classes.tif"
    image = np.zeros((24, 28, 2), dtype=np.uint16)
    image[..., 0] = 100
    image[..., 1] = 200
    masks = np.zeros((24, 28), dtype=np.uint16)
    masks[5:12, 6:14] = 1
    classes = np.zeros((24, 28), dtype=np.uint8)
    classes[masks == 1] = 2
    tifffile.imwrite(image_path, image)
    tifffile.imwrite(label_path, masks)
    tifffile.imwrite(class_path, classes)

    records = [{
        "image": str(image_path),
        "label": str(label_path),
        "frame_id": None,
        "source_group": "unit",
        "split": "train",
    }]

    image_files, label_files, valid_records, invalid, class_rows = build_lazy_semantic_cache(
        records,
        tmp_path / "cache",
        npz_mask_channel="last",
        channel_sampling_mode="single-and-all",
        max_all_channel_combos=0,
        seed=1,
        npz_cache_dir=None,
        semantic_classes=4,
        split_name="train",
    )

    assert not invalid
    assert len(image_files) == 3
    assert len(label_files) == 3
    assert len(valid_records) == 3
    assert class_rows
    label = np.load(label_files[0])
    assert label.shape[0] == 5
    assert int(label[1].max()) == 2
