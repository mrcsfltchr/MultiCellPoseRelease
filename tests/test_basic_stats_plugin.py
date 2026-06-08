import numpy as np

from guv_app.plugins.basic_stats import BasicStatsPlugin


def test_basic_stats_ignores_missing_sparse_labels():
    masks = np.zeros((12, 12), dtype=np.int32)
    masks[1:3, 1:3] = 1
    masks[6:9, 6:10] = 3
    image = np.dstack([
        np.full(masks.shape, 10.0),
        np.full(masks.shape, 20.0),
    ])

    df = BasicStatsPlugin().run(image, masks, intensity_channel="all")

    assert df["mask_id"].tolist() == [1, 3]
    assert df["area"].tolist() == [4, 12]
    assert (df["area"] >= 1).all()
    assert "mean_intensity_ch1" in df.columns
    assert "mean_intensity_ch2" in df.columns
