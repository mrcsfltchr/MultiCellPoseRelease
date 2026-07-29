import numpy as np

from cellpose.models import CellposeModel


def test_resize_gradients_restores_multiple_singleton_axes():
    model = CellposeModel.__new__(CellposeModel)
    grads = np.zeros((1, 2, 8, 8), dtype=np.float32)

    resized = model._resize_gradients(grads, to_y_size=16, to_x_size=16)

    assert resized.shape == (1, 2, 16, 16)


def test_resize_cellprob_restores_singleton_axis():
    model = CellposeModel.__new__(CellposeModel)
    prob = np.zeros((1, 8, 8), dtype=np.float32)

    resized = model._resize_cellprob(prob, to_y_size=16, to_x_size=16)

    assert resized.shape == (1, 16, 16)
