import numpy as np
import pytest

from latedeliveries.evaluation import choose_threshold, order_cost


def test_cost_of_small_example():
    y_true = [1, 0, 0, 1, 0]
    flagged = [True, True, False, False, False]
    assert order_cost(y_true, flagged) == 2 * 4 + 1 * 15


@pytest.mark.parametrize(
    ("y_true", "flagged", "expected"),
    [
        ([0, 0, 0], [False, False, False], 0),
        ([1, 1], [False, False], 30),
        ([1, 0, 1], [True, True, True], 12),
    ],
)
def test_cost_examples(y_true, flagged, expected):
    assert order_cost(y_true, flagged) == expected


def test_threshold_can_choose_flag_nobody():
    threshold, results = choose_threshold(
        np.array([0, 0, 0, 0]), np.array([0.01, 0.02, 0.03, 0.04])
    )
    assert threshold == 0.95
    assert min(item["cost"] for item in results) == 0


def test_threshold_can_choose_flag_everybody():
    threshold, results = choose_threshold(
        np.array([1, 1, 1, 1]), np.array([0.96, 0.97, 0.98, 0.99])
    )
    assert threshold == 0.05
    assert min(item["cost"] for item in results) == 16
