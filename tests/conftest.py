import pytest

from latedeliveries.data import make_orders


@pytest.fixture
def sample_data():
    return make_orders(n=300, seed=7)
