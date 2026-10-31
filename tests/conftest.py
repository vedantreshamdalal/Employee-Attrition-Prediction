"""Train the saved pipeline once per test session."""

import pytest

from ml.train import train_models


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    return train_models()
