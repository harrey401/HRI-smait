"""Shared test fixtures for SMAIT-v3 unit tests."""

import pytest
import numpy as np
from smait.core.config import Config, reset_config
from smait.core.events import EventBus


@pytest.fixture(autouse=True)
def reset_config_singleton():
    """Reset config singleton before and after each test to prevent state leakage."""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def config():
    """Return a fresh default Config instance."""
    return Config()


@pytest.fixture
def event_bus():
    """Return a fresh EventBus instance."""
    return EventBus()


@pytest.fixture
def silence_audio():
    """16kHz mono silence, 1 second (int16)."""
    return np.zeros(16000, dtype=np.int16)


@pytest.fixture
def speech_audio():
    """16kHz mono noise simulating speech, 1 second (int16)."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal(16000) * 3000).astype(np.int16)
