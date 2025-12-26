import pytest
from pathlib import Path


@pytest.fixture
def project_root():
    return Path(__file__).parent.parent


@pytest.fixture
def sample_school():
    return {
        "dbn": "07X001",
        "school_name": "PS 1 COURTLANDT SCHOOL",
        "address": "335 East 152 Street",
        "latitude": 40.8245,
        "longitude": -73.9067,
        "borough": "X"
    }

