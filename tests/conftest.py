import pytest
from transformers import AutoTokenizer


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
