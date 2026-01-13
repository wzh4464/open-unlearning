"""
Shared test fixtures for LMCleaner tests.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
import torch.nn as nn
import tempfile


@pytest.fixture
def simple_model():
    """Create a simple linear model for testing"""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 5)
            self.linear2 = nn.Linear(5, 2)

        def forward(self, input_ids, labels=None, **kwargs):
            x = self.linear1(input_ids.float())
            logits = self.linear2(x)
            loss = None
            if labels is not None:
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
            return type('Output', (), {'logits': logits, 'loss': loss})()

    model = SimpleModel()
    return model


@pytest.fixture
def batch_data():
    """Create sample batch data"""
    return {
        'input_ids': torch.randn(4, 10),
        'labels': torch.randint(0, 2, (4,))
    }


@pytest.fixture
def temp_log_dir():
    """Create temporary directory for logs"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
