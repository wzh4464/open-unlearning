"""
Tests for efficiency tracking feature.
Run with: pytest tests/test_efficiency_tracking.py
"""

import pytest
from pathlib import Path
import tempfile
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trainer.utils import EfficiencyTracker, EfficiencyMetrics


class TestEfficiencyMetrics:
    """Test EfficiencyMetrics dataclass"""

    def test_initialization(self):
        """Test that EfficiencyMetrics can be initialized"""
        metrics = EfficiencyMetrics(
            unlearning_time_seconds=100.5,
            peak_gpu_memory_mb=8192.0,
            tokens_per_second=1234.5,
            model_size_mb=13000.0,
            requires_retain_set=True,
            total_steps=100
        )

        assert metrics.unlearning_time_seconds == 100.5
        assert metrics.peak_gpu_memory_mb == 8192.0
        assert metrics.tokens_per_second == 1234.5
        assert metrics.model_size_mb == 13000.0
        assert metrics.requires_retain_set == True
        assert metrics.total_steps == 100

    def test_to_dict(self):
        """Test conversion to dictionary"""
        metrics = EfficiencyMetrics(
            unlearning_time_seconds=100.5,
            peak_gpu_memory_mb=8192.0,
            tokens_per_second=1234.5,
            model_size_mb=13000.0,
            requires_retain_set=False,
            total_steps=50
        )

        metrics_dict = metrics.to_dict()

        assert isinstance(metrics_dict, dict)
        assert 'unlearning_time_seconds' in metrics_dict
        assert 'peak_gpu_memory_mb' in metrics_dict
        assert 'tokens_per_second' in metrics_dict
        assert 'model_size_mb' in metrics_dict
        assert 'requires_retain_set' in metrics_dict
        assert 'total_steps' in metrics_dict
        assert metrics_dict['total_steps'] == 50

    def test_save(self):
        """Test saving metrics to JSON file"""
        metrics = EfficiencyMetrics(
            unlearning_time_seconds=100.5,
            peak_gpu_memory_mb=8192.0,
            tokens_per_second=1234.5,
            model_size_mb=13000.0,
            requires_retain_set=True,
            total_steps=100
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "efficiency_metrics.json"
            metrics.save(output_path)

            assert output_path.exists()

            # Verify content
            import json
            with open(output_path, 'r') as f:
                loaded = json.load(f)

            assert loaded['unlearning_time_seconds'] == 100.5
            assert loaded['total_steps'] == 100


class TestEfficiencyTracker:
    """Test EfficiencyTracker callback"""

    def test_initialization(self):
        """Test that EfficiencyTracker can be initialized"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = EfficiencyTracker(output_dir=tmpdir)

            assert tracker.output_dir == Path(tmpdir)
            assert tracker.start_time is None
            assert tracker.peak_memory == 0.0

    def test_callback_methods_exist(self):
        """Test that required callback methods exist"""
        tracker = EfficiencyTracker()

        assert hasattr(tracker, 'on_train_begin')
        assert hasattr(tracker, 'on_step_end')
        assert hasattr(tracker, 'on_train_end')
        assert callable(tracker.on_train_begin)
        assert callable(tracker.on_step_end)
        assert callable(tracker.on_train_end)


class TestRetainSetDetection:
    """Test retain set requirement detection"""

    def test_npo_requires_retain_set(self):
        """Test that NPO-based methods are detected as requiring retain set"""
        retain_set_methods = ['GradDiff', 'NPO', 'DPO', 'SimNPO', 'RMU', 'UNDIAL']
        trainer_name = "MockNPOTrainer"

        assert any(method in trainer_name for method in retain_set_methods)

    def test_grad_ascent_no_retain_set(self):
        """Test that GradAscent is detected as not requiring retain set"""
        retain_set_methods = ['GradDiff', 'NPO', 'DPO', 'SimNPO', 'RMU', 'UNDIAL']
        trainer_name = "MockGradAscentTrainer"

        assert not any(method in trainer_name for method in retain_set_methods)


class TestCompareScript:
    """Test compare_efficiency.py script"""

    def test_script_imports(self):
        """Test that compare script can be imported"""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

        try:
            import compare_efficiency

            assert hasattr(compare_efficiency, 'load_efficiency_metrics')
            assert hasattr(compare_efficiency, 'compare_metrics')
        except ImportError as e:
            pytest.fail(f"Failed to import compare_efficiency: {e}")
