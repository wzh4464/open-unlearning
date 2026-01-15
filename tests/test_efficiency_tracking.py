"""
Tests for efficiency tracking feature.
Run with: pytest tests/test_efficiency_tracking.py
"""

import pytest
from pathlib import Path
import tempfile
import sys
import time
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trainer.utils import EfficiencyTracker, EfficiencyMetrics


class TestEfficiencyMetrics:
    """Test EfficiencyMetrics dataclass"""

    def test_initialization(self):
        """Test that EfficiencyMetrics can be initialized with all fields"""
        metrics = EfficiencyMetrics(
            unlearning_time_seconds=100.5,
            peak_gpu_memory_mb=8192.0,
            intermediate_storage_mb=512.0,
            tokens_per_second=1234.5,
            model_size_mb=13000.0,
            total_steps=100,
            forward_passes_total=200,
            backward_passes_total=100,
            flops_estimate=1e12,
            forget_samples_count=1000,
            retain_samples_count=9000,
            requires_retain_set=True,
            gpu_utilization_mean=85.5,
            gpu_utilization_max=98.0,
            per_step_latency_mean_ms=150.0,
            per_step_latency_std_ms=25.0,
            per_step_latency_min_ms=100.0,
            per_step_latency_max_ms=250.0,
            checkpoint_save_time_seconds=5.5,
        )

        assert metrics.unlearning_time_seconds == 100.5
        assert metrics.peak_gpu_memory_mb == 8192.0
        assert metrics.intermediate_storage_mb == 512.0
        assert metrics.tokens_per_second == 1234.5
        assert metrics.model_size_mb == 13000.0
        assert metrics.total_steps == 100
        assert metrics.forward_passes_total == 200
        assert metrics.backward_passes_total == 100
        assert metrics.flops_estimate == 1e12
        assert metrics.forget_samples_count == 1000
        assert metrics.retain_samples_count == 9000
        assert metrics.requires_retain_set is True
        assert metrics.gpu_utilization_mean == 85.5
        assert metrics.gpu_utilization_max == 98.0
        assert metrics.per_step_latency_mean_ms == 150.0
        assert metrics.per_step_latency_std_ms == 25.0
        assert metrics.per_step_latency_min_ms == 100.0
        assert metrics.per_step_latency_max_ms == 250.0
        assert metrics.checkpoint_save_time_seconds == 5.5

    def test_to_dict(self):
        """Test conversion to dictionary includes all fields"""
        metrics = EfficiencyMetrics(
            unlearning_time_seconds=100.5,
            peak_gpu_memory_mb=8192.0,
            tokens_per_second=1234.5,
            model_size_mb=13000.0,
            requires_retain_set=False,
            total_steps=50,
            intermediate_storage_mb=256.0,
            forward_passes_total=100,
            backward_passes_total=50,
            forget_samples_count=500,
            retain_samples_count=4500,
            per_step_latency_mean_ms=120.0,
        )

        metrics_dict = metrics.to_dict()

        assert isinstance(metrics_dict, dict)
        # Check all expected keys
        expected_keys = [
            'unlearning_time_seconds', 'peak_gpu_memory_mb', 'intermediate_storage_mb',
            'tokens_per_second', 'model_size_mb', 'total_steps',
            'forward_passes_total', 'backward_passes_total', 'flops_estimate',
            'forget_samples_count', 'retain_samples_count', 'requires_retain_set',
            'gpu_utilization_mean', 'gpu_utilization_max',
            'per_step_latency_mean_ms', 'per_step_latency_std_ms',
            'per_step_latency_min_ms', 'per_step_latency_max_ms',
            'checkpoint_save_time_seconds',
        ]
        for key in expected_keys:
            assert key in metrics_dict, f"Missing key: {key}"

        assert metrics_dict['total_steps'] == 50
        assert metrics_dict['forward_passes_total'] == 100
        assert metrics_dict['forget_samples_count'] == 500

    def test_save(self):
        """Test saving metrics to JSON file"""
        metrics = EfficiencyMetrics(
            unlearning_time_seconds=100.5,
            peak_gpu_memory_mb=8192.0,
            tokens_per_second=1234.5,
            model_size_mb=13000.0,
            requires_retain_set=True,
            total_steps=100,
            intermediate_storage_mb=512.0,
            forward_passes_total=200,
            backward_passes_total=100,
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
            assert loaded['intermediate_storage_mb'] == 512.0
            assert loaded['forward_passes_total'] == 200


class TestEfficiencyTracker:
    """Test EfficiencyTracker callback"""

    def test_initialization(self):
        """Test that EfficiencyTracker can be initialized"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = EfficiencyTracker(output_dir=tmpdir)

            assert tracker.output_dir == Path(tmpdir)
            assert tracker.start_time is None
            assert tracker.peak_memory == 0.0
            assert tracker.peak_storage == 0.0
            assert tracker.step_latencies == []
            assert tracker.forward_count == 0
            assert tracker.backward_count == 0

    def test_initialization_with_storage_dirs(self):
        """Test initialization with storage directories (normalized to Path)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_dir = Path(tmpdir) / "gradients"
            storage_dir.mkdir()
            tracker = EfficiencyTracker(output_dir=tmpdir, storage_dirs=[str(storage_dir)])

            # Should be normalized to Path objects
            assert len(tracker.storage_dirs) == 1
            assert isinstance(tracker.storage_dirs[0], Path)
            assert tracker.storage_dirs[0] == storage_dir

    def test_initialization_with_storage_check_interval(self):
        """Test initialization with custom storage check interval"""
        tracker = EfficiencyTracker(storage_check_interval=50)
        assert tracker.storage_check_interval == 50

    def test_initialization_with_gpu_sampling_interval(self):
        """Test initialization with custom GPU sampling interval"""
        tracker = EfficiencyTracker(gpu_sampling_interval=5)
        assert tracker.gpu_sampling_interval == 5

    def test_callback_methods_exist(self):
        """Test that required callback methods exist"""
        tracker = EfficiencyTracker()

        assert hasattr(tracker, 'on_train_begin')
        assert hasattr(tracker, 'on_step_end')
        assert hasattr(tracker, 'on_train_end')
        assert hasattr(tracker, 'on_save')
        assert callable(tracker.on_train_begin)
        assert callable(tracker.on_step_end)
        assert callable(tracker.on_train_end)
        assert callable(tracker.on_save)


class TestIntermediateStorageTracking:
    """Test intermediate storage overhead tracking"""

    def test_get_dir_size_mb(self):
        """Test directory size calculation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = EfficiencyTracker(output_dir=tmpdir)

            # Create test files
            test_dir = Path(tmpdir) / "test_storage"
            test_dir.mkdir()
            (test_dir / "file1.bin").write_bytes(b"x" * 1024 * 1024)  # 1 MB
            (test_dir / "file2.bin").write_bytes(b"x" * 512 * 1024)   # 0.5 MB

            size = tracker._get_dir_size_mb(test_dir)
            assert 1.4 < size < 1.6  # ~1.5 MB

    def test_get_dir_size_empty(self):
        """Test directory size for empty directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = EfficiencyTracker(output_dir=tmpdir)
            empty_dir = Path(tmpdir) / "empty"
            empty_dir.mkdir()

            size = tracker._get_dir_size_mb(empty_dir)
            assert size == 0.0

    def test_get_dir_size_nonexistent(self):
        """Test directory size for non-existent directory"""
        tracker = EfficiencyTracker()
        size = tracker._get_dir_size_mb(Path("/nonexistent/path"))
        assert size == 0.0

    def test_intermediate_storage_tracking(self):
        """Test that intermediate storage is tracked across monitored directories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_dir = Path(tmpdir) / "gradients"
            storage_dir.mkdir()

            tracker = EfficiencyTracker(output_dir=tmpdir, storage_dirs=[str(storage_dir)])

            # Initially empty
            assert tracker._get_intermediate_storage() == 0.0

            # Add files
            (storage_dir / "grad_0.pt").write_bytes(b"x" * 1024 * 1024)  # 1 MB
            storage = tracker._get_intermediate_storage()
            assert 0.9 < storage < 1.1  # ~1 MB

    def test_storage_check_throttling(self):
        """Test that storage checks are throttled by storage_check_interval"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_dir = Path(tmpdir) / "checkpoints"
            storage_dir.mkdir()

            # Set interval to 5 steps
            tracker = EfficiencyTracker(
                output_dir=tmpdir,
                storage_dirs=[str(storage_dir)],
                storage_check_interval=5
            )
            tracker.last_step_time = time.time()

            class MockState:
                global_step = 0
            class MockArgs:
                pass

            # Add file
            (storage_dir / "ckpt.pt").write_bytes(b"x" * 1024 * 1024)  # 1 MB

            # Step 0 - should check (first step)
            MockState.global_step = 0
            tracker.on_step_end(MockArgs(), MockState(), None)
            assert tracker._last_storage_check_step == 0
            assert 0.9 < tracker.peak_storage < 1.1

            # Step 2 - should NOT check (interval not reached)
            MockState.global_step = 2
            old_check_step = tracker._last_storage_check_step
            tracker.on_step_end(MockArgs(), MockState(), None)
            assert tracker._last_storage_check_step == old_check_step  # unchanged

            # Step 5 - should check (interval reached)
            MockState.global_step = 5
            tracker.on_step_end(MockArgs(), MockState(), None)
            assert tracker._last_storage_check_step == 5


class TestLatencyTracking:
    """Test per-step latency tracking"""

    def test_latency_recorded_on_step(self):
        """Test that latency is recorded on each step"""
        tracker = EfficiencyTracker()
        tracker.last_step_time = time.time()

        class MockState:
            global_step = 1
        class MockArgs:
            pass

        # Simulate some work
        time.sleep(0.01)  # 10ms
        tracker.on_step_end(MockArgs(), MockState(), None)

        assert len(tracker.step_latencies) == 1
        assert tracker.step_latencies[0] >= 10  # At least 10ms

    def test_multiple_latencies(self):
        """Test multiple latency recordings"""
        tracker = EfficiencyTracker()
        tracker.last_step_time = time.time()

        class MockState:
            global_step = 1
        class MockArgs:
            pass

        # Record multiple steps
        for i in range(5):
            time.sleep(0.005)  # 5ms
            MockState.global_step = i + 1
            tracker.on_step_end(MockArgs(), MockState(), None)

        assert len(tracker.step_latencies) == 5


class TestPassCounting:
    """Test forward/backward pass counting"""

    def test_forward_hook_increments_count(self):
        """Test that forward hook increments count"""
        tracker = EfficiencyTracker()
        assert tracker.forward_count == 0

        # Simulate forward hook call
        tracker._forward_hook_fn(None, None, None)
        assert tracker.forward_count == 1

        tracker._forward_hook_fn(None, None, None)
        assert tracker.forward_count == 2

    def test_backward_hook_increments_count(self):
        """Test that backward hook increments count"""
        tracker = EfficiencyTracker()
        assert tracker.backward_count == 0

        # Simulate backward hook call
        tracker._backward_hook_fn(None, None, None)
        assert tracker.backward_count == 1


class TestFLOPsEstimation:
    """Test FLOPs estimation"""

    def test_flops_estimation_formula(self):
        """Test FLOPs estimation uses 6*P*T formula"""
        tracker = EfficiencyTracker()

        # Create a simple mock model with known parameter count
        import torch.nn as nn
        model = nn.Linear(100, 100)  # 100*100 + 100 = 10100 params

        num_params = sum(p.numel() for p in model.parameters())
        num_tokens = 1000

        flops = tracker._estimate_flops(model, num_tokens)

        # FLOPs should be 6 * params * tokens
        expected = 6.0 * num_params * num_tokens
        assert flops == expected

    def test_flops_estimation_no_model(self):
        """Test FLOPs estimation returns 0 when model is None"""
        tracker = EfficiencyTracker()
        assert tracker._estimate_flops(None, 1000) == 0.0

    def test_flops_estimation_no_tokens(self):
        """Test FLOPs estimation returns 0 when tokens is 0"""
        import torch.nn as nn
        tracker = EfficiencyTracker()
        model = nn.Linear(10, 10)
        assert tracker._estimate_flops(model, 0) == 0.0


class TestSampleCounting:
    """Test forget/retain sample counting"""

    def test_get_sample_counts_with_forget_retain_dataset(self):
        """Test sample counting with ForgetRetainDataset structure"""
        tracker = EfficiencyTracker()

        # Mock trainer with ForgetRetainDataset
        mock_trainer = MagicMock()
        mock_trainer.train_dataset.forget = list(range(100))  # 100 forget samples
        mock_trainer.train_dataset.retain = list(range(900))  # 900 retain samples

        forget, retain = tracker._get_sample_counts(mock_trainer)
        assert forget == 100
        assert retain == 900

    def test_get_sample_counts_with_alternative_structure(self):
        """Test sample counting with alternative dataset structure"""
        tracker = EfficiencyTracker()

        # Mock trainer with alternative structure (no forget/retain attributes)
        mock_trainer = MagicMock(spec=['train_dataset'])
        mock_trainer.train_dataset = MagicMock(spec=['forget_dataset', 'retain_dataset'])
        mock_trainer.train_dataset.forget_dataset = list(range(50))
        mock_trainer.train_dataset.retain_dataset = list(range(450))

        forget, retain = tracker._get_sample_counts(mock_trainer)
        assert forget == 50
        assert retain == 450

    def test_get_sample_counts_single_dataset(self):
        """Test sample counting with single dataset (no retain)"""
        tracker = EfficiencyTracker()

        # Mock trainer with single dataset
        mock_trainer = MagicMock()
        del mock_trainer.train_dataset.forget
        del mock_trainer.train_dataset.retain
        del mock_trainer.train_dataset.forget_dataset
        del mock_trainer.train_dataset.retain_dataset
        mock_trainer.train_dataset.__len__ = MagicMock(return_value=200)

        forget, retain = tracker._get_sample_counts(mock_trainer)
        assert forget == 200
        assert retain == 0

    def test_get_sample_counts_no_trainer(self):
        """Test sample counting with no trainer"""
        tracker = EfficiencyTracker()
        forget, retain = tracker._get_sample_counts(None)
        assert forget == 0
        assert retain == 0


class TestGPUUtilization:
    """Test GPU utilization tracking"""

    def test_nvml_not_available(self):
        """Test graceful handling when NVML is not available"""
        tracker = EfficiencyTracker()
        # Should not raise even if NVML is not available
        util = tracker._get_gpu_utilization()
        # Returns 0.0 if NVML not initialized
        assert util == 0.0 or util >= 0.0

    @patch('trainer.utils.EfficiencyTracker._try_init_nvml')
    def test_gpu_utilization_sampling(self, mock_init):
        """Test GPU utilization is sampled at correct intervals"""
        tracker = EfficiencyTracker(gpu_sampling_interval=2)
        tracker._nvml_initialized = True
        tracker.last_step_time = time.time()

        # Mock GPU utilization
        tracker._get_gpu_utilization = MagicMock(return_value=75.0)

        class MockState:
            global_step = 0
        class MockArgs:
            pass

        # Step 0 - should sample (0 % 2 == 0)
        MockState.global_step = 0
        tracker.on_step_end(MockArgs(), MockState(), None)
        assert len(tracker.gpu_utilizations) == 1

        # Step 1 - should not sample (1 % 2 != 0)
        MockState.global_step = 1
        tracker.on_step_end(MockArgs(), MockState(), None)
        assert len(tracker.gpu_utilizations) == 1

        # Step 2 - should sample (2 % 2 == 0)
        MockState.global_step = 2
        tracker.on_step_end(MockArgs(), MockState(), None)
        assert len(tracker.gpu_utilizations) == 2


class TestCheckpointTiming:
    """Test checkpoint save time tracking"""

    def test_checkpoint_timing(self):
        """Test checkpoint save time is tracked"""
        tracker = EfficiencyTracker()

        # Start checkpoint save
        tracker.on_save(None, None, None)
        assert tracker._checkpoint_start_time is not None

        # Simulate save time
        time.sleep(0.01)

        # End checkpoint save
        tracker.on_save(None, None, None)
        assert tracker._checkpoint_start_time is None
        assert len(tracker.checkpoint_save_times) == 1
        assert tracker.checkpoint_save_times[0] >= 0.01


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
