"""
Comprehensive tests for TrainingLogger.

Tests cover:
- Initialization with different modes and options
- Step registration (batch and sample modes)
- Memory management and pruning
- Save/load functionality
- BatchReconstructor
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile

from trainer.training_logger import TrainingLogger, BatchReconstructor


# ============================================================================
# Test Fixtures
# ============================================================================

# simple_model, batch_data, and temp_log_dir fixtures are defined in conftest.py


# ============================================================================
# Test TrainingLogger Initialization
# ============================================================================

class TestTrainingLoggerInit:
    """Test TrainingLogger initialization"""

    def test_basic_initialization(self, temp_log_dir):
        """Test basic initialization"""
        logger = TrainingLogger(log_dir=str(temp_log_dir))

        assert logger.log_dir == temp_log_dir
        assert logger.max_steps == 1000
        assert logger.mode == "batch"
        assert logger.save_interval == 100
        assert not logger.save_batch_data
        assert not logger.save_indices_only
        assert not logger.save_rng_state
        assert not logger.compute_diag_h

    def test_custom_initialization(self, temp_log_dir):
        """Test initialization with custom parameters"""
        logger = TrainingLogger(
            log_dir=str(temp_log_dir),
            max_steps=500,
            mode="sample",
            save_interval=50,
            save_batch_data=True,
            save_indices_only=True,
            save_rng_state=True,
            compute_diag_h=True,
            batch_size_at_training=32
        )

        assert logger.max_steps == 500
        assert logger.mode == "sample"
        assert logger.save_interval == 50
        assert logger.save_batch_data
        assert logger.save_indices_only
        assert logger.save_rng_state
        assert logger.compute_diag_h
        assert logger.batch_size_at_training == 32

    def test_directory_creation(self):
        """Test that log directory is created"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "new_logs"
            assert not log_dir.exists()

            TrainingLogger(log_dir=str(log_dir))
            assert log_dir.exists()


# ============================================================================
# Test Step Registration
# ============================================================================

class TestStepRegistration:
    """Test step registration functionality"""

    def test_register_step_basic(self, temp_log_dir, simple_model):
        """Test basic step registration"""
        logger = TrainingLogger(log_dir=str(temp_log_dir))

        u = torch.randn(sum(p.numel() for p in simple_model.parameters()))
        logger.register_step(
            step_id=1,
            batch_id="batch_1",
            eta=0.01,
            u=u
        )

        assert logger.current_step == 1
        record = logger.step_log.get(1)
        assert record is not None
        assert record.step_id == 1
        assert record.eta == 0.01
        assert torch.equal(record.u, u)

    def test_register_step_with_model(self, temp_log_dir, simple_model):
        """Test step registration with model (computes u automatically)"""
        logger = TrainingLogger(log_dir=str(temp_log_dir))

        # First step - initialize prev_params
        logger.register_step(
            step_id=1,
            batch_id="batch_1",
            eta=0.01,
            model=simple_model
        )

        # Modify model parameters
        with torch.no_grad():
            for p in simple_model.parameters():
                p.add_(torch.randn_like(p) * 0.01)

        # Second step - should compute u
        logger.register_step(
            step_id=2,
            batch_id="batch_2",
            eta=0.01,
            model=simple_model
        )

        record = logger.step_log.get(2)
        assert record.u is not None

    def test_register_step_with_gbar(self, temp_log_dir):
        """Test step registration with gradient"""
        logger = TrainingLogger(log_dir=str(temp_log_dir))

        gbar = torch.randn(100)
        logger.register_step(
            step_id=1,
            batch_id="batch_1",
            eta=0.01,
            gbar=gbar
        )

        record = logger.step_log.get(1)
        assert torch.equal(record.gbar, gbar)

    def test_register_step_with_batch_data(self, temp_log_dir, batch_data):
        """Test step registration with batch data"""
        logger = TrainingLogger(
            log_dir=str(temp_log_dir),
            save_batch_data=True
        )

        logger.register_step(
            step_id=1,
            batch_id="batch_1",
            eta=0.01,
            batch_data=batch_data
        )

        record = logger.step_log.get(1)
        assert record.batch_data is not None
        assert 'input_ids' in record.batch_data

    def test_register_step_with_sample_indices(self, temp_log_dir):
        """Test step registration with sample indices"""
        logger = TrainingLogger(
            log_dir=str(temp_log_dir),
            save_indices_only=True
        )

        sample_indices = [0, 1, 2, 3]
        logger.register_step(
            step_id=1,
            batch_id="batch_1",
            eta=0.01,
            sample_indices=sample_indices
        )

        assert 1 in logger.sample_indices_per_step
        assert logger.sample_indices_per_step[1] == sample_indices

    def test_register_multiple_steps(self, temp_log_dir):
        """Test registering multiple steps"""
        # Use save_interval > 10 to avoid buffer clearing during registration
        # Start from step 1 to avoid triggering save at step 0 (0 % N == 0)
        logger = TrainingLogger(log_dir=str(temp_log_dir), save_interval=20)

        for i in range(1, 11):  # Steps 1-10
            u = torch.randn(100)
            logger.register_step(
                step_id=i,
                batch_id=f"batch_{i}",
                eta=0.01,
                u=u
            )

        assert logger.current_step == 10
        assert len(logger.step_log.buffer) == 10


# ============================================================================
# Test Memory Management
# ============================================================================

class TestMemoryManagement:
    """Test memory management and pruning"""

    def test_prune_old_entries(self, temp_log_dir):
        """Test pruning of old entries"""
        logger = TrainingLogger(
            log_dir=str(temp_log_dir),
            save_interval=10,
            save_indices_only=True
        )

        # Add many entries
        for i in range(50):
            logger.register_step(
                step_id=i,
                batch_id=f"batch_{i}",
                eta=0.01,
                sample_indices=[i]
            )

        # Manually trigger pruning
        logger._prune_old_entries()

        # Should keep at most 2x save_interval entries
        max_entries = logger.save_interval * 2
        assert len(logger.sample_indices_per_step) <= max_entries

    def test_no_pruning_when_save_interval_zero(self, temp_log_dir):
        """Test that pruning doesn't happen when save_interval=0"""
        logger = TrainingLogger(
            log_dir=str(temp_log_dir),
            save_interval=0,
            save_indices_only=True
        )

        for i in range(50):
            logger.register_step(
                step_id=i,
                batch_id=f"batch_{i}",
                eta=0.01,
                sample_indices=[i]
            )

        logger._prune_old_entries()

        # All entries should still be there
        assert len(logger.sample_indices_per_step) == 50


# ============================================================================
# Test Save/Load Functionality
# ============================================================================

class TestSaveLoad:
    """Test save and load functionality"""

    def test_save_to_disk(self, temp_log_dir):
        """Test saving to disk"""
        logger = TrainingLogger(
            log_dir=str(temp_log_dir),
            save_interval=5
        )

        # Register steps
        for i in range(10):
            u = torch.randn(100)
            logger.register_step(
                step_id=i,
                batch_id=f"batch_{i}",
                eta=0.01,
                u=u
            )

        # Save to disk
        logger.save_to_disk()

        # Check files exist - incremental save uses chunk files
        chunk_files = list(temp_log_dir.glob("step_records_chunk_*.pkl"))
        assert len(chunk_files) > 0, "No chunk files found"
        assert (temp_log_dir / "meta.json").exists()

    def test_load_from_disk(self, temp_log_dir):
        """Test loading from disk with tensors"""
        # Create and save logger (use save_interval > 5 to avoid buffer clearing)
        logger1 = TrainingLogger(log_dir=str(temp_log_dir), save_interval=10)

        for i in range(5):
            u = torch.randn(100)
            logger1.register_step(
                step_id=i,
                batch_id=f"batch_{i}",
                eta=0.01,
                u=u
            )

        logger1.save_to_disk()

        # Create new logger and load with tensors (for verification)
        logger2 = TrainingLogger(log_dir=str(temp_log_dir))
        logger2.load_from_disk(load_tensors=True)

        # Verify data was loaded when load_tensors=True
        assert len(logger2.step_log.buffer) == 5
        for i in range(5):
            assert logger2.step_log.get(i) is not None

    def test_load_from_disk_metadata_only(self, temp_log_dir):
        """Test loading from disk without tensors (memory-efficient resume)"""
        # Create and save logger
        logger1 = TrainingLogger(log_dir=str(temp_log_dir), save_interval=10)

        for i in range(5):
            u = torch.randn(100)
            logger1.register_step(
                step_id=i,
                batch_id=f"batch_{i}",
                eta=0.01,
                u=u
            )

        logger1.save_to_disk()

        # Create new logger and load without tensors (default, for resume)
        logger2 = TrainingLogger(log_dir=str(temp_log_dir))
        logger2.load_from_disk()  # load_tensors=False by default

        # Verify metadata was restored but buffer is empty (memory efficient)
        assert logger2.current_step == 4
        assert logger2._last_saved_step_id == 4
        assert len(logger2.step_log.buffer) == 0  # No tensors loaded

    def test_load_nonexistent_directory(self):
        """Test loading from nonexistent directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent = Path(tmpdir) / "nonexistent"
            logger = TrainingLogger(log_dir=str(nonexistent))

            # Should not raise error, just log warning
            logger.load_from_disk()

    def test_save_with_sample_indices(self, temp_log_dir):
        """Test saving with sample indices"""
        logger = TrainingLogger(
            log_dir=str(temp_log_dir),
            save_indices_only=True,
            save_interval=10  # Save at step 0 only
        )

        for i in range(5):
            logger.register_step(
                step_id=i,
                batch_id=f"batch_{i}",
                eta=0.01,
                sample_indices=[i, i+1, i+2]
            )

        logger.save_to_disk()

        # Load with tensors to verify the saved data
        logger2 = TrainingLogger(log_dir=str(temp_log_dir))
        logger2.load_from_disk(load_tensors=True)

        # After save_to_disk(), indices are cleared (memory-efficient behavior)
        # When loading with load_tensors=True, we reload them
        # The final save includes all remaining indices (steps 1-4 after step 0 auto-save)
        assert len(logger2.sample_indices_per_step) == 4


# ============================================================================
# Test BatchReconstructor
# ============================================================================

class TestBatchReconstructor:
    """Test BatchReconstructor functionality"""

    def test_initialization(self, temp_log_dir):
        """Test BatchReconstructor initialization"""
        # Create mock training logger
        logger = TrainingLogger(log_dir=str(temp_log_dir))

        # Create mock dataset and collator
        dataset = list(range(100))
        collator = lambda x: {'data': torch.tensor(x)}

        reconstructor = BatchReconstructor(
            training_logger=logger,
            dataset=dataset,
            data_collator=collator
        )

        assert reconstructor.dataset == dataset
        assert reconstructor.data_collator == collator

    def test_get_batch_for_step_with_indices(self, temp_log_dir):
        """Test batch reconstruction with indices"""
        # Create training logger with sample indices
        logger = TrainingLogger(
            log_dir=str(temp_log_dir),
            save_indices_only=True
        )

        # Register step 10 so it exists in the log
        logger.register_step(
            step_id=10,
            batch_id="batch_10",
            eta=0.01,
            sample_indices=[0, 1, 2, 3]
        )

        dataset = [{'id': i, 'value': i*2} for i in range(100)]
        collator = lambda batch: {
            'ids': torch.tensor([item['id'] for item in batch]),
            'values': torch.tensor([item['value'] for item in batch])
        }

        reconstructor = BatchReconstructor(
            training_logger=logger,
            dataset=dataset,
            data_collator=collator
        )

        batch = reconstructor.get_batch_for_step(10)

        assert batch is not None
        assert 'ids' in batch
        assert 'values' in batch
        assert batch['ids'].tolist() == [0, 1, 2, 3]
        assert batch['values'].tolist() == [0, 2, 4, 6]

    def test_get_batch_missing_indices(self, temp_log_dir):
        """Test batch reconstruction with missing indices"""
        logger = TrainingLogger(log_dir=str(temp_log_dir))
        dataset = list(range(100))
        collator = lambda x: {'data': torch.tensor(x)}

        reconstructor = BatchReconstructor(
            training_logger=logger,
            dataset=dataset,
            data_collator=collator
        )

        # No indices set
        batch = reconstructor.get_batch_for_step(10)
        assert batch is None


# ============================================================================
# Test Integration
# ============================================================================

class TestIntegration:
    """Test integration scenarios"""

    def test_full_workflow(self, temp_log_dir, simple_model):
        """Test complete workflow: register, save, load"""
        # Phase 1: Training with logging
        logger1 = TrainingLogger(
            log_dir=str(temp_log_dir),
            save_interval=5,
            save_indices_only=True
        )

        for i in range(10):
            logger1.register_step(
                step_id=i,
                batch_id=f"batch_{i}",
                eta=0.01,
                model=simple_model,
                sample_indices=[i*4, i*4+1, i*4+2, i*4+3]
            )

            # Simulate parameter updates
            with torch.no_grad():
                for p in simple_model.parameters():
                    p.add_(torch.randn_like(p) * 0.001)

        logger1.save_to_disk()

        # Phase 2: Load for unlearning (with load_tensors=True to access data)
        logger2 = TrainingLogger(log_dir=str(temp_log_dir))
        logger2.load_from_disk(load_tensors=True)

        # Verify loaded data
        assert len(logger2.step_log.buffer) == 10

        # With save_interval=5, automatic saves happen at steps 0 and 5
        # After final save_to_disk(), indices from steps 6-9 remain (4 indices)
        # because they were added after the last automatic save at step 5
        assert len(logger2.sample_indices_per_step) == 4

        # Verify specific step
        record = logger2.step_log.get(5)
        assert record is not None
        assert record.step_id == 5
        assert record.eta == 0.01


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
