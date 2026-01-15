"""
Tests for MIA (Membership Inference Attack) metrics.
Run with: pytest tests/test_mia_attacks.py -v
"""

import pytest
import torch
import numpy as np

from evals.metrics.mia.min_k import MinKProbAttack
from evals.metrics.mia.min_k_plus_plus import MinKPlusPlusAttack


def is_numeric(value):
    """Check if value is a numeric type (float or numpy float)"""
    return isinstance(value, (float, np.floating))


class TestMinKProbAttack:
    """Test MinKProbAttack score computation"""

    def test_compute_score_basic(self):
        """Test basic score computation with float32 tensor"""
        attack = MinKProbAttack.__new__(MinKProbAttack)
        attack.k = 0.2

        # Create sample log probabilities
        log_probs = torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0], dtype=torch.float32)
        score = attack.compute_score(log_probs)

        # Should return negative mean of bottom k% (20% = 1 element)
        assert is_numeric(score)
        assert score > 0  # Negative of negative values

    def test_compute_score_bfloat16(self):
        """Test score computation with bfloat16 tensor"""
        attack = MinKProbAttack.__new__(MinKProbAttack)
        attack.k = 0.2

        # Create bfloat16 tensor
        log_probs = torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0], dtype=torch.bfloat16)
        score = attack.compute_score(log_probs)

        assert is_numeric(score)
        assert not np.isnan(score)

    def test_compute_score_empty(self):
        """Test score computation with empty tensor"""
        attack = MinKProbAttack.__new__(MinKProbAttack)
        attack.k = 0.2

        log_probs = torch.tensor([], dtype=torch.float32)
        score = attack.compute_score(log_probs)

        assert score == 0

    def test_compute_score_single_element(self):
        """Test score computation with single element"""
        attack = MinKProbAttack.__new__(MinKProbAttack)
        attack.k = 0.2

        log_probs = torch.tensor([-2.5], dtype=torch.float32)
        score = attack.compute_score(log_probs)

        assert is_numeric(score)
        assert score == pytest.approx(2.5)  # -(-2.5) = 2.5


class TestMinKPlusPlusAttack:
    """Test MinKPlusPlusAttack score computation"""

    def test_compute_score_basic(self):
        """Test basic score computation with float32 tensors"""
        attack = MinKPlusPlusAttack.__new__(MinKPlusPlusAttack)
        attack.k = 0.2

        # Create sample stats
        seq_len = 10
        vocab_size = 100
        vocab_log_probs = torch.randn(seq_len, vocab_size, dtype=torch.float32)
        vocab_log_probs = torch.log_softmax(vocab_log_probs, dim=-1)
        token_log_probs = torch.randn(seq_len, dtype=torch.float32)

        sample_stats = {
            "vocab_log_probs": vocab_log_probs,
            "token_log_probs": token_log_probs,
        }

        score = attack.compute_score(sample_stats)

        assert is_numeric(score)
        assert not np.isnan(score)

    def test_compute_score_bfloat16(self):
        """Test score computation with bfloat16 tensors - regression test for dtype bug"""
        attack = MinKPlusPlusAttack.__new__(MinKPlusPlusAttack)
        attack.k = 0.2

        # Create bfloat16 tensors (this was causing the original bug)
        seq_len = 10
        vocab_size = 100
        vocab_log_probs = torch.randn(seq_len, vocab_size, dtype=torch.bfloat16)
        vocab_log_probs = torch.log_softmax(vocab_log_probs, dim=-1)
        token_log_probs = torch.randn(seq_len, dtype=torch.bfloat16)

        sample_stats = {
            "vocab_log_probs": vocab_log_probs,
            "token_log_probs": token_log_probs,
        }

        # This should not raise TypeError: Got unsupported ScalarType BFloat16
        score = attack.compute_score(sample_stats)

        assert is_numeric(score)
        assert not np.isnan(score)

    def test_compute_score_float16(self):
        """Test score computation with float16 tensors"""
        attack = MinKPlusPlusAttack.__new__(MinKPlusPlusAttack)
        attack.k = 0.2

        seq_len = 10
        vocab_size = 100
        vocab_log_probs = torch.randn(seq_len, vocab_size, dtype=torch.float16)
        vocab_log_probs = torch.log_softmax(vocab_log_probs, dim=-1)
        token_log_probs = torch.randn(seq_len, dtype=torch.float16)

        sample_stats = {
            "vocab_log_probs": vocab_log_probs,
            "token_log_probs": token_log_probs,
        }

        score = attack.compute_score(sample_stats)

        assert is_numeric(score)
        assert not np.isnan(score)

    def test_compute_score_empty(self):
        """Test score computation with empty tensors"""
        attack = MinKPlusPlusAttack.__new__(MinKPlusPlusAttack)
        attack.k = 0.2

        sample_stats = {
            "vocab_log_probs": torch.tensor([]).reshape(0, 100),
            "token_log_probs": torch.tensor([]),
        }

        score = attack.compute_score(sample_stats)
        assert score == 0

    def test_compute_score_numerical_stability(self):
        """Test that sigma clamping prevents numerical issues"""
        attack = MinKPlusPlusAttack.__new__(MinKPlusPlusAttack)
        attack.k = 0.2

        # Create uniform distribution (sigma would be very small)
        seq_len = 5
        vocab_size = 10
        # Uniform log probs
        vocab_log_probs = torch.full(
            (seq_len, vocab_size), -np.log(vocab_size), dtype=torch.float32
        )
        token_log_probs = torch.full((seq_len,), -np.log(vocab_size), dtype=torch.float32)

        sample_stats = {
            "vocab_log_probs": vocab_log_probs,
            "token_log_probs": token_log_probs,
        }

        score = attack.compute_score(sample_stats)

        assert is_numeric(score)
        assert not np.isnan(score)
        assert not np.isinf(score)


class TestDtypeCompatibility:
    """Test dtype compatibility across all MIA attacks"""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_min_k_dtype_compatibility(self, dtype):
        """Test MinKProbAttack works with various dtypes"""
        attack = MinKProbAttack.__new__(MinKProbAttack)
        attack.k = 0.2

        log_probs = torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0], dtype=dtype)
        score = attack.compute_score(log_probs)

        assert is_numeric(score)
        assert not np.isnan(score)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_min_k_plus_plus_dtype_compatibility(self, dtype):
        """Test MinKPlusPlusAttack works with various dtypes"""
        attack = MinKPlusPlusAttack.__new__(MinKPlusPlusAttack)
        attack.k = 0.2

        seq_len = 10
        vocab_size = 100
        vocab_log_probs = torch.randn(seq_len, vocab_size, dtype=dtype)
        vocab_log_probs = torch.log_softmax(vocab_log_probs.float(), dim=-1).to(dtype)
        token_log_probs = torch.randn(seq_len, dtype=dtype)

        sample_stats = {
            "vocab_log_probs": vocab_log_probs,
            "token_log_probs": token_log_probs,
        }

        score = attack.compute_score(sample_stats)

        assert is_numeric(score)
        assert not np.isnan(score)
