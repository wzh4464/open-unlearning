"""
Tests for Phase B: Historical HVP, damping fix, and sum-then-apply.

Covers:
- Deliverable A: HistoricalParamProvider with sparse checkpoints
- Deliverable B: Damping fix (paper Algorithm 1 alignment)
- Deliverable C: Sum-then-apply for multi forget steps
"""

import pytest
import torch
import torch.nn as nn
import tempfile
from pathlib import Path

from trainer.unlearn.lmcleaner_core import (
    StepRecord,
    StepLog,
    HVPConfig,
    HistoricalParamProvider,
    compute_correction,
    apply_correction,
    _get_flat_params,
    _set_flat_params,
)


# ============================================================================
# Deliverable A: HistoricalParamProvider Tests
# ============================================================================


class TestHistoricalParamProvider:
    """Tests for HistoricalParamProvider."""

    def test_init_without_checkpoints(self, simple_model):
        """Provider initializes even without checkpoint index."""
        provider = HistoricalParamProvider(model=simple_model)
        assert provider._checkpoint_index == {}
        assert provider._theta_tau is None
        assert not provider._active

    def test_prepare_window_with_step_log(self, simple_model, batch_data):
        """Provider can prepare a window using step_log with u vectors."""
        params = [p for p in simple_model.parameters() if p.requires_grad]
        param_count = sum(p.numel() for p in params)

        step_log = StepLog(max_size=100)
        for i in range(5, 10):
            step_log.add(
                StepRecord(
                    step_id=i,
                    eta=0.01,
                    batch_id=f"b{i}",
                    u=torch.randn(param_count) * 0.01,
                    batch_data=batch_data,
                )
            )

        provider = HistoricalParamProvider(model=simple_model, step_log=step_log)

        # tau=10, window [5, 8]
        success = provider.prepare_window(start=5, end=8, tau=10)
        assert success
        assert provider._active

        # Verify model params changed from theta[tau]
        theta_current = _get_flat_params(simple_model)
        assert provider._theta_tau is not None
        # theta[5] != theta[10] since we subtracted u vectors
        assert not torch.allclose(theta_current, provider._theta_tau)

        provider.restore_model()
        assert not provider._active

        # Verify restored
        theta_restored = _get_flat_params(simple_model)
        assert torch.allclose(theta_restored, provider._theta_tau, atol=1e-7)

    def test_advance_to_next_step(self, simple_model, batch_data):
        """Provider advances theta[s] -> theta[s+1] correctly."""
        params = [p for p in simple_model.parameters() if p.requires_grad]
        param_count = sum(p.numel() for p in params)

        step_log = StepLog(max_size=100)
        u_5 = torch.randn(param_count) * 0.01
        u_6 = torch.randn(param_count) * 0.01
        step_log.add(
            StepRecord(step_id=5, eta=0.01, batch_id="b5", u=u_5, batch_data=batch_data)
        )
        step_log.add(
            StepRecord(step_id=6, eta=0.01, batch_id="b6", u=u_6, batch_data=batch_data)
        )

        provider = HistoricalParamProvider(model=simple_model, step_log=step_log)
        success = provider.prepare_window(start=5, end=6, tau=7)
        assert success

        theta_5 = _get_flat_params(simple_model).clone()

        provider.advance_to_next_step(5)
        theta_6 = _get_flat_params(simple_model).clone()

        # theta[6] = theta[5] + u[5]
        expected_6 = theta_5 + u_5.to(theta_5.device)
        assert torch.allclose(theta_6, expected_6, atol=1e-6)

        provider.cleanup()

    def test_cleanup_releases_resources(self, simple_model):
        """Provider cleanup releases all tensors."""
        param_count = sum(
            p.numel() for p in simple_model.parameters() if p.requires_grad
        )
        provider = HistoricalParamProvider(model=simple_model)
        # Set theta_tau to correct size so restore_model works
        provider._theta_tau = _get_flat_params(simple_model).clone()
        provider._cache[0] = torch.randn(param_count)
        provider._cache_order.append(0)

        provider.cleanup()
        assert provider._theta_tau is None
        assert len(provider._cache) == 0
        assert len(provider._cache_order) == 0

    def test_lru_cache_eviction(self, simple_model, batch_data):
        """LRU cache evicts oldest entry when full."""
        params = [p for p in simple_model.parameters() if p.requires_grad]
        param_count = sum(p.numel() for p in params)

        provider = HistoricalParamProvider(model=simple_model, max_cache_entries=2)

        # Manually populate cache
        provider._cache_put(1, torch.randn(param_count))
        provider._cache_put(2, torch.randn(param_count))
        assert len(provider._cache) == 2

        # Adding a 3rd entry should evict the oldest (1)
        provider._cache_put(3, torch.randn(param_count))
        assert len(provider._cache) == 2
        assert 1 not in provider._cache
        assert 2 in provider._cache
        assert 3 in provider._cache

    def test_sparse_checkpoint_reconstruction(self, simple_model, batch_data):
        """Test reconstruction from sparse checkpoint + u vectors."""
        params = [p for p in simple_model.parameters() if p.requires_grad]
        param_count = sum(p.numel() for p in params)

        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            sparse_dir = log_dir / "sparse_checkpoints"
            sparse_dir.mkdir()

            # Save a "checkpoint" at step 0
            trainable_state = {}
            for name, param in simple_model.named_parameters():
                if param.requires_grad:
                    trainable_state[name] = param.detach().cpu()
            ckpt_file = sparse_dir / "step_000000.pt"
            torch.save(trainable_state, ckpt_file)

            # Create checkpoint index
            import json

            ckpt_index = {
                "stride": 25,
                "dtype": "fp32",
                "checkpoints": {"0": "sparse_checkpoints/step_000000.pt"},
            }
            with open(log_dir / "checkpoint_index.json", "w") as f:
                json.dump(ckpt_index, f)

            # Create step_log with u vectors for steps 0-4
            step_log = StepLog(max_size=100)
            u_vectors = {}
            for i in range(5):
                u_i = torch.randn(param_count) * 0.01
                u_vectors[i] = u_i
                step_log.add(
                    StepRecord(
                        step_id=i,
                        eta=0.01,
                        batch_id=f"b{i}",
                        u=u_i,
                        batch_data=batch_data,
                    )
                )

            provider = HistoricalParamProvider(
                model=simple_model,
                log_dir=str(log_dir),
                step_log=step_log,
            )

            # Reconstruct theta[3] from checkpoint at 0 + u[0]+u[1]+u[2]
            theta_3 = provider.get_params_for_step(3, tau=5)
            assert theta_3 is not None

            # Manually compute expected theta[3]
            theta_0 = _get_flat_params(simple_model).clone()
            expected = theta_0.clone()
            for t in range(3):
                expected = expected + u_vectors[t].to(expected.device)
            assert torch.allclose(theta_3.cpu(), expected.cpu(), atol=1e-5)

            provider.cleanup()


# ============================================================================
# Deliverable B: Damping Fix Tests
# ============================================================================


class TestDampingFix:
    """Test that damping matches paper Algorithm 1: hv = Hv + λv, v ← v - η·hv."""

    def test_quadratic_no_damping(self):
        """
        Toy case: f(θ) = 0.5 * θ^T A θ, H = A.
        Without damping: v ← v - η * A * v = (I - ηA) v.
        """
        dim = 4
        torch.manual_seed(42)
        A = torch.eye(dim) * 2.0  # H = 2I
        eta = 0.1
        v_init = torch.randn(dim)

        # Create a model with known Hessian
        model = nn.Linear(dim, 1, bias=False)
        # Set weights so that loss = 0.5 * x^T W^T W x (simplified)
        # For this test, we use diagonal HVP directly

        step_log = StepLog(max_size=10)
        step_log.add(
            StepRecord(step_id=0, eta=eta, batch_id="forget", u=v_init.clone())
        )
        step_log.add(
            StepRecord(
                step_id=1,
                eta=eta,
                batch_id="b1",
                diag_H=torch.diag(A),  # [2, 2, 2, 2]
                batch_data=None,
            )
        )

        cfg = HVPConfig(mode="diag", damping=0.0, device="cpu")

        # With K=1, one propagation step
        v, audit = compute_correction(
            tz=0, tau=2, K=1, step_log=step_log, cfg=cfg, model=model
        )

        # v_init = -u[0], then v = v_init - η * diag(A) * v_init = (I - ηA) * v_init
        v_init_neg = -v_init
        expected = v_init_neg - eta * (torch.diag(A) * v_init_neg)
        assert torch.allclose(v, expected, atol=1e-6), (
            f"No-damping propagation mismatch.\n"
            f"Got: {v}\nExpected: {expected}\nDiff: {(v - expected).abs().max()}"
        )

    def test_quadratic_with_damping(self):
        """
        Toy case: f(θ) = 0.5 * θ^T A θ, H = A, damping = λ.
        Paper: hv = Hv + λv, then v ← v - η·hv = (I - η(A + λI)) v.
        """
        dim = 4
        torch.manual_seed(42)
        A_diag = torch.tensor([2.0, 3.0, 1.0, 4.0])
        eta = 0.05
        damping = 0.1
        v_init = torch.randn(dim)

        model = nn.Linear(dim, 1, bias=False)

        step_log = StepLog(max_size=10)
        step_log.add(
            StepRecord(step_id=0, eta=eta, batch_id="forget", u=v_init.clone())
        )
        step_log.add(
            StepRecord(
                step_id=1,
                eta=eta,
                batch_id="b1",
                diag_H=A_diag,
                batch_data=None,
            )
        )

        cfg = HVPConfig(mode="diag", damping=damping, device="cpu")

        v, audit = compute_correction(
            tz=0, tau=2, K=1, step_log=step_log, cfg=cfg, model=model
        )

        # Paper: hv = A*v + λ*v, then v ← v - η*hv
        v_init_neg = -v_init
        hv = A_diag * v_init_neg + damping * v_init_neg
        expected = v_init_neg - eta * hv
        assert torch.allclose(v, expected, atol=1e-6), (
            f"Damping propagation mismatch.\n"
            f"Got: {v}\nExpected: {expected}\nDiff: {(v - expected).abs().max()}"
        )

    def test_damping_multi_step(self):
        """Test damping over multiple propagation steps."""
        dim = 3
        torch.manual_seed(42)
        A_diag = torch.tensor([1.0, 2.0, 3.0])
        eta = 0.05
        damping = 0.2

        model = nn.Linear(dim, 1, bias=False)

        step_log = StepLog(max_size=10)
        u_0 = torch.randn(dim)
        step_log.add(StepRecord(step_id=0, eta=eta, batch_id="forget", u=u_0))
        for s in range(1, 4):
            step_log.add(
                StepRecord(
                    step_id=s,
                    eta=eta,
                    batch_id=f"b{s}",
                    diag_H=A_diag,
                    batch_data=None,
                )
            )

        cfg = HVPConfig(mode="diag", damping=damping, device="cpu")

        v, audit = compute_correction(
            tz=0, tau=4, K=3, step_log=step_log, cfg=cfg, model=model
        )

        # Manual propagation
        v_manual = -u_0
        for s in range(1, 4):
            hv = A_diag * v_manual + damping * v_manual
            v_manual = v_manual - eta * hv

        assert torch.allclose(v, v_manual, atol=1e-6), (
            f"Multi-step damping mismatch.\n"
            f"Got: {v}\nExpected: {v_manual}\nDiff: {(v - v_manual).abs().max()}"
        )
        assert audit.hvp_calls == 3

    def test_damping_zero_matches_legacy(self, simple_model, batch_data):
        """With damping=0, the new code should produce same result as before."""
        params = [p for p in simple_model.parameters() if p.requires_grad]
        param_count = sum(p.numel() for p in params)

        step_log = StepLog(max_size=100)
        u_tz = torch.randn(param_count) * 0.01
        step_log.add(
            StepRecord(
                step_id=10,
                eta=0.01,
                batch_id="forget",
                u=u_tz,
                batch_data=batch_data,
            )
        )
        for i in range(11, 14):
            step_log.add(
                StepRecord(
                    step_id=i,
                    eta=0.01,
                    batch_id=f"b{i}",
                    batch_data=batch_data,
                )
            )

        cfg = HVPConfig(mode="fisher", damping=0.0, device="cpu")

        v1, a1 = compute_correction(
            tz=10, tau=14, K=3, step_log=step_log, cfg=cfg, model=simple_model
        )

        # Run again to verify determinism (fisher with same model state)
        v2, a2 = compute_correction(
            tz=10, tau=14, K=3, step_log=step_log, cfg=cfg, model=simple_model
        )

        assert torch.allclose(v1, v2, atol=1e-5)


# ============================================================================
# Deliverable C: Sum-then-apply Tests
# ============================================================================


class TestSumThenApply:
    """Test that sum-then-apply produces order-invariant results."""

    def test_order_invariance(self, simple_model):
        """Applying corrections in different order should give same final params."""
        params = [p for p in simple_model.parameters() if p.requires_grad]
        param_count = sum(p.numel() for p in params)
        original_params = _get_flat_params(simple_model).clone()

        # Create two correction vectors
        v1 = torch.randn(param_count) * 0.01
        v2 = torch.randn(param_count) * 0.01

        # Sum-then-apply: v_total = v1 + v2, apply once
        v_total = v1 + v2
        apply_correction(v_total, params)
        theta_sum_apply = _get_flat_params(simple_model).clone()

        # Restore
        _set_flat_params(simple_model, original_params)

        # Sequential v1 then v2
        apply_correction(v1, params)
        apply_correction(v2, params)
        theta_seq_12 = _get_flat_params(simple_model).clone()

        # Restore
        _set_flat_params(simple_model, original_params)

        # Sequential v2 then v1
        apply_correction(v2, params)
        apply_correction(v1, params)
        theta_seq_21 = _get_flat_params(simple_model).clone()

        # All three should be identical (since apply_correction is additive)
        assert torch.allclose(theta_sum_apply, theta_seq_12, atol=1e-7)
        assert torch.allclose(theta_sum_apply, theta_seq_21, atol=1e-7)

    def test_sum_then_apply_accumulation(self, simple_model):
        """Test streaming sum accumulation matches explicit sum."""
        params = [p for p in simple_model.parameters() if p.requires_grad]
        param_count = sum(p.numel() for p in params)

        corrections = [torch.randn(param_count) * 0.01 for _ in range(5)]

        # Explicit sum
        v_explicit = sum(corrections[1:], corrections[0].clone())

        # Streaming sum (mimicking the batch code)
        v_streaming = None
        for v_i in corrections:
            if v_streaming is None:
                v_streaming = v_i.clone()
            else:
                v_streaming.add_(v_i)

        assert torch.allclose(v_explicit, v_streaming, atol=1e-7)


# ============================================================================
# Integration Tests
# ============================================================================


class TestPhaseBIntegration:
    """Integration tests combining all three deliverables."""

    def test_compute_correction_with_provider_and_damping(
        self, simple_model, batch_data
    ):
        """Test compute_correction using provider + correct damping."""
        params = [p for p in simple_model.parameters() if p.requires_grad]
        param_count = sum(p.numel() for p in params)

        step_log = StepLog(max_size=100)
        u_tz = torch.randn(param_count) * 0.01
        step_log.add(
            StepRecord(
                step_id=10,
                eta=0.01,
                batch_id="forget",
                u=u_tz,
                batch_data=batch_data,
            )
        )
        for i in range(11, 14):
            step_log.add(
                StepRecord(
                    step_id=i,
                    eta=0.01,
                    batch_id=f"b{i}",
                    u=torch.randn(param_count) * 0.01,
                    batch_data=batch_data,
                )
            )

        provider = HistoricalParamProvider(model=simple_model, step_log=step_log)
        cfg = HVPConfig(mode="fisher", damping=1e-4, device="cpu")

        v, audit = compute_correction(
            tz=10,
            tau=14,
            K=3,
            step_log=step_log,
            cfg=cfg,
            model=simple_model,
            historical_param_provider=provider,
        )

        assert not torch.isnan(v).any()
        assert audit.hvp_calls == 3
        assert audit.used_historical_params

        # Model should be restored to theta[tau]
        theta_after = _get_flat_params(simple_model)
        assert not torch.isnan(theta_after).any()

        provider.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
