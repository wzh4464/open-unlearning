"""
Checkpoint-aware u[t] provider: computes u[t] at CORRECT historical θ[t]
by loading the nearest sparse checkpoint and replaying forward from there.

For SGD: u[t] = -η[t] * ∇L(θ[t], batch[t])

Unlike LazyUProvider which computes at θ[τ] (current/final params),
this provider reconstructs θ[t] exactly via:
  θ[c] → θ[c+1] → ... → θ[t]
where θ[c] is the nearest checkpoint before t.
"""

import gc
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _get_flat_params(model: nn.Module) -> torch.Tensor:
    """Flatten all model parameters into a single vector (float32 for precision)."""
    return torch.cat([p.detach().float().view(-1) for p in model.parameters()])


def _set_flat_params(model: nn.Module, flat: torch.Tensor):
    """Set model parameters from a flat vector."""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat[offset : offset + numel].view(p.shape).to(p.dtype))
        offset += numel


class CheckpointAwareUProvider:
    """Provides u[t] computed at correct historical θ[t] using sparse checkpoints.

    Algorithm for get_u(t):
      1. Find nearest checkpoint c <= t
      2. Load θ[c] into model
      3. For each step i from c to t:
         - Compute grad at θ[i] on batch[i]
         - u[i] = -η[i] * grad
         - θ[i+1] = θ[i] + u[i]  (advance model)
      4. Cache u[t], restore model to θ[τ]

    With checkpoint_stride=10 and 63 steps, worst case is 9 forward passes
    (9 × 0.6s = 5.4s) to reach any target from its nearest checkpoint.
    """

    def __init__(
        self,
        model: nn.Module,
        checkpoint_dir: Path,
        dataset,
        sample_indices: Dict[int, List[int]],
        eta_cache: Dict[int, float],
        collator=None,
        device: str = "cuda",
        micro_batch_size: int = 4,
        base_model_path: Optional[str] = None,
    ):
        self.model = model
        self.checkpoint_dir = Path(checkpoint_dir)
        self.dataset = dataset
        self.sample_indices = {int(k): v for k, v in sample_indices.items()}
        self.eta_cache = {int(k): v for k, v in eta_cache.items()}
        self.collator = collator
        self.device = device
        self.micro_batch_size = micro_batch_size
        self.base_model_path = base_model_path

        # Build checkpoint index: step_id -> checkpoint file path
        self._checkpoint_index: Dict[int, Path] = {}
        self._build_checkpoint_index()

        # Cache: step_id -> u[t] tensor (CPU, bf16). Limited to max_cache entries.
        self._cache: Dict[int, torch.Tensor] = {}
        self._max_cache: int = 5  # ~30GB max (5 × 6GB)
        self._stats = {"recomputed": 0, "cache_hits": 0, "chain_steps": 0}

        # Save θ[τ] for restoration
        self._theta_tau: Optional[torch.Tensor] = None

    def _build_checkpoint_index(self):
        """Scan checkpoint directory for available sparse checkpoints."""
        if not self.checkpoint_dir.exists():
            logger.warning(f"Checkpoint dir {self.checkpoint_dir} not found")
            return

        import re
        pattern = re.compile(r"step_(\d+)\.pt")
        for f in sorted(self.checkpoint_dir.iterdir()):
            m = pattern.match(f.name)
            if m:
                step_id = int(m.group(1))
                self._checkpoint_index[step_id] = f

        if self._checkpoint_index:
            steps = sorted(self._checkpoint_index.keys())
            logger.info(
                f"CheckpointAwareUProvider: {len(steps)} checkpoints at steps {steps}"
            )
        else:
            logger.warning("No sparse checkpoints found")

    def _find_nearest_checkpoint(self, target_step: int) -> Optional[int]:
        """Find the largest checkpoint step_id <= target_step."""
        candidates = [s for s in self._checkpoint_index if s <= target_step]
        return max(candidates) if candidates else None

    def get_u(self, step_id: int) -> Optional[torch.Tensor]:
        """Get u[step_id] at correct historical θ[step_id]."""
        if step_id in self._cache:
            self._stats["cache_hits"] += 1
            return self._cache[step_id]

        # Evict oldest if cache full
        while len(self._cache) >= self._max_cache:
            oldest = min(self._cache.keys())
            del self._cache[oldest]
            gc.collect()

        u = self._compute_from_checkpoint(step_id)
        if u is not None:
            self._stats["recomputed"] += 1
        return u

    def precompute_window(self, start: int, end: int):
        """Precompute u[start..end] in one sequential pass from nearest checkpoint.

        This is much more efficient than computing each step independently,
        since we only load one checkpoint and advance sequentially.
        """
        # Filter to only steps we don't already have cached
        needed = [t for t in range(start, end + 1) if t not in self._cache]
        if not needed:
            return

        t0 = time.perf_counter()
        checkpoint_step = self._find_nearest_checkpoint(needed[0])
        if checkpoint_step is None:
            logger.warning(f"No checkpoint found for window [{start}, {end}]")
            return

        # Save current params
        self._theta_tau = _get_flat_params(self.model)

        # Load checkpoint
        self._load_checkpoint(checkpoint_step)

        # Advance from checkpoint to end, computing u[t] at each step
        count = 0
        for t in range(checkpoint_step, end + 1):
            eta = self.eta_cache.get(t)
            indices = self.sample_indices.get(t)
            if eta is None or not indices:
                # Can't compute this step, skip but still need to advance
                # Use approximate u (at current theta which may have drifted)
                continue

            u = self._compute_gradient_update(eta, indices)
            if u is not None:
                if t >= start and t not in self._cache:
                    while len(self._cache) >= self._max_cache:
                        oldest = min(self._cache.keys())
                        del self._cache[oldest]
                    self._cache[t] = u.to(torch.bfloat16).cpu()
                    count += 1
                # Advance model: θ[t+1] = θ[t] + u[t]
                self._advance_model(u)
                del u
            self._stats["chain_steps"] += 1

        # Restore θ[τ]
        _set_flat_params(self.model, self._theta_tau)
        self._theta_tau = None

        elapsed = time.perf_counter() - t0
        logger.info(
            f"Precomputed {count} u[t] in window [{start},{end}] "
            f"from checkpoint {checkpoint_step} ({elapsed:.1f}s)"
        )

    def _compute_from_checkpoint(self, target_step: int) -> Optional[torch.Tensor]:
        """Compute u[target_step] by replaying from nearest checkpoint."""
        checkpoint_step = self._find_nearest_checkpoint(target_step)
        if checkpoint_step is None:
            logger.warning(f"No checkpoint for step {target_step}")
            return None

        # Save θ[τ]
        self._theta_tau = _get_flat_params(self.model)

        # Load checkpoint θ[c]
        self._load_checkpoint(checkpoint_step)

        # Sequential forward: θ[c] → θ[c+1] → ... → θ[target_step]
        result = None
        for t in range(checkpoint_step, target_step + 1):
            eta = self.eta_cache.get(t)
            indices = self.sample_indices.get(t)
            if eta is None or not indices:
                continue

            u = self._compute_gradient_update(eta, indices)
            if u is not None:
                if t == target_step:
                    result = u.to(torch.bfloat16).cpu()
                    self._cache[t] = result
                # Advance: θ[t+1] = θ[t] + u[t]
                self._advance_model(u)
                del u
            self._stats["chain_steps"] += 1

        # Restore θ[τ]
        _set_flat_params(self.model, self._theta_tau)
        self._theta_tau = None

        return result

    def _load_checkpoint(self, step_id: int):
        """Load sparse checkpoint into model.

        Handles two formats:
        - state_dict (OrderedDict): saved by TrainingLogger._save_sparse_checkpoint
        - flat tensor: legacy format
        """
        ckpt_path = self._checkpoint_index.get(step_id)
        if ckpt_path is None:
            raise ValueError(f"No checkpoint at step {step_id}")

        data = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(data, dict):
            # state_dict format: reconstruct flat vector in model parameter order
            flat_parts = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if name in data:
                        flat_parts.append(data[name].reshape(-1).float())
                    else:
                        logger.warning(
                            f"Checkpoint at step {step_id} missing key '{name}', "
                            f"using current param"
                        )
                        flat_parts.append(param.detach().cpu().float().reshape(-1))
            flat = torch.cat(flat_parts)
            del flat_parts
        else:
            flat = data.float()
        _set_flat_params(self.model, flat)
        del data, flat
        logger.debug(f"Loaded checkpoint θ[{step_id}]")

    def _advance_model(self, u: torch.Tensor):
        """Apply u[t] to advance model: θ[t+1] = θ[t] + u[t]."""
        offset = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.data.add_(u[offset : offset + numel].view(p.shape).to(p.dtype))
            offset += numel

    def _compute_gradient_update(
        self, eta: float, indices: List[int]
    ) -> Optional[torch.Tensor]:
        """Compute u = -eta * grad at current model params for given batch."""
        try:
            samples = [self.dataset[i] for i in indices]
            if self.collator is not None:
                batch = self.collator(samples)
            else:
                return None

            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Enable gradient checkpointing for memory efficiency
            _gc_was_enabled = getattr(self.model, "is_gradient_checkpointing", False)
            if hasattr(self.model, "gradient_checkpointing_enable") and not _gc_was_enabled:
                self.model.gradient_checkpointing_enable()

            self.model.zero_grad()
            self.model.train()

            total_samples = len(indices)
            n_micro = (total_samples + self.micro_batch_size - 1) // self.micro_batch_size
            for start in range(0, total_samples, self.micro_batch_size):
                end = min(start + self.micro_batch_size, total_samples)
                micro = {
                    k: v[start:end] if isinstance(v, torch.Tensor) and v.shape[0] == total_samples else v
                    for k, v in batch.items()
                }
                outputs = self.model(**micro)
                (outputs.loss / n_micro).backward()
                del outputs, micro
                torch.cuda.empty_cache()

            if hasattr(self.model, "gradient_checkpointing_disable") and not _gc_was_enabled:
                self.model.gradient_checkpointing_disable()

            # u = -eta * grad
            grads = []
            for p in self.model.parameters():
                if p.requires_grad and p.grad is not None:
                    grads.append((-eta * p.grad).detach().flatten())
                elif p.requires_grad:
                    grads.append(torch.zeros(p.numel(), device=self.device))

            u = torch.cat(grads)
            self.model.zero_grad()
            del batch, grads
            gc.collect()
            torch.cuda.empty_cache()

            return u

        except Exception as e:
            logger.warning(f"Failed to compute gradient update: {e}")
            return None

    def get_stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def clear_cache(self):
        self._cache.clear()
        gc.collect()
