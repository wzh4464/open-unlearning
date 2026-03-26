"""
Lazy u[t] provider: loads u[t] from pkl if available, otherwise recomputes
u[t] = -η[t] * ḡ[t] on the fly using sample_indices + dataset.

Assumes SGD training (no momentum/adaptive state needed).
"""

import gc
import logging
from typing import Dict, List, Optional, Protocol

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class UProviderProtocol(Protocol):
    """Interface for on-demand u[t] vector providers."""

    def get_u(self, step_id: int) -> Optional[torch.Tensor]:
        ...


class LazyLoaderProtocol(Protocol):
    """Minimal interface for loading step records."""

    def load_single_step(
        self, step_id: int, include_tensors: bool = True
    ) -> Optional[Dict]:
        ...

    def get_etas_for_steps(self, step_ids: List[int]) -> Dict[int, float]:
        ...


class LazyUProvider:
    """Provides u[t] vectors on demand: from pkl if available, else recomputed.

    For SGD training: u[t] = -η[t] * ḡ[t] where ḡ is the average gradient
    over the mini-batch at step t.

    Usage:
        provider = LazyUProvider(
            lazy_loader=loader,
            model=model,
            dataset=dataset,
            sample_indices=sample_indices_map,
            eta_cache=eta_map,
            collator=collator,
            device="cuda",
        )
        u = provider.get_u(step_id=42)  # from pkl or recomputed
    """

    def __init__(
        self,
        lazy_loader: LazyLoaderProtocol,
        model: nn.Module,
        dataset,
        sample_indices: Dict[int, List[int]],
        eta_cache: Dict[int, float],
        collator=None,
        device: str = "cuda",
        micro_batch_size: int = 8,
    ):
        self.lazy_loader = lazy_loader
        self.model = model
        self.dataset = dataset
        self.sample_indices = sample_indices
        self.eta_cache = eta_cache
        self.collator = collator
        self.device = device
        self.micro_batch_size = micro_batch_size

        # Cache: step_id -> u[t] tensor (CPU, to save GPU memory)
        self._cache: Dict[int, torch.Tensor] = {}
        self._stats = {"recomputed": 0, "cache_hits": 0}

    def get_u(self, step_id: int) -> Optional[torch.Tensor]:
        """Recompute u[t] = -η * grad for a given step.

        Note: pkl loading is handled by the caller (_load_u_for_step);
        this method only does cache lookup + recomputation to avoid
        double pkl reads.
        """
        # Check memory cache
        if step_id in self._cache:
            self._stats["cache_hits"] += 1
            return self._cache[step_id]

        # Recompute: u[t] = -η[t] * ḡ[t]
        u = self._recompute(step_id)
        if u is not None:
            self._stats["recomputed"] += 1
            self._cache[step_id] = u.cpu()
            return u

        return None

    def _recompute(self, step_id: int) -> Optional[torch.Tensor]:
        """Recompute u[t] = -η[t] * ḡ[t] via forward+backward on the batch."""
        # Get eta
        eta = self.eta_cache.get(step_id)
        if eta is None:
            logger.warning(f"Cannot recompute u[{step_id}]: eta not in cache")
            return None

        # Get sample indices
        indices = self.sample_indices.get(step_id)
        if not indices:
            logger.warning(f"Cannot recompute u[{step_id}]: no sample_indices")
            return None

        # Reconstruct batch and compute gradient
        try:
            # Collect samples
            samples = [self.dataset[i] for i in indices]

            # Collate
            if self.collator is not None:
                batch = self.collator(samples)
            else:
                batch = self._default_collate(samples)

            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Forward + backward (accumulate over micro-batches if needed)
            self.model.zero_grad()

            # For large batches, split into micro-batches
            total_samples = len(indices)
            if total_samples <= self.micro_batch_size:
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
            else:
                # Micro-batch accumulation: slice all tensor fields generically
                n_micro_batches = (total_samples + self.micro_batch_size - 1) // self.micro_batch_size
                for start in range(0, total_samples, self.micro_batch_size):
                    end = min(start + self.micro_batch_size, total_samples)
                    micro = {
                        k: v[start:end] if isinstance(v, torch.Tensor) and v.shape[0] == total_samples else v
                        for k, v in batch.items()
                    }
                    outputs = self.model(**micro)
                    (outputs.loss / n_micro_batches).backward()

            # Collect gradient → u = -η * grad
            grads = []
            for p in self.model.parameters():
                if p.requires_grad:
                    if p.grad is not None:
                        grads.append(p.grad.detach().flatten())
                    else:
                        grads.append(torch.zeros(p.numel(), device=self.device))

            gbar = torch.cat(grads)
            u = -eta * gbar

            # Cleanup
            self.model.zero_grad()
            del batch, gbar, grads
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return u.cpu()

        except Exception as e:
            logger.warning(f"Failed to recompute u[{step_id}]: {e}")
            return None

    def _default_collate(self, samples):
        """Simple collation for tokenized samples."""
        from torch.nn.utils.rnn import pad_sequence

        input_ids = [torch.tensor(s["input_ids"]) for s in samples]
        labels = [torch.tensor(s["labels"]) for s in samples]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = (input_ids != 0).long()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def clear_cache(self):
        """Clear the in-memory u[t] cache."""
        self._cache.clear()
        gc.collect()

    def get_stats(self) -> Dict[str, int]:
        """Return usage statistics."""
        return dict(self._stats)

    def preload_range(self, start: int, end: int):
        """Pre-load/compute u[t] for a range of steps [start, end]."""
        for t in range(start, end + 1):
            self.get_u(t)
