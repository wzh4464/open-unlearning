import torch
import random
import numpy as np
from torch import nn
import torch.nn.functional as F
from transformers import TrainerCallback
import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_kl_divergence(model, target_model, inputs):
    with torch.no_grad():
        ref_outputs = target_model(**inputs)

    ref_probs = F.log_softmax(ref_outputs.logits, dim=-1)
    ref_probs = ref_probs.view(-1, ref_outputs.logits.shape[-1])

    outputs = model(**inputs)
    current_probs = F.log_softmax(outputs.logits, dim=-1)
    current_probs = current_probs.view(-1, outputs.logits.shape[-1])

    # minimum KL divergence
    return nn.functional.kl_div(
        current_probs, ref_probs, reduction="batchmean", log_target=True
    ), outputs


def compute_batch_nll(model, inputs):
    # get the sum loss for each sequence in a batch
    # NOTE: not same as model(**inputs).loss but has sum loss for each seq in a batch
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return loss, outputs


def compute_dpo_loss(model, ref_model, win_inputs=None, lose_inputs=None, beta=1.0):
    if win_inputs is None and lose_inputs is None:
        raise ValueError("Both win_inputs and lose_inputs can't be None")

    win_log_ratio, lose_log_ratio = 0.0, 0.0
    win_outputs, lose_outputs = None, None

    if win_inputs is not None:
        win_loss, win_outputs = compute_batch_nll(model, win_inputs)
        with torch.no_grad():
            win_ref_loss, _ = compute_batch_nll(ref_model, win_inputs)
        win_log_ratio = -(win_loss - win_ref_loss)

    if lose_inputs is not None:
        lose_loss, lose_outputs = compute_batch_nll(model, lose_inputs)
        with torch.no_grad():
            lose_ref_loss, _ = compute_batch_nll(ref_model, lose_inputs)
        lose_log_ratio = -(lose_loss - lose_ref_loss)

    loss = -2 / beta * F.logsigmoid(beta * (win_log_ratio - lose_log_ratio)).mean()
    return loss, (win_outputs, lose_outputs)


def compute_undial_loss(model, ref_model, inputs, beta):
    # Forward pass on the student (trainable) model
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]

    shift_labels = labels[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()

    # Forward pass on the teacher model (no grad)
    with torch.no_grad():
        teacher_logits = ref_model(**inputs).logits
    shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()

    # Build the mask that identifies the tokens need to be unlearned
    mask = torch.zeros_like(shift_teacher_logits)
    batch_idx = torch.arange(mask.shape[0]).view(-1, 1, 1)
    seq_idx = torch.arange(mask.shape[1]).view(1, -1, 1)
    mask[batch_idx, seq_idx, shift_labels.unsqueeze(-1)] = 1.0

    # Adjust teacher logits: subtract di_strength on the correct token
    pre_softmax = shift_teacher_logits - mask * beta
    soft_label = F.softmax(pre_softmax, dim=-1)

    loss_fct = nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        soft_label.view(-1, soft_label.size(-1)),
    )
    return loss.mean(), outputs


def compute_wga_loss(model, inputs, beta):
    outputs = model(**inputs)
    labels = inputs["labels"]
    labels = labels.to(outputs.logits.device)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    lm_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    weight_ce = ((-lm_loss).exp().detach()) ** beta
    forget_loss = -(weight_ce * lm_loss)[shift_labels.view(-1) != -100].mean()
    return forget_loss, outputs


def compute_satimp_loss(model, inputs, beta1, beta2):
    outputs = model(**inputs)
    labels = inputs["labels"]
    labels = labels.to(outputs.logits.device)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    lm_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    weight_sat = ((-lm_loss).exp().detach()) ** beta1
    weight_imp = (1 - (-lm_loss).exp().detach()) ** beta2
    forget_loss = -((weight_sat * weight_imp) * lm_loss)[
        shift_labels.view(-1) != -100
    ].mean()
    return forget_loss, outputs


# Memory optimization callback for periodic CUDA cache clearing
class CudaCacheCallback(TrainerCallback):
    """
    Callback to periodically clear CUDA cache during training to reduce memory fragmentation.
    This callback synchronizes all processes before clearing the cache to ensure consistency.

    Instead of creating its own Accelerator instance, this callback uses the trainer's
    Accelerator to avoid conflicts in distributed training scenarios.

    Args:
        interval (int): Number of steps between cache clearing operations. Default: 10
    """
    def __init__(self, interval=10):
        self.interval = interval

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step"""
        if state.global_step > 0 and state.global_step % self.interval == 0:
            # Get the trainer instance from kwargs to access its accelerator
            trainer = kwargs.get("trainer")
            if trainer is not None and hasattr(trainer, "accelerator"):
                # Use trainer's accelerator to synchronize processes
                trainer.accelerator.wait_for_everyone()
                # Clear CUDA cache on all devices
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Synchronize again after clearing
                trainer.accelerator.wait_for_everyone()
            else:
                # Fallback if trainer or accelerator is not available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


@dataclass
class EfficiencyMetrics:
    """Efficiency metrics for unlearning methods"""
    # Time metrics
    unlearning_time_seconds: float = 0.0  # Time for unlearning operation only

    # Memory metrics
    peak_gpu_memory_mb: float = 0.0
    intermediate_storage_mb: float = 0.0  # Temporary storage (gradients, checkpoints, etc.)

    # Throughput metrics
    tokens_per_second: float = 0.0
    model_size_mb: float = 0.0
    total_steps: int = 0  # Number of training steps during unlearning

    # Computational metrics
    forward_passes_total: int = 0
    backward_passes_total: int = 0
    flops_estimate: float = 0.0  # Estimated FLOPs (if available)

    # Data requirements
    forget_samples_count: int = 0
    retain_samples_count: int = 0
    requires_retain_set: bool = False

    # GPU utilization (optional, requires pynvml)
    gpu_utilization_mean: float = 0.0
    gpu_utilization_max: float = 0.0

    # Latency metrics
    per_step_latency_mean_ms: float = 0.0
    per_step_latency_std_ms: float = 0.0
    per_step_latency_min_ms: float = 0.0
    per_step_latency_max_ms: float = 0.0

    # I/O metrics
    checkpoint_save_time_seconds: float = 0.0
    data_loading_time_seconds: float = 0.0

    def to_dict(self):
        return asdict(self)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class EfficiencyTracker(TrainerCallback):
    """Track computational and storage efficiency metrics during unlearning

    Note: This tracks UNLEARNING time specifically, not the initial training time.
    For unlearning benchmarks, we care about the cost of the unlearning operation itself.
    """

    def __init__(self, output_dir: str = None, storage_dirs: list = None,
                 gpu_sampling_interval: int = 10):
        self.output_dir = Path(output_dir) if output_dir else None
        self.storage_dirs = storage_dirs or []  # Directories to monitor for intermediate storage
        self.gpu_sampling_interval = gpu_sampling_interval

        # Time tracking
        self.start_time = None
        self.last_step_time = None
        self.step_latencies = []  # Per-step latencies in ms

        # Memory tracking
        self.peak_memory = 0.0
        self.peak_storage = 0.0  # Peak intermediate storage in MB

        # Pass counting
        self.forward_count = 0
        self.backward_count = 0
        self._forward_hook = None
        self._backward_hook = None

        # GPU utilization tracking (optional)
        self.gpu_utilizations = []
        self._nvml_initialized = False
        self._nvml_handle = None

        # Sample counts
        self.forget_samples = 0
        self.retain_samples = 0

        # Checkpoint timing
        self.checkpoint_save_times = []
        self._checkpoint_start_time = None

        # Data loading timing
        self.data_loading_time = 0.0

        # Try to initialize NVML for GPU utilization tracking
        self._try_init_nvml()

    def _try_init_nvml(self):
        """Try to initialize NVML for GPU utilization tracking"""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_initialized = True
            # Get handle for first GPU (or current device)
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            logger.info("NVML initialized for GPU utilization tracking")
        except (ImportError, Exception) as e:
            logger.debug(f"NVML not available for GPU tracking: {e}")
            self._nvml_initialized = False

    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization percentage"""
        if not self._nvml_initialized:
            return 0.0
        try:
            import pynvml
            util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
            return float(util.gpu)
        except Exception:
            return 0.0

    def _forward_hook_fn(self, module, input, output):
        """Hook to count forward passes"""
        self.forward_count += 1

    def _backward_hook_fn(self, module, grad_input, grad_output):
        """Hook to count backward passes"""
        self.backward_count += 1

    def _get_dir_size_mb(self, path: Path) -> float:
        """Get total size of directory in MB"""
        if not path.exists():
            return 0.0
        total = 0
        try:
            for f in path.rglob('*'):
                if f.is_file():
                    total += f.stat().st_size
        except (PermissionError, OSError):
            pass
        return total / (1024 ** 2)

    def _get_intermediate_storage(self) -> float:
        """Calculate total intermediate storage across monitored directories"""
        total = 0.0
        for dir_path in self.storage_dirs:
            path = Path(dir_path) if isinstance(dir_path, str) else dir_path
            total += self._get_dir_size_mb(path)
        return total

    def _get_sample_counts(self, trainer):
        """Extract forget and retain sample counts from dataset"""
        if trainer is None or not hasattr(trainer, 'train_dataset'):
            return 0, 0

        dataset = trainer.train_dataset
        forget_count = 0
        retain_count = 0

        # Check for ForgetRetainDataset structure
        if hasattr(dataset, 'forget') and hasattr(dataset, 'retain'):
            forget_count = len(dataset.forget) if dataset.forget else 0
            retain_count = len(dataset.retain) if dataset.retain else 0
        elif hasattr(dataset, 'forget_dataset') and hasattr(dataset, 'retain_dataset'):
            forget_count = len(dataset.forget_dataset) if dataset.forget_dataset else 0
            retain_count = len(dataset.retain_dataset) if dataset.retain_dataset else 0
        else:
            # Single dataset, assume all are forget samples
            forget_count = len(dataset) if hasattr(dataset, '__len__') else 0

        return forget_count, retain_count

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Called at the beginning of unlearning training"""
        self.start_time = time.time()
        self.last_step_time = self.start_time

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Register hooks for forward/backward counting
        if model is not None:
            # Register on the model itself (counts full forward/backward passes)
            self._forward_hook = model.register_forward_hook(self._forward_hook_fn)
            try:
                self._backward_hook = model.register_full_backward_hook(
                    self._backward_hook_fn
                )
            except Exception:
                # Fallback for older PyTorch versions
                pass

        # Get sample counts
        trainer = kwargs.get('trainer')
        self.forget_samples, self.retain_samples = self._get_sample_counts(trainer)

    def on_step_end(self, args, state, control, **kwargs):
        """Update metrics after each step"""
        current_time = time.time()

        # Track per-step latency
        if self.last_step_time is not None:
            latency_ms = (current_time - self.last_step_time) * 1000
            self.step_latencies.append(latency_ms)
        self.last_step_time = current_time

        # Track peak GPU memory
        if torch.cuda.is_available():
            current_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
            self.peak_memory = max(self.peak_memory, current_memory)

        # Track peak intermediate storage
        current_storage = self._get_intermediate_storage()
        self.peak_storage = max(self.peak_storage, current_storage)

        # Sample GPU utilization periodically
        if (self._nvml_initialized and
                state.global_step % self.gpu_sampling_interval == 0):
            util = self._get_gpu_utilization()
            if util > 0:
                self.gpu_utilizations.append(util)

    def on_save(self, args, state, control, **kwargs):
        """Track checkpoint save time"""
        if self._checkpoint_start_time is None:
            # Start timing
            self._checkpoint_start_time = time.time()
        else:
            # End timing (called after save completes)
            save_time = time.time() - self._checkpoint_start_time
            self.checkpoint_save_times.append(save_time)
            self._checkpoint_start_time = None

    def on_train_end(self, args, state, control, model=None, **kwargs):
        """Compute and save final unlearning efficiency metrics"""
        if not self.start_time:
            logger.warning("Start time was not recorded, timing metrics may be inaccurate")

        unlearning_time = time.time() - self.start_time if self.start_time else 0

        # Remove hooks
        if self._forward_hook is not None:
            self._forward_hook.remove()
        if self._backward_hook is not None:
            self._backward_hook.remove()

        # Shutdown NVML
        if self._nvml_initialized:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass

        # Use state.global_step for accurate step count
        total_steps = state.global_step if hasattr(state, 'global_step') else 0

        # Estimate tokens processed during unlearning
        tokens_per_sec = 0
        if hasattr(state, 'num_input_tokens_seen') and state.num_input_tokens_seen > 0:
            tokens_per_sec = (state.num_input_tokens_seen / unlearning_time
                             if unlearning_time > 0 else 0)

        # Get model size
        model_size_mb = 0
        if model is not None:
            model_size_mb = sum(
                p.numel() * p.element_size() for p in model.parameters()
            ) / (1024 ** 2)

        # Determine if method requires retain set (check trainer type)
        requires_retain_set = False
        trainer = kwargs.get('trainer')
        if trainer is not None:
            trainer_name = trainer.__class__.__name__
            # Methods that use retain set for regularization
            retain_set_methods = ['GradDiff', 'NPO', 'DPO', 'SimNPO', 'RMU', 'UNDIAL']
            requires_retain_set = any(method in trainer_name for method in retain_set_methods)

        # Compute latency statistics
        latency_mean = 0.0
        latency_std = 0.0
        latency_min = 0.0
        latency_max = 0.0
        if self.step_latencies:
            latency_mean = np.mean(self.step_latencies)
            latency_std = np.std(self.step_latencies)
            latency_min = np.min(self.step_latencies)
            latency_max = np.max(self.step_latencies)

        # Compute GPU utilization statistics
        gpu_util_mean = 0.0
        gpu_util_max = 0.0
        if self.gpu_utilizations:
            gpu_util_mean = np.mean(self.gpu_utilizations)
            gpu_util_max = np.max(self.gpu_utilizations)

        # Compute checkpoint save time
        checkpoint_time = sum(self.checkpoint_save_times)

        metrics = EfficiencyMetrics(
            unlearning_time_seconds=unlearning_time,
            peak_gpu_memory_mb=self.peak_memory,
            intermediate_storage_mb=self.peak_storage,
            tokens_per_second=tokens_per_sec,
            model_size_mb=model_size_mb,
            total_steps=total_steps,
            forward_passes_total=self.forward_count,
            backward_passes_total=self.backward_count,
            flops_estimate=0.0,  # Would require additional profiling
            forget_samples_count=self.forget_samples,
            retain_samples_count=self.retain_samples,
            requires_retain_set=requires_retain_set,
            gpu_utilization_mean=gpu_util_mean,
            gpu_utilization_max=gpu_util_max,
            per_step_latency_mean_ms=latency_mean,
            per_step_latency_std_ms=latency_std,
            per_step_latency_min_ms=latency_min,
            per_step_latency_max_ms=latency_max,
            checkpoint_save_time_seconds=checkpoint_time,
            data_loading_time_seconds=self.data_loading_time,
        )

        # Save metrics
        if self.output_dir:
            metrics.save(self.output_dir / "efficiency_metrics.json")

        # Log efficiency metrics via project logger
        logger.info(
            "\n%s\n"
            "Unlearning Efficiency Metrics:\n"
            "  Unlearning Time: %.2fs\n"
            "  Total Steps: %d\n"
            "  Peak GPU Memory: %.2f MB\n"
            "  Intermediate Storage: %.2f MB\n"
            "  Tokens/Second: %.2f\n"
            "  Model Size: %.2f MB\n"
            "  Forward Passes: %d\n"
            "  Backward Passes: %d\n"
            "  Forget Samples: %d\n"
            "  Retain Samples: %d\n"
            "  Requires Retain Set: %s\n"
            "  GPU Utilization (mean/max): %.1f%% / %.1f%%\n"
            "  Step Latency (mean/std): %.2f / %.2f ms\n"
            "  Checkpoint Save Time: %.2fs\n"
            "%s",
            "=" * 50,
            unlearning_time,
            total_steps,
            self.peak_memory,
            self.peak_storage,
            tokens_per_sec,
            model_size_mb,
            self.forward_count,
            self.backward_count,
            self.forget_samples,
            self.retain_samples,
            requires_retain_set,
            gpu_util_mean,
            gpu_util_max,
            latency_mean,
            latency_std,
            checkpoint_time,
            "=" * 50,
        )
