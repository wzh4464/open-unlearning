import torch
from typing import Dict, Any
from omegaconf import DictConfig
from transformers import Trainer, TrainingArguments

from trainer.base import FinetuneTrainer
from trainer.training_logger import TrainingLogger
from trainer.unlearn.grad_ascent import GradAscent
from trainer.unlearn.grad_diff import GradDiff
from trainer.unlearn.npo import NPO
from trainer.unlearn.dpo import DPO
from trainer.unlearn.simnpo import SimNPO
from trainer.unlearn.rmu import RMU
from trainer.unlearn.undial import UNDIAL
from trainer.unlearn.ceu import CEU
from trainer.unlearn.satimp import SatImp
from trainer.unlearn.wga import WGA
from trainer.unlearn.pdu import PDU
from trainer.unlearn.lmcleaner_batch import LMCleanerBatchLevel
from trainer.unlearn.lmcleaner_sample import LMCleanerSampleLevel


import logging

logger = logging.getLogger(__name__)

TRAINER_REGISTRY: Dict[str, Any] = {}


def _register_trainer(trainer_class):
    TRAINER_REGISTRY[trainer_class.__name__] = trainer_class


def load_trainer_args(trainer_args: DictConfig, dataset):
    trainer_args = dict(trainer_args)
    warmup_epochs = trainer_args.pop("warmup_epochs", None)

    # Remove training_logger config as it's not a TrainingArguments parameter
    trainer_args.pop("training_logger", None)

    if warmup_epochs:
        batch_size = trainer_args["per_device_train_batch_size"]
        grad_accum_steps = trainer_args["gradient_accumulation_steps"]
        num_devices = torch.cuda.device_count()
        dataset_len = len(dataset)
        trainer_args["warmup_steps"] = int(
            (warmup_epochs * dataset_len)
            // (batch_size * grad_accum_steps * num_devices)
        )

    trainer_args = TrainingArguments(**trainer_args)
    return trainer_args


def load_trainer(
    trainer_cfg: DictConfig,
    model,
    train_dataset=None,
    eval_dataset=None,
    tokenizer=None,
    data_collator=None,
    evaluators=None,
    template_args=None,
):
    trainer_args = trainer_cfg.args
    method_args = trainer_cfg.get("method_args", {})
    trainer_args = load_trainer_args(trainer_args, train_dataset)
    trainer_handler_name = trainer_cfg.get("handler")
    assert trainer_handler_name is not None, ValueError(
        f"{trainer_handler_name} handler not set"
    )
    trainer_cls = TRAINER_REGISTRY.get(trainer_handler_name, None)
    assert trainer_cls is not None, NotImplementedError(
        f"{trainer_handler_name} not implemented or not registered"
    )

    # Initialize TrainingLogger if configured
    # Only initialize on main process to avoid DeepSpeed ZeRO-3 conflicts
    training_logger = None
    logger_cfg = trainer_cfg.args.get("training_logger", None)
    if logger_cfg and logger_cfg.get("enabled", False):
        # Check if we're on the main process in distributed training
        is_main_process = True
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                is_main_process = dist.get_rank() == 0
        except Exception:
            # Not in distributed mode or dist not available
            is_main_process = True

        if is_main_process:
            logger.info("Initializing TrainingLogger on main process...")
            training_logger = TrainingLogger(
                log_dir=logger_cfg.get("log_dir", "saves/train_logs/default"),
                max_steps=logger_cfg.get("max_steps", 1000),
                mode=logger_cfg.get("mode", "batch"),
                save_interval=logger_cfg.get("save_interval", 100),
                save_batch_data=logger_cfg.get("save_batch_data", False),
                save_indices_only=logger_cfg.get("save_indices_only", False),
                save_rng_state=logger_cfg.get("save_rng_state", False),
                compute_diag_h=logger_cfg.get("compute_diag_h", False),
                batch_size_at_training=logger_cfg.get("batch_size_at_training", None),
            )
            logger.info(f"TrainingLogger initialized: {training_logger.log_dir}")
        else:
            logger.info("Skipping TrainingLogger initialization on non-main process (distributed training)")

    trainer = trainer_cls(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=trainer_args,
        evaluators=evaluators,
        template_args=template_args,
        training_logger=training_logger,
        **method_args,
    )
    logger.info(
        f"{trainer_handler_name} Trainer loaded, output_dir: {trainer_args.output_dir}"
    )
    return trainer, trainer_args


# Register Finetuning Trainer
_register_trainer(Trainer)
_register_trainer(FinetuneTrainer)

# Register Unlearning Trainer
_register_trainer(GradAscent)
_register_trainer(GradDiff)
_register_trainer(NPO)
_register_trainer(DPO)
_register_trainer(SimNPO)
_register_trainer(RMU)
_register_trainer(UNDIAL)
_register_trainer(CEU)
_register_trainer(SatImp)
_register_trainer(WGA)
_register_trainer(PDU)
_register_trainer(LMCleanerBatchLevel)
_register_trainer(LMCleanerSampleLevel)
