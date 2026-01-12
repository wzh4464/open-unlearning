# Modified from https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py

from typing import Dict, List, Optional, Union

import os
import logging
import torch
from transformers import Trainer
from torch.utils.data import Dataset
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from typing import Any

logger = logging.getLogger(__name__)


class FinetuneTrainer(Trainer):
    def __init__(self, evaluators=None, template_args=None, training_logger=None, *args, **kwargs):
        self.evaluators = evaluators
        self.template_args = template_args
        self.training_logger = training_logger
        super().__init__(*args, **kwargs)

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        trial: Dict[str, Any] = None,
    ) -> Dict[str, float]:
        # Run a custom evaluator and save results
        if self.evaluators:
            if self.accelerator.is_local_main_process:
                eval_metrics = {}
                if self.accelerator.num_processes == 1:
                    run_dir = self._get_output_dir(trial=trial)
                    checkpoint_folder = (
                        f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                    )
                    output_dir = os.path.join(run_dir, checkpoint_folder, "evals")
                    os.makedirs(output_dir, exist_ok=True)
                    eval_metrics = {}
                    for _, evaluator in self.evaluators.items():
                        eval_args = {
                            "output_dir": output_dir,
                            "template_args": self.template_args,
                            "model": self.model,
                            "tokenizer": self.tokenizer,
                        }
                        eval_metrics.update(evaluator.evaluate(**eval_args))
                    self.log(eval_metrics)
                else:
                    logger.warning(
                        "Custom evaluator can be run with this Trainer only when a single accelerator process is running."
                    )
                return eval_metrics

        if eval_dataset is None:
            return {}
        # Run the default HF Trainer evaluate method when eval dataset is provided
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training_step to log training data if TrainingLogger is enabled"""
        # Save model state before training step (for computing u[t])
        if self.training_logger is not None:
            # Let the logger track the model state before the step
            # Note: Skip with DeepSpeed ZeRO-3 as parameters are sharded
            if self.training_logger.prev_params is None:
                try:
                    from trainer.unlearn.lmcleaner_core import clone_parameters
                    self.training_logger.prev_params = clone_parameters(model)
                except Exception as e:
                    logger.debug(f"Failed to clone initial parameters: {e}. This is expected with DeepSpeed ZeRO-3.")

        # Perform the normal training step
        if num_items_in_batch is not None:
            loss = super().training_step(model, inputs, num_items_in_batch)
        else:
            loss = super().training_step(model, inputs)

        # Log the training step after optimization
        if self.training_logger is not None:
            step_id = self.state.global_step
            eta = self.optimizer.param_groups[0]['lr']

            # Get sample indices if available
            sample_indices = None
            if 'idx' in inputs:
                sample_indices = inputs['idx'].cpu().tolist() if torch.is_tensor(inputs['idx']) else inputs['idx']
            elif 'index' in inputs:
                sample_indices = inputs['index'].cpu().tolist() if torch.is_tensor(inputs['index']) else inputs['index']

            # Prepare batch data for logging (optional, based on configuration)
            batch_data = None
            if self.training_logger.save_batch_data:
                batch_data = {
                    k: v.detach().cpu() if torch.is_tensor(v) else v
                    for k, v in inputs.items()
                }

            # Compute diagonal Hessian if enabled
            diag_H = None
            if self.training_logger.compute_diag_h:
                # Compute diagonal Hessian approximation (using gradient squared as approximation)
                # This is a simplified version - can be enhanced with proper Hessian computation
                try:
                    if hasattr(model, 'named_parameters'):
                        valid_grads = [
                            p.grad.detach().pow(2).view(-1)
                            for p in model.parameters()
                            if p.grad is not None and p.grad.numel() > 0
                        ]
                        if valid_grads:
                            diag_H = torch.cat(valid_grads)
                except Exception as e:
                    logger.debug(f"Failed to compute diagonal Hessian: {e}. Skipping.")

            # Register the step
            self.training_logger.register_step(
                step_id=step_id,
                batch_id=step_id,  # Use step_id as batch_id for now
                eta=eta,
                model=model,
                batch_data=batch_data,
                diag_H=diag_H,
                sample_indices=sample_indices,
            )

        return loss

    def train(self, *args, **kwargs):
        """Override train to save logger at the end"""
        result = super().train(*args, **kwargs)

        # Save training logger at the end of training
        if self.training_logger is not None:
            logger.info("Saving training logger at the end of training...")
            self.training_logger.save_to_disk()
            logger.info(f"Training logger saved to {self.training_logger.log_dir}")

        return result
