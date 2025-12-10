#!/usr/bin/env python3
"""
Async SRL Trainer with vLLM Sleep Mode

This module implements the SRL training loop with:
- vLLM sleep/wake cycle for VRAM<->RAM tiering
- Parallel CPU reward computation
- GRPO loss with PPO-clip objective

Optimized for RTX 5060 (8GB VRAM) + 32GB RAM.
"""

import os
import gc
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count

# Set environment variable before any imports
os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Import Unsloth FIRST (critical for proper patching)
import unsloth
from unsloth import FastLanguageModel, PatchFastRL

from srl_config import SRLConfig, DEFAULT_CONFIG
from srl_reward_function import SRLRewardFunction, DynamicSamplingFilter
from resource_monitor import ResourceMonitor


class AsyncSRLTrainer:
    """
    SRL Trainer with vLLM sleep mode for memory-efficient training.
    
    Implements time-division multiplexing:
    - Phase A (Inference): vLLM active, generate rollouts
    - Phase B (Training): vLLM asleep, compute loss and update weights
    """
    
    def __init__(self, config: Optional[SRLConfig] = None):
        """
        Initialize the trainer.
        
        Args:
            config: SRLConfig instance. If None, uses DEFAULT_CONFIG.
        """
        self.config = config or DEFAULT_CONFIG
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("=" * 60)
        print("Initializing AsyncSRLTrainer")
        print("=" * 60)
        print(f"Model: {self.config.model.model_name}")
        print(f"Device: {self.device}")
        print(f"Max Seq Length: {self.config.model.max_seq_length}")
        print(f"LoRA Rank: {self.config.model.lora_rank}")
        print(f"Num Rollouts: {self.config.training.num_rollouts}")
        print("=" * 60)
        
        # Patch Unsloth for RL routines
        try:
            PatchFastRL("GRPO", FastLanguageModel)
            print("✓ GRPO patches applied")
        except Exception as e:
            print(f"⚠ GRPO patch skipped: {e}")
        
        # Load model with vLLM integration
        self._load_model()
        
        # Initialize reward function
        self.reward_fn = SRLRewardFunction(
            format_check=self.config.reward.format_check,
            min_similarity=self.config.reward.similarity_threshold,
            penalty_for_format_error=self.config.reward.format_penalty,
        )
        self.sampler = DynamicSamplingFilter(variance_threshold=0.01)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # Thread pool for parallel reward computation
        self._executor = ThreadPoolExecutor(max_workers=cpu_count())
        
        # Resource monitor for logging CPU/GPU/VRAM/RAM usage
        # Convert config to dict for wandb logging
        config_dict = {
            "model_name": self.config.model.model_name,
            "max_seq_length": self.config.model.max_seq_length,
            "lora_rank": self.config.model.lora_rank,
            "learning_rate": self.config.training.learning_rate,
            "batch_size": self.config.training.batch_size,
            "num_rollouts": self.config.training.num_rollouts,
            "num_epochs": self.config.training.num_epochs,
        }
        
        self._resource_monitor = ResourceMonitor(
            verbose=self.config.logging.verbose,
            backend=self.config.logging.backend,
            log_dir=self.config.logging.log_dir,
            project=self.config.logging.wandb_project,
            run_name=self.config.logging.wandb_run_name,
            config=config_dict,
        )
        
        print("✓ AsyncSRLTrainer initialized successfully")
    
    def _load_model(self):
        """Load model with Unsloth and attach LoRA adapters."""
        print("\nLoading model...")
        
        # Build kwargs for from_pretrained
        load_kwargs = {
            "model_name": self.config.model.model_name,
            "max_seq_length": self.config.model.max_seq_length,
            "load_in_4bit": self.config.model.load_in_4bit,
        }
        
        # Only add vLLM options if fast_inference is enabled
        if self.config.vllm.fast_inference:
            load_kwargs["fast_inference"] = True
            load_kwargs["gpu_memory_utilization"] = self.config.vllm.gpu_memory_utilization
            print("  [vLLM] fast_inference enabled")
        else:
            print("  [vLLM] fast_inference disabled (using standard HF inference)")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
        
        print("✓ Base model loaded")
        
        # Attach LoRA adapters
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.model.lora_rank,
            target_modules=self.config.model.target_modules,
            lora_alpha=self.config.model.lora_alpha,
            lora_dropout=self.config.model.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        
        print("✓ LoRA adapters attached")
        self.model.print_trainable_parameters()
        
        # Verify trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable parameters! LoRA adapters may not be attached correctly.")
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.training.learning_rate,
            betas=(self.config.training.adam_beta1, self.config.training.adam_beta2),
            weight_decay=self.config.training.weight_decay,
        )
        
        print(f"✓ Optimizer initialized with {len(trainable_params)} parameter groups")
    
    def _vllm_sleep(self):
        """Put vLLM engine to sleep, offloading weights to CPU RAM."""
        if hasattr(self.model, 'vllm_engine') and self.model.vllm_engine is not None:
            try:
                self.model.vllm_engine.sleep(level=self.config.vllm.sleep_level)
                print("  [vLLM] Sleeping (weights offloaded to RAM)")
            except Exception as e:
                print(f"  [vLLM] Sleep failed: {e}")
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        # Log resources after sleep
        self._resource_monitor.log_snapshot("After vLLM Sleep")
    
    def _vllm_wake(self):
        """Wake vLLM engine, loading weights back to VRAM."""
        if hasattr(self.model, 'vllm_engine') and self.model.vllm_engine is not None:
            try:
                self.model.vllm_engine.wake_up()
                print("  [vLLM] Awake (weights loaded to VRAM)")
            except Exception as e:
                print(f"  [vLLM] Wake failed: {e}")
        
        # Log resources after wake
        self._resource_monitor.log_snapshot("After vLLM Wake")
    
    def generate_rollouts(self, prompts: List[str]) -> List[List[str]]:
        """
        Generate K rollouts for each prompt.
        
        Args:
            prompts: List of input prompts.
            
        Returns:
            List of lists, where each inner list contains K rollouts for a prompt.
        """
        all_rollouts = []
        
        # Switch to inference mode
        FastLanguageModel.for_inference(self.model)
        
        # Check if we have fast_generate (vLLM mode)
        use_vllm = hasattr(self.model, 'fast_generate')
        
        if use_vllm:
            # Import vLLM's SamplingParams
            from vllm import SamplingParams
            
            sampling_params = SamplingParams(
                max_tokens=self.config.vllm.max_completion_length,
                temperature=self.config.vllm.temperature,
                top_p=self.config.vllm.top_p,
            )
            
            for prompt in prompts:
                rollouts = []
                # Generate K rollouts for this prompt
                # vLLM fast_generate takes text prompts directly
                for _ in range(self.config.training.num_rollouts):
                    outputs = self.model.fast_generate(
                        [prompt],  # List of text prompts
                        sampling_params=sampling_params,
                        use_tqdm=False,
                    )
                    # Extract generated text from vLLM output
                    generated_text = outputs[0].outputs[0].text
                    action = self.reward_fn._extract_action_part(prompt + generated_text)
                    rollouts.append(action)
                
                all_rollouts.append(rollouts)
        else:
            # Standard HuggingFace generate fallback
            for prompt in prompts:
                rollouts = []
                input_ids = self.tokenizer(
                    prompt, 
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.model.max_seq_length,
                ).input_ids.to(self.device)
                
                with torch.no_grad():
                    for _ in range(self.config.training.num_rollouts):
                        output_ids = self.model.generate(
                            input_ids,
                            max_new_tokens=self.config.vllm.max_completion_length,
                            do_sample=True,
                            temperature=self.config.vllm.temperature,
                            top_p=self.config.vllm.top_p,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )[0]
                        
                        # Decode and extract action
                        text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                        action = self.reward_fn._extract_action_part(text)
                        rollouts.append(action)
                
                all_rollouts.append(rollouts)
        
        return all_rollouts
    
    def compute_rewards_parallel(
        self, 
        rollouts: List[List[str]], 
        expert_actions: List[str]
    ) -> List[List[float]]:
        """
        Compute rewards in parallel using ThreadPoolExecutor.
        
        This runs on CPU while GPU can be doing other work.
        
        Args:
            rollouts: List of rollout lists per prompt.
            expert_actions: List of expert actions.
            
        Returns:
            List of reward lists per prompt.
        """
        all_rewards = []
        
        def compute_single_reward(action: str, expert: str) -> float:
            return self.reward_fn(action, expert)
        
        for prompt_rollouts, expert in zip(rollouts, expert_actions):
            # Submit all rollouts for this prompt
            futures = [
                self._executor.submit(compute_single_reward, rollout, expert)
                for rollout in prompt_rollouts
            ]
            # Collect results
            rewards = [f.result() for f in futures]
            all_rewards.append(rewards)
        
        return all_rewards
    
    def compute_logprob(
        self, 
        prompt: str, 
        action: str, 
        return_tensor: bool = True
    ) -> torch.Tensor:
        """
        Compute log probability of action given prompt.
        
        Args:
            prompt: Input prompt.
            action: Action text.
            return_tensor: If True, return tensor (for gradients).
            
        Returns:
            Log probability (sum over tokens).
        """
        full_text = prompt + action
        
        input_ids = self.tokenizer(
            full_text, 
            return_tensors="pt",
            truncation=True,
            max_length=self.config.model.max_seq_length,
        ).input_ids.to(self.device)
        
        prompt_ids = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=self.config.model.max_seq_length,
        ).input_ids.to(self.device)
        
        # Forward pass
        if return_tensor:
            outputs = self.model(input_ids)
        else:
            with torch.no_grad():
                outputs = self.model(input_ids)
        
        logits = outputs.logits
        
        # Get log probs for action tokens only
        prompt_len = prompt_ids.shape[1]
        if prompt_len >= input_ids.shape[1]:
            # Edge case: action is empty or truncated
            return torch.tensor(0.0, device=self.device, requires_grad=return_tensor)
        
        action_logits = logits[0, prompt_len-1:-1, :]
        action_ids = input_ids[0, prompt_len:]
        
        if len(action_ids) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=return_tensor)
        
        # Compute log softmax and select token probs
        log_probs = torch.nn.functional.log_softmax(action_logits, dim=-1)
        token_log_probs = log_probs[torch.arange(len(action_ids)), action_ids]
        
        return token_log_probs.sum()
    
    def compute_grpo_loss(
        self,
        prompt: str,
        rollouts: List[str],
        rewards: List[float],
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute GRPO loss for one sample.
        
        Implements a simplified policy gradient with group-normalized advantages.
        
        Args:
            prompt: Input prompt.
            rollouts: List of generated actions (K rollouts).
            rewards: List of rewards for each rollout.
            
        Returns:
            Tuple of (loss tensor, KL divergence value).
        """
        device = self.device
        
        # Normalize advantages (group-level) - detached, no gradient needed
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        advantages = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std(unbiased=False) + 1e-8)
        
        # Compute log probabilities (with gradients)
        logprobs = []
        for action in rollouts:
            lp = self.compute_logprob(prompt, action, return_tensor=True)
            logprobs.append(lp)
        
        logprobs = torch.stack(logprobs)  # Shape: [K]
        
        # Policy gradient loss: -E[log_prob * advantage]
        # Negative because we want to maximize expected advantage
        policy_loss = -(logprobs * advantages.detach()).mean()
        
        # For now, KL = 0 since we're not using a reference model
        kl_div = 0.0
        
        return policy_loss, kl_div
    
    def train_step(self, batch: Dict) -> Dict:
        """
        Execute one training step (one batch).
        
        Implements the sleep/wake cycle:
        1. Wake vLLM
        2. Generate rollouts
        3. Sleep vLLM
        4. Compute rewards (parallel CPU)
        5. Compute loss and update weights
        
        Args:
            batch: Dictionary with 'prompts' and 'expert_actions'.
            
        Returns:
            Dictionary with training metrics.
        """
        prompts = batch.get("prompts", [])
        expert_actions = batch.get("expert_actions", [])
        
        if not prompts:
            return {"loss": 0.0, "kept": 0, "kl": 0.0}
        
        # Phase A: Rollout Generation (vLLM Active)
        self._vllm_wake()
        with self._resource_monitor.log_phase("Generation", step=self.global_step):
            rollouts = self.generate_rollouts(prompts)
        
        # Phase B: Training (vLLM Sleeping)
        self._vllm_sleep()
        
        # Compute rewards in parallel (CPU)
        with self._resource_monitor.log_phase("Reward Calculation", step=self.global_step):
            rewards = self.compute_rewards_parallel(rollouts, expert_actions)
        
        # Switch to training mode
        FastLanguageModel.for_training(self.model)
        self.optimizer.zero_grad()
        
        batch_losses = []
        batch_kl = []
        kept = 0
        
        with self._resource_monitor.log_phase("Training (Loss + Backward)", step=self.global_step):
            for prompt, prompt_rollouts, prompt_rewards in zip(prompts, rollouts, rewards):
                # Skip if low variance (dynamic sampling)
                if not self.sampler.should_keep_sample(prompt_rewards):
                    continue
                
                loss, kl = self.compute_grpo_loss(prompt, prompt_rollouts, prompt_rewards)
                loss.backward()
                
                batch_losses.append(loss.item())
                batch_kl.append(kl)
                kept += 1
            
            # Gradient clipping and optimizer step
            if kept > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    max_norm=self.config.training.max_grad_norm,
                )
                self.optimizer.step()
        
        self.global_step += 1
        self.sampler.step()  # Advance dynamic variance threshold
        
        # Log training metrics to backend
        step_metrics = {
            "train/loss": np.mean(batch_losses) if batch_losses else 0.0,
            "train/kept_samples": kept,
            "train/kl_divergence": np.mean(batch_kl) if batch_kl else 0.0,
            "train/avg_reward": np.mean([np.mean(r) for r in rewards]),
        }
        self._resource_monitor.log_metrics(step_metrics, step=self.global_step)
        
        return {
            "loss": step_metrics["train/loss"],
            "kept": kept,
            "kl": step_metrics["train/kl_divergence"],
            "avg_reward": step_metrics["train/avg_reward"],
        }
    
    def train_epoch(self, dataloader) -> Dict:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader yielding batches.
            
        Returns:
            Dictionary with epoch metrics.
        """
        self.epoch += 1
        epoch_losses = []
        epoch_kl = []
        epoch_kept = 0
        epoch_rewards = []
        
        print(f"\n{'='*60}")
        print(f"Epoch {self.epoch}")
        print('='*60)
        
        for batch_idx, batch in enumerate(dataloader):
            metrics = self.train_step(batch)
            
            epoch_losses.append(metrics["loss"])
            epoch_kl.append(metrics["kl"])
            epoch_kept += metrics["kept"]
            epoch_rewards.append(metrics["avg_reward"])
            
            if (batch_idx + 1) % self.config.log_steps == 0:
                print(f"  Batch {batch_idx + 1}: loss={metrics['loss']:.4f}, "
                      f"reward={metrics['avg_reward']:.4f}, kept={metrics['kept']}")
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        avg_kl = np.mean(epoch_kl) if epoch_kl else 0.0
        avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
        
        print(f"\nEpoch {self.epoch} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Average KL: {avg_kl:.4f}")
        print(f"  Average Reward: {avg_reward:.4f}")
        print(f"  Total Kept Samples: {epoch_kept}")
        
        # Log epoch metrics to backend
        epoch_metrics = {
            "epoch/avg_loss": avg_loss,
            "epoch/avg_kl": avg_kl,
            "epoch/avg_reward": avg_reward,
            "epoch/total_kept_samples": epoch_kept,
            "epoch/number": self.epoch,
        }
        self._resource_monitor.log_metrics(epoch_metrics, step=self.global_step)
        
        return {
            "epoch": self.epoch,
            "avg_loss": avg_loss,
            "avg_kl": avg_kl,
            "avg_reward": avg_reward,
            "total_kept": epoch_kept,
        }
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        print(f"Saving checkpoint to {path}...")
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"✓ Checkpoint saved")
    
    def __del__(self):
        """Cleanup executor and resource monitor on deletion."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
        if hasattr(self, '_resource_monitor'):
            self._resource_monitor.close()
