"""
SleepAwareGRPOTrainer - GRPOTrainer with external vLLM server sleep coordination.

This subclass adds HTTP-based sleep/wake coordination for external vLLM servers,
enabling single-GPU RLHF by freeing vLLM GPU memory during backward passes.

Usage:
    from sleep_aware_grpo_trainer import SleepAwareGRPOTrainer
    from vllm_server_client import VLLMServerClient
    
    client = VLLMServerClient("http://localhost:8000/v1")
    
    trainer = SleepAwareGRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        tokenizer=tokenizer,
        vllm_server_client=client,  # Enable sleep mode
    )
"""

import torch
from trl import GRPOTrainer
from typing import Optional, Any

from vllm_server_client import VLLMServerClient


class SleepAwareGRPOTrainer(GRPOTrainer):
    """
    GRPOTrainer subclass with external vLLM server sleep/wake coordination.
    
    Key changes from base GRPOTrainer:
    1. Wakes vLLM before generation
    2. Sleeps vLLM before backward pass
    3. Reloads weights after optimizer step
    
    This enables sharing a single GPU between vLLM inference and model training.
    """
    
    def __init__(
        self,
        *args,
        vllm_server_client: Optional[VLLMServerClient] = None,
        sleep_level: int = 2,
        **kwargs
    ):
        """
        Initialize the sleep-aware trainer.
        
        Args:
            vllm_server_client: Client for HTTP-based sleep/wake coordination
            sleep_level: vLLM sleep level (1=CPU offload, 2=discard for updates)
            *args, **kwargs: Passed to GRPOTrainer
        """
        super().__init__(*args, **kwargs)
        self.vllm_server_client = vllm_server_client
        self.sleep_level = sleep_level
        self._server_sleeping = False
        
        if self.vllm_server_client:
            print("[SleepAwareGRPOTrainer] External vLLM sleep mode enabled")
    
    def _ensure_vllm_awake(self):
        """Wake up vLLM server if sleeping."""
        if self.vllm_server_client and self._server_sleeping:
            self.vllm_server_client.wake_up()
            self._server_sleeping = False
    
    def _sleep_vllm(self):
        """Put vLLM server to sleep to free GPU memory."""
        if self.vllm_server_client and not self._server_sleeping:
            self.vllm_server_client.sleep(level=self.sleep_level)
            self._server_sleeping = True
    
    def _reload_vllm_weights(self):
        """Reload updated weights into vLLM after training step."""
        if self.vllm_server_client:
            # Wake up weights memory first
            self.vllm_server_client.wake_up(tags="weights")
            # Trigger weight reload
            self.vllm_server_client.reload_weights()
            # Then restore KV cache
            self.vllm_server_client.wake_up(tags="kv_cache")
            self._server_sleeping = False
    
    def _prepare_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Override to wake vLLM before generation.
        
        _prepare_inputs is called before generation in GRPOTrainer.
        This is the right place to ensure vLLM is awake.
        """
        # Ensure vLLM is awake before generation
        self._ensure_vllm_awake()
        
        # Call parent implementation
        return super()._prepare_inputs(inputs)
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Override to add sleep/wake around backward pass.
        
        Flow:
        1. _prepare_inputs already woke vLLM (generation done)
        2. Sleep vLLM to free GPU memory
        3. Do forward/backward pass
        4. Wake and reload weights
        """
        # Sleep vLLM before training computation
        self._sleep_vllm()
        
        # Clear GPU cache before training
        if self.vllm_server_client:
            torch.cuda.empty_cache()
        
        # Call parent training step (forward + backward + optimizer)
        try:
            loss = super().training_step(model, inputs, num_items_in_batch)
        finally:
            # Always reload weights after training step
            if self.vllm_server_client:
                self._reload_vllm_weights()
        
        return loss
    
    def train(self, *args, **kwargs):
        """Override to ensure vLLM is awake at start and end."""
        if self.vllm_server_client:
            # Wait for server if needed
            if not self.vllm_server_client.health_check():
                print("[SleepAwareGRPOTrainer] Waiting for vLLM server...")
                self.vllm_server_client.wait_for_ready()
            
            # Ensure awake at start
            self._ensure_vllm_awake()
        
        try:
            result = super().train(*args, **kwargs)
        finally:
            # Ensure awake at end
            if self.vllm_server_client:
                self._ensure_vllm_awake()
        
        return result
