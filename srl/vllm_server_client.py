"""
vLLM Server Client for Sleep/Wake Coordination

This module provides utilities for coordinating with an external vLLM server
during RLHF training, enabling sleep mode to free GPU memory during gradient updates.

Usage:
    client = VLLMServerClient("http://localhost:8000")
    
    # Before generation
    client.wake_up()
    
    # Generate completions via vLLM...
    
    # Before training step
    client.sleep(level=2)
    
    # After weight update
    client.wake_up(tags="weights")
    client.reload_weights()
    client.wake_up(tags="kv_cache")
"""

import requests
import time
from typing import Optional


class VLLMServerClient:
    """
    HTTP client for vLLM server sleep mode coordination.
    
    Enables RLHF training to coordinate with external vLLM server:
    - Sleep: Free GPU memory for training
    - Wake: Resume inference
    - Reload: Load updated weights after gradient step
    """
    
    def __init__(self, base_url: str, timeout: int = 30):
        """
        Initialize the vLLM server client.
        
        Args:
            base_url: vLLM server URL (e.g., "http://localhost:8000/v1")
            timeout: Request timeout in seconds
        """
        # Remove /v1 suffix for dev endpoints
        self.base_url = base_url.rstrip('/').replace('/v1', '')
        self.timeout = timeout
        self._sleeping = False
    
    def is_sleeping(self) -> bool:
        """Check if server is currently sleeping."""
        try:
            response = requests.get(
                f"{self.base_url}/is_sleeping",
                timeout=self.timeout
            )
            if response.status_code == 200:
                return response.json().get("is_sleeping", False)
            return self._sleeping
        except requests.RequestException:
            return self._sleeping
    
    def sleep(self, level: int = 2) -> bool:
        """
        Put vLLM server to sleep.
        
        Args:
            level: Sleep level
                - 1: Offload weights to CPU, discard KV cache
                - 2: Discard weights and KV cache (for weight updates)
        
        Returns:
            True if successful
        """
        try:
            response = requests.post(
                f"{self.base_url}/sleep?level={level}",
                timeout=self.timeout
            )
            success = response.status_code == 200
            if success:
                self._sleeping = True
                print(f"  [vLLM] Server sleeping (level={level})")
            return success
        except requests.RequestException as e:
            print(f"  [vLLM] Sleep failed: {e}")
            return False
    
    def wake_up(self, tags: Optional[str] = None) -> bool:
        """
        Wake up vLLM server.
        
        Args:
            tags: Optional partial wake-up
                - "weights": Only restore model weights
                - "kv_cache": Only restore KV cache
                - None: Full wake-up
        
        Returns:
            True if successful
        """
        try:
            url = f"{self.base_url}/wake_up"
            if tags:
                url += f"?tags={tags}"
            
            response = requests.post(url, timeout=self.timeout)
            success = response.status_code == 200
            if success:
                if not tags:
                    self._sleeping = False
                print(f"  [vLLM] Server woke up" + (f" (tags={tags})" if tags else ""))
            return success
        except requests.RequestException as e:
            print(f"  [vLLM] Wake up failed: {e}")
            return False
    
    def reload_weights(self) -> bool:
        """
        Reload model weights in-place after training update.
        
        Call this after:
        1. sleep(level=2)
        2. Training gradient update
        3. wake_up(tags="weights")
        
        Returns:
            True if successful
        """
        try:
            response = requests.post(
                f"{self.base_url}/collective_rpc",
                json={"method": "reload_weights"},
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
            success = response.status_code == 200
            if success:
                print("  [vLLM] Weights reloaded")
            return success
        except requests.RequestException as e:
            print(f"  [vLLM] Reload weights failed: {e}")
            return False
    
    def health_check(self) -> bool:
        """Check if server is healthy and responsive."""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def wait_for_ready(self, max_wait: int = 60) -> bool:
        """
        Wait for server to be ready.
        
        Args:
            max_wait: Maximum seconds to wait
        
        Returns:
            True if server became ready
        """
        print(f"  [vLLM] Waiting for server at {self.base_url}...")
        start = time.time()
        while time.time() - start < max_wait:
            if self.health_check():
                print("  [vLLM] Server is ready")
                return True
            time.sleep(2)
        print("  [vLLM] Server not ready after timeout")
        return False


class VLLMSleepModeContext:
    """
    Context manager for vLLM sleep mode during training steps.
    
    Usage:
        client = VLLMServerClient("http://localhost:8000")
        
        with VLLMSleepModeContext(client):
            # vLLM is sleeping here
            loss.backward()
            optimizer.step()
        # vLLM wakes up and reloads weights
    """
    
    def __init__(self, client: VLLMServerClient, sleep_level: int = 2):
        self.client = client
        self.sleep_level = sleep_level
    
    def __enter__(self):
        self.client.sleep(level=self.sleep_level)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Wake up weights memory first
        self.client.wake_up(tags="weights")
        # Reload weights in-place
        self.client.reload_weights()
        # Then restore KV cache
        self.client.wake_up(tags="kv_cache")
        return False


# Import for callback
try:
    from transformers import TrainerCallback
    
    class VLLMSleepModeCallback(TrainerCallback):
        """
        Trainer callback that coordinates sleep/wake with external vLLM server.
        
        This enables single-GPU RLHF by:
        1. Waking vLLM before generation
        2. Sleeping vLLM during gradient updates
        3. Reloading updated weights after training step
        """
        
        def __init__(self, client: VLLMServerClient, sleep_level: int = 2):
            self.client = client
            self.sleep_level = sleep_level
        
        def on_step_begin(self, args, state, control, **kwargs):
            """Wake up vLLM before generation step."""
            if self.client.is_sleeping():
                self.client.wake_up()
        
        def on_step_end(self, args, state, control, **kwargs):
            """After step, prepare for next iteration."""
            # Note: In GRPOTrainer, weight updates happen during training_step
            # We sleep after the step is complete to prepare for next batch
            pass
        
        def on_train_begin(self, args, state, control, **kwargs):
            """Ensure vLLM is awake at training start."""
            if self.client.is_sleeping():
                self.client.wake_up()
            print("[vLLM Sleep Mode] Training started, server coordination active")
        
        def on_train_end(self, args, state, control, **kwargs):
            """Ensure vLLM is awake at training end."""
            if self.client.is_sleeping():
                self.client.wake_up()
            print("[vLLM Sleep Mode] Training complete")

except ImportError:
    # transformers not installed
    VLLMSleepModeCallback = None

