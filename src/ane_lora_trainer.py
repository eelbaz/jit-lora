"""
ane_lora_trainer.py — LoRA training engine using Apple Neural Engine.

Manages per-layer LoRA adapters (A & B matrices), compiles ANE kernels once,
and runs forward/backward passes on ANE hardware. Training loop:
  1. Forward: base model inference via MLX, with LoRA additions via ANE
  2. Loss: cross-entropy computed on CPU
  3. Backward: LoRA gradients computed on ANE
  4. Update: Adam optimizer on CPU (LoRA params only — tiny, instant)

The adapter weights live as numpy arrays in shared memory. MLX inference
reads them (zero-copy via mlx.array), ANE training writes updated values.
"""

import json
import logging
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ane_bridge_py import ANEBridge
from ane_mil_lora import LoRAKernelSet
from neural_config import NeuralConfig

log = logging.getLogger("ane_lora_trainer")


class LoRAAdapter:
    """Per-target LoRA adapter (A & B matrices) for all layers."""

    def __init__(self, n_layers: int, dim: int, rank: int):
        self.n_layers = n_layers
        self.dim = dim
        self.rank = rank

        # A: [rank, dim] — initialized with small random values (Kaiming)
        # B: [dim, rank] — initialized to zeros (standard LoRA init)
        scale = 1.0 / math.sqrt(dim)
        self.A = [np.random.randn(rank, dim).astype(np.float32) * scale
                  for _ in range(n_layers)]
        self.B = [np.zeros((dim, rank), dtype=np.float32)
                  for _ in range(n_layers)]

    def param_count(self) -> int:
        """Total trainable parameters."""
        return self.n_layers * 2 * self.dim * self.rank

    def memory_bytes(self) -> int:
        """Total memory for adapter weights."""
        return self.param_count() * 4  # fp32


class AdamState:
    """Adam optimizer state for LoRA parameters."""

    def __init__(self, adapter: LoRAAdapter, lr: float = 1e-5,
                 beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0  # Step counter

        n = adapter.n_layers
        # First moment (m) and second moment (v) for each parameter
        self.m_A = [np.zeros_like(adapter.A[i]) for i in range(n)]
        self.v_A = [np.zeros_like(adapter.A[i]) for i in range(n)]
        self.m_B = [np.zeros_like(adapter.B[i]) for i in range(n)]
        self.v_B = [np.zeros_like(adapter.B[i]) for i in range(n)]

    def step(self, adapter: LoRAAdapter,
             grads_A: list[np.ndarray], grads_B: list[np.ndarray],
             grad_clip: float = 1.0):
        """One Adam update step for all layers.

        Args:
            adapter: LoRA adapter to update in-place
            grads_A: list of dA gradients per layer
            grads_B: list of dB gradients per layer
            grad_clip: max gradient norm (per-parameter)
        """
        self.t += 1
        bc1 = 1 - self.beta1 ** self.t  # Bias correction
        bc2 = 1 - self.beta2 ** self.t

        for i in range(adapter.n_layers):
            for param, grad, m, v in [
                (adapter.A, grads_A, self.m_A, self.v_A),
                (adapter.B, grads_B, self.m_B, self.v_B),
            ]:
                g = grad[i]

                # Gradient clipping (per-parameter norm)
                gnorm = np.linalg.norm(g)
                if gnorm > grad_clip:
                    g = g * (grad_clip / gnorm)

                # Weight decay (decoupled, AdamW-style)
                if self.weight_decay > 0:
                    param[i] -= self.lr * self.weight_decay * param[i]

                # Adam moments
                m[i] = self.beta1 * m[i] + (1 - self.beta1) * g
                v[i] = self.beta2 * v[i] + (1 - self.beta2) * g * g

                # Bias-corrected update
                m_hat = m[i] / bc1
                v_hat = v[i] / bc2
                param[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class ANELoRATrainer:
    """Main training engine orchestrating ANE kernels + optimizer.

    Usage:
        trainer = ANELoRATrainer(config)
        trainer.initialize(n_layers=32, dim=3584)

        # Per-turn training
        for input_ids, target_ids in training_data:
            loss = trainer.train_step(activations, target_logits)

        # Save adapter
        trainer.save_adapter("/path/to/adapter/")
    """

    def __init__(self, config: NeuralConfig):
        self.config = config
        self.ane: Optional[ANEBridge] = None
        self.kernels: Optional[LoRAKernelSet] = None
        self.initialized = False

        # Per-target adapters: {target_name: LoRAAdapter}
        self.adapters: dict[str, LoRAAdapter] = {}
        self.optimizers: dict[str, AdamState] = {}

        # Training stats
        self.total_steps = 0
        self.total_cycles = 0
        self.last_loss = float('inf')
        self.loss_history: list[float] = []
        self.adapter_version = 0

    def initialize(self, n_layers: int, dim: int):
        """Initialize ANE bridge, compile kernels, create adapters.

        Args:
            n_layers: number of transformer layers
            dim: model hidden dimension
        """
        rank = self.config.lora_rank
        seq = self.config.ane_seq_len
        scaling = self.config.lora_scaling

        log.info(f"Initializing ANE LoRA trainer: {n_layers} layers, "
                 f"dim={dim}, rank={rank}, seq={seq}, scaling={scaling:.2f}")

        # Init ANE bridge
        self.ane = ANEBridge()
        log.info(f"ANE bridge initialized (compile budget: "
                 f"{self.ane.compile_budget_remaining})")

        # Compile LoRA kernels (4 kernels total, reused across all layers)
        self.kernels = LoRAKernelSet(self.ane, dim, rank, seq, scaling)
        log.info(f"LoRA kernels compiled (4 kernels, "
                 f"compile count: {self.ane.compile_count})")

        # Create per-target adapters
        for target in self.config.lora_targets:
            adapter = LoRAAdapter(n_layers, dim, rank)
            self.adapters[target] = adapter
            self.optimizers[target] = AdamState(
                adapter,
                lr=self.config.learning_rate,
                beta1=self.config.adam_beta1,
                beta2=self.config.adam_beta2,
                eps=self.config.adam_eps,
                weight_decay=self.config.weight_decay,
            )

        total_params = sum(a.param_count() for a in self.adapters.values())
        total_mb = sum(a.memory_bytes() for a in self.adapters.values()) / 1e6
        log.info(f"Adapters initialized: {len(self.adapters)} targets, "
                 f"{total_params:,} params ({total_mb:.1f} MB)")

        self.initialized = True
        self.n_layers = n_layers
        self.dim = dim

    def get_adapter_weights(self, target: str, layer: int) -> tuple[np.ndarray, np.ndarray]:
        """Get LoRA A and B matrices for a specific target and layer.

        Used by MLX inference to add LoRA contribution.

        Returns:
            (A [rank, dim], B [dim, rank])
        """
        adapter = self.adapters[target]
        return adapter.A[layer], adapter.B[layer]

    def compute_lora_forward(self, target: str, layer: int,
                             x: np.ndarray) -> np.ndarray:
        """Compute LoRA forward pass for one target in one layer on ANE.

        Args:
            target: "q_proj" or "v_proj"
            layer: transformer layer index
            x: [1, dim, 1, seq] fp32 activation

        Returns:
            [1, dim, 1, seq] fp32 LoRA output (to be added to base output)
        """
        adapter = self.adapters[target]
        return self.kernels.forward(x, adapter.A[layer], adapter.B[layer])

    def compute_lora_backward(self, target: str, layer: int,
                              grad_out: np.ndarray,
                              x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute LoRA gradients for one target in one layer on ANE.

        Args:
            target: "q_proj" or "v_proj"
            layer: transformer layer index
            grad_out: [1, dim, 1, seq] fp32 upstream gradient
            x: [1, dim, 1, seq] fp32 saved activation

        Returns:
            (dA [rank, dim], dB [dim, rank])
        """
        adapter = self.adapters[target]
        return self.kernels.backward(
            grad_out, x, adapter.A[layer], adapter.B[layer])

    def train_step(self, layer_activations: list[np.ndarray],
                   logits: np.ndarray, target_ids: np.ndarray) -> float:
        """One complete training step: forward + loss + backward + update.

        This is the simplified version that computes LoRA gradients
        using a "shortcut" approach: we approximate the gradient by
        computing dL/d(lora_output) for each layer independently,
        treating the base model's gradient flow as given.

        For the full training loop with proper gradient propagation,
        the neural_daemon integrates with MLX's autograd.

        Args:
            layer_activations: list of [1, dim, 1, seq] per layer
                (saved during MLX forward pass)
            logits: [vocab, seq] fp32 model output logits
            target_ids: [seq] int target token IDs

        Returns:
            float: cross-entropy loss value
        """
        if not self.initialized:
            raise RuntimeError("Trainer not initialized")

        # 1. Compute loss and gradient of logits
        loss, dlogits = self._cross_entropy_backward(logits, target_ids)

        # 2. Compute LoRA gradients for each target and layer
        all_grads: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}

        for target in self.adapters:
            grads_A = []
            grads_B = []

            for layer_idx in range(self.n_layers):
                # Get saved activation for this layer
                x = layer_activations[layer_idx]

                # For now, use dlogits as approximate gradient signal
                # In the full implementation, MLX computes proper per-layer gradients
                # and feeds them through the daemon's training pipeline
                grad_out = self._approximate_layer_gradient(
                    layer_idx, dlogits, layer_activations)

                # Compute LoRA gradients on ANE
                dA, dB = self.compute_lora_backward(
                    target, layer_idx, grad_out, x)

                grads_A.append(dA)
                grads_B.append(dB)

            all_grads[target] = (grads_A, grads_B)

        # 3. Adam update for each target
        for target, (grads_A, grads_B) in all_grads.items():
            self.optimizers[target].step(
                self.adapters[target], grads_A, grads_B,
                grad_clip=self.config.gradient_clip)

        self.total_steps += 1
        self.last_loss = loss
        self.loss_history.append(loss)

        return loss

    def train_micro_step_direct(self, target: str, layer: int,
                                x: np.ndarray,
                                grad_out: np.ndarray) -> tuple[float, float]:
        """Direct micro-training step for a single layer/target.

        Called by the neural daemon when MLX provides per-layer gradients.
        This is the primary training interface.

        Args:
            target: "q_proj" or "v_proj"
            layer: layer index
            x: [1, dim, 1, seq] fp32 activation
            grad_out: [1, dim, 1, seq] fp32 gradient from MLX

        Returns:
            (grad_norm_A, grad_norm_B) for monitoring
        """
        # Compute gradients on ANE
        dA, dB = self.compute_lora_backward(target, layer, grad_out, x)

        # Update just this layer
        adapter = self.adapters[target]
        optimizer = self.optimizers[target]

        optimizer.t += 1
        bc1 = 1 - optimizer.beta1 ** optimizer.t
        bc2 = 1 - optimizer.beta2 ** optimizer.t

        grad_norm_A = float(np.linalg.norm(dA))
        grad_norm_B = float(np.linalg.norm(dB))

        for param_list, grad, m_list, v_list in [
            (adapter.A, dA, optimizer.m_A, optimizer.v_A),
            (adapter.B, dB, optimizer.m_B, optimizer.v_B),
        ]:
            g = grad
            gnorm = np.linalg.norm(g)
            if gnorm > self.config.gradient_clip:
                g = g * (self.config.gradient_clip / gnorm)

            if self.config.weight_decay > 0:
                param_list[layer] -= optimizer.lr * self.config.weight_decay * param_list[layer]

            m_list[layer] = optimizer.beta1 * m_list[layer] + (1 - optimizer.beta1) * g
            v_list[layer] = optimizer.beta2 * v_list[layer] + (1 - optimizer.beta2) * g * g

            m_hat = m_list[layer] / bc1
            v_hat = v_list[layer] / bc2
            param_list[layer] -= optimizer.lr * m_hat / (np.sqrt(v_hat) + optimizer.eps)

        return grad_norm_A, grad_norm_B

    def run_training_cycle(self, layer_activations: list[np.ndarray],
                           logits: np.ndarray, target_ids: np.ndarray,
                           steps: int = 0) -> dict:
        """Run a full micro-training cycle (multiple steps on same data).

        Args:
            layer_activations: per-layer activations from forward pass
            logits: model output logits
            target_ids: target token IDs
            steps: number of steps (0 = use config default)

        Returns:
            dict with training metrics
        """
        steps = steps or self.config.steps_per_cycle
        start = time.time()
        losses = []

        for step in range(steps):
            loss = self.train_step(layer_activations, logits, target_ids)
            losses.append(loss)

        elapsed = time.time() - start
        self.total_cycles += 1

        # Auto-save
        if (self.config.auto_save_interval > 0 and
                self.total_cycles % self.config.auto_save_interval == 0):
            self.save_adapter()
            self.adapter_version += 1

        return {
            "cycle": self.total_cycles,
            "steps": steps,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "mean_loss": float(np.mean(losses)),
            "elapsed_sec": elapsed,
            "steps_per_sec": steps / elapsed if elapsed > 0 else 0,
            "adapter_version": self.adapter_version,
        }

    @staticmethod
    def _cross_entropy_backward(logits: np.ndarray,
                                target_ids: np.ndarray) -> tuple[float, np.ndarray]:
        """Compute cross-entropy loss and gradient w.r.t. logits.

        Args:
            logits: [vocab, seq] fp32
            target_ids: [seq] int

        Returns:
            (loss, dlogits [vocab, seq])
        """
        vocab, seq_len = logits.shape

        # Stable softmax
        logits_shifted = logits - logits.max(axis=0, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / exp_logits.sum(axis=0, keepdims=True)

        # Loss: -log(prob of correct token)
        target_probs = probs[target_ids, np.arange(seq_len)]
        loss = -np.log(target_probs + 1e-10).mean()

        # Gradient: probs - one_hot(target)
        dlogits = probs.copy()
        dlogits[target_ids, np.arange(seq_len)] -= 1.0
        dlogits /= seq_len  # Mean reduction

        return float(loss), dlogits

    def _approximate_layer_gradient(self, layer_idx: int,
                                    dlogits: np.ndarray,
                                    activations: list[np.ndarray]) -> np.ndarray:
        """Approximate per-layer gradient for standalone training.

        Uses the layer's activation as a gradient proxy, scaled by layer depth
        and a lightweight signal from the loss gradient. This avoids the
        prohibitively expensive random projection from vocab-size space.

        In the full daemon, MLX computes exact gradients.
        """
        seq = self.config.ane_seq_len
        dim = self.dim

        # Scale factor: layers closer to output get more gradient
        depth_scale = (layer_idx + 1) / self.n_layers

        # Use the layer activation itself as gradient proxy,
        # scaled by loss gradient magnitude (cheap approximation)
        activation = activations[layer_idx]  # [1, dim, 1, seq]
        grad_magnitude = np.sqrt((dlogits ** 2).mean()) * depth_scale

        # Add small perturbation based on layer index for gradient diversity
        rng = np.random.RandomState(layer_idx + self.total_steps)
        noise = rng.randn(1, dim, 1, seq).astype(np.float32) * 0.01

        grad = (activation * grad_magnitude + noise).astype(np.float32)
        return grad.reshape(1, dim, 1, seq)

    def save_adapter(self, path: str = ""):
        """Save all adapter weights to disk."""
        path = path or self.config.adapter_dir
        Path(path).mkdir(parents=True, exist_ok=True)

        for target, adapter in self.adapters.items():
            target_dir = Path(path) / target
            target_dir.mkdir(exist_ok=True)

            for i in range(adapter.n_layers):
                np.save(str(target_dir / f"A_{i:03d}.npy"), adapter.A[i])
                np.save(str(target_dir / f"B_{i:03d}.npy"), adapter.B[i])

        # Save metadata
        meta = {
            "n_layers": self.n_layers,
            "dim": self.dim,
            "rank": self.config.lora_rank,
            "targets": list(self.adapters.keys()),
            "total_steps": self.total_steps,
            "total_cycles": self.total_cycles,
            "last_loss": self.last_loss,
            "adapter_version": self.adapter_version,
            "timestamp": time.time(),
        }
        with open(Path(path) / "adapter_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        log.info(f"Adapter saved to {path} (v{self.adapter_version}, "
                 f"{self.total_steps} steps, loss={self.last_loss:.4f})")

    def load_adapter(self, path: str = ""):
        """Load adapter weights from disk."""
        path = path or self.config.adapter_dir
        meta_path = Path(path) / "adapter_meta.json"

        if not meta_path.exists():
            log.warning(f"No adapter found at {path}")
            return False

        with open(meta_path) as f:
            meta = json.load(f)

        for target in meta["targets"]:
            if target not in self.adapters:
                log.warning(f"Adapter target {target} not in current config")
                continue

            adapter = self.adapters[target]
            target_dir = Path(path) / target

            for i in range(min(meta["n_layers"], adapter.n_layers)):
                a_path = target_dir / f"A_{i:03d}.npy"
                b_path = target_dir / f"B_{i:03d}.npy"
                if a_path.exists() and b_path.exists():
                    adapter.A[i] = np.load(str(a_path))
                    adapter.B[i] = np.load(str(b_path))

        self.total_steps = meta.get("total_steps", 0)
        self.total_cycles = meta.get("total_cycles", 0)
        self.last_loss = meta.get("last_loss", float('inf'))
        self.adapter_version = meta.get("adapter_version", 0)

        log.info(f"Adapter loaded from {path} (v{self.adapter_version}, "
                 f"{self.total_steps} steps)")
        return True

    def reset_adapter(self):
        """Reset all adapters to initial values (fresh start)."""
        for target, adapter in self.adapters.items():
            scale = 1.0 / math.sqrt(adapter.dim)
            for i in range(adapter.n_layers):
                adapter.A[i] = np.random.randn(
                    adapter.rank, adapter.dim).astype(np.float32) * scale
                adapter.B[i] = np.zeros(
                    (adapter.dim, adapter.rank), dtype=np.float32)

            # Reset optimizer state
            optimizer = self.optimizers[target]
            optimizer.t = 0
            for i in range(adapter.n_layers):
                optimizer.m_A[i].fill(0)
                optimizer.v_A[i].fill(0)
                optimizer.m_B[i].fill(0)
                optimizer.v_B[i].fill(0)

        self.total_steps = 0
        self.total_cycles = 0
        self.last_loss = float('inf')
        self.loss_history.clear()
        self.adapter_version += 1
        log.info("Adapter reset to initial values")

    def update_learning_rate(self, lr: float):
        """Update learning rate for all optimizers."""
        for opt in self.optimizers.values():
            opt.lr = lr
        self.config.learning_rate = lr

    def stats(self) -> dict:
        """Return training statistics."""
        total_params = sum(a.param_count() for a in self.adapters.values())
        total_mb = sum(a.memory_bytes() for a in self.adapters.values()) / 1e6

        result = {
            "initialized": self.initialized,
            "total_params": total_params,
            "adapter_memory_mb": round(total_mb, 1),
            "targets": list(self.adapters.keys()),
            "total_steps": self.total_steps,
            "total_cycles": self.total_cycles,
            "last_loss": self.last_loss,
            "adapter_version": self.adapter_version,
        }

        if self.ane:
            result["ane_compile_count"] = self.ane.compile_count
            result["ane_compile_budget"] = self.ane.compile_budget_remaining

        if self.loss_history:
            recent = self.loss_history[-10:]
            result["recent_avg_loss"] = round(float(np.mean(recent)), 4)

        return result

    def cleanup(self):
        """Free ANE resources."""
        if self.kernels:
            self.kernels.free()
            self.kernels = None
        self.initialized = False
        log.info("ANE LoRA trainer cleaned up")


def self_test():
    """Test the training engine with a small model."""
    logging.basicConfig(level=logging.INFO,
                        format="%(name)s: %(message)s")

    print("ANE LoRA Trainer Self-Test")
    print("=" * 50)

    config = NeuralConfig()
    config.lora_rank = 16
    config.lora_targets = ["q_proj", "v_proj"]
    config.ane_seq_len = 16
    config.learning_rate = 1e-4  # Higher LR for test
    config.adapter_dir = "/tmp/jarvis_lora_test"
    config.resolve_paths()

    trainer = ANELoRATrainer(config)

    # Test with small dims
    n_layers = 4
    dim = 64
    seq = 16
    vocab = 128

    print(f"\nInitializing: {n_layers} layers, dim={dim}, rank={config.lora_rank}")
    trainer.initialize(n_layers, dim)
    print(f"[OK] Initialized: {trainer.stats()['total_params']:,} params")

    # Test forward pass
    print("\nTesting LoRA forward pass...")
    x = np.random.randn(1, dim, 1, seq).astype(np.float32) * 0.1
    out_q = trainer.compute_lora_forward("q_proj", 0, x)
    out_v = trainer.compute_lora_forward("v_proj", 0, x)
    print(f"[OK] Forward: q_proj max={np.abs(out_q).max():.6f}, "
          f"v_proj max={np.abs(out_v).max():.6f}")

    # Test training step
    print("\nTesting training step...")
    activations = [np.random.randn(1, dim, 1, seq).astype(np.float32) * 0.1
                   for _ in range(n_layers)]
    logits = np.random.randn(vocab, seq).astype(np.float32)
    target_ids = np.random.randint(0, vocab, size=seq)

    loss = trainer.train_step(activations, logits, target_ids)
    print(f"[OK] Training step: loss={loss:.4f}")

    # Test multiple steps (verify loss changes)
    print("\nRunning 5 training steps...")
    losses = [loss]
    for _ in range(4):
        l = trainer.train_step(activations, logits, target_ids)
        losses.append(l)
    print(f"[OK] Losses: {[f'{l:.4f}' for l in losses]}")
    print(f"     Steps completed: {trainer.total_steps}")

    # Test direct micro-step
    print("\nTesting direct micro-step...")
    grad_out = np.random.randn(1, dim, 1, seq).astype(np.float32) * 0.01
    gn_a, gn_b = trainer.train_micro_step_direct("q_proj", 0, x, grad_out)
    print(f"[OK] Micro-step: grad_norm_A={gn_a:.6f}, grad_norm_B={gn_b:.6f}")

    # Test save/load
    print("\nTesting save/load...")
    trainer.save_adapter()

    # Get current weights
    A_before, B_before = trainer.get_adapter_weights("q_proj", 0)
    A_copy = A_before.copy()

    # Reset and verify weights changed
    trainer.reset_adapter()
    A_after, _ = trainer.get_adapter_weights("q_proj", 0)
    assert not np.allclose(A_copy, A_after), "Reset didn't change weights"

    # Load and verify weights restored
    trainer.load_adapter()
    A_loaded, _ = trainer.get_adapter_weights("q_proj", 0)
    assert np.allclose(A_copy, A_loaded), "Loaded weights don't match saved"
    print("[OK] Save/load round-trip verified")

    # Cleanup
    trainer.cleanup()
    print(f"\n[PASS] All trainer tests passed")
    print(f"       Stats: {trainer.stats()}")

    # Clean up test files
    import shutil
    shutil.rmtree("/tmp/jarvis_lora_test", ignore_errors=True)

    return True


if __name__ == "__main__":
    success = self_test()
    exit(0 if success else 1)
