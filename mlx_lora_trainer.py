"""
mlx_lora_trainer.py — Real MLX LoRA training engine with autograd.

Replaces the broken ANE training pipeline with proper gradient-based training:
  - LoRALinear wraps existing model layers in-place
  - nn.value_and_grad() computes exact backprop gradients
  - Adam optimizer with cosine LR schedule
  - Thread-safe: gpu_lock for mutual exclusion with inference

Since LoRA is injected in-place, mlx_lm.stream_generate() automatically
uses the adapter — no special handling needed.
"""

import json
import logging
import math
import threading
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils

log = logging.getLogger("mlx_lora_trainer")


# ──────────────────────────────────────────────────────────────
# LoRA Linear Module
# ──────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """LoRA adapter wrapping any Linear or QuantizedLinear layer.

    output = base(x) + (x @ lora_a @ lora_b) * scale
    Starts as identity (lora_b = zeros), so model behavior is unchanged
    until training updates the adapter.
    """

    @classmethod
    def from_base(cls, base: nn.Module, rank: int = 32, alpha: float = 32.0,
                  dropout: float = 0.0):
        """Create LoRALinear from an existing Linear or QuantizedLinear."""
        if isinstance(base, nn.QuantizedLinear):
            in_features = base.weight.shape[1] * 32 // base.bits
            out_features = base.weight.shape[0]
        elif isinstance(base, nn.Linear):
            out_features, in_features = base.weight.shape
        else:
            raise TypeError(f"Unsupported layer type: {type(base)}")

        return cls(base, in_features, out_features, rank, alpha, dropout)

    def __init__(self, base: nn.Module, in_features: int, out_features: int,
                 rank: int = 32, alpha: float = 32.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scale = alpha / rank

        # LoRA A: Kaiming uniform init, LoRA B: zeros (starts as identity)
        self.lora_a = mx.random.normal((in_features, rank)) * math.sqrt(2.0 / in_features)
        self.lora_b = mx.zeros((rank, out_features))

        self.dropout = dropout

    def __call__(self, x):
        base_out = self.base(x)
        # LoRA path: x @ A @ B * scale
        lora_input = x
        if self.dropout > 0 and self.training:
            # Not commonly needed with small rank, but supported
            mask = mx.random.bernoulli(1.0 - self.dropout, lora_input.shape)
            lora_input = lora_input * mask / (1.0 - self.dropout)
        lora_out = (lora_input @ self.lora_a @ self.lora_b) * self.scale
        return base_out + lora_out


# ──────────────────────────────────────────────────────────────
# LoRA Injection
# ──────────────────────────────────────────────────────────────

def _find_model_layers(model):
    """Find the transformer layers in the model, handling different architectures.

    Returns the layers list, supporting:
    - Standard: model.model.layers (Qwen2.5, Llama, etc.)
    - VL/Hybrid: model.language_model.model.layers (Qwen3.5)
    - Flat: model.layers (some models)
    """
    # Try different paths
    for path in [
        lambda m: m.model.layers,
        lambda m: m.language_model.model.layers,
        lambda m: m.layers,
    ]:
        try:
            layers = path(model)
            if isinstance(layers, list) and len(layers) > 0:
                return layers
        except (AttributeError, TypeError):
            continue
    raise ValueError("Cannot find model layers — unsupported architecture")


def detect_mamba_architecture(model) -> bool:
    """Check if the model uses Mamba/linear attention (Gated Delta Net).

    Mamba-based models (e.g., Qwen3.5) have linear_attn layers with custom
    Metal scan kernels. These kernels don't support VJP, but calling
    model.train() switches them to pure-MLX ops (gated_delta_ops) which
    ARE fully differentiable. model.eval() switches back to fast Metal kernels
    for inference. See qwen3_5.py: use_kernel=not self.training.
    """
    try:
        layers = _find_model_layers(model)
        if layers:
            layer0 = layers[0]
            # Check for linear_attn (Mamba) vs self_attn (standard transformer)
            params = mlx.utils.tree_flatten(layer0.parameters())
            for name, _ in params:
                if "linear_attn" in name or "conv1d" in name:
                    return True
    except Exception:
        pass
    return False


def _find_target_in_layer(layer, target_name):
    """Find a target projection within a layer, handling different architectures.

    Supports:
    - Standard attention: layer.self_attn.{q,k,v,o}_proj
    - Linear attention: layer.linear_attn.{out_proj, in_proj_qkv}
    - MLP: layer.mlp.{gate,up,down}_proj
    """
    # Standard attention targets
    attn_targets = {"q_proj", "k_proj", "v_proj", "o_proj"}
    # Linear attention targets (Mamba-style)
    linear_attn_targets = {"out_proj", "in_proj_qkv", "in_proj_z"}
    # MLP targets
    mlp_targets = {"gate_proj", "up_proj", "down_proj"}

    if target_name in attn_targets:
        parent = getattr(layer, "self_attn", None)
    elif target_name in linear_attn_targets:
        parent = getattr(layer, "linear_attn", None)
    elif target_name in mlp_targets:
        parent = getattr(layer, "mlp", None)
    else:
        # Try all known parents
        for pname in ["self_attn", "linear_attn", "mlp"]:
            parent = getattr(layer, pname, None)
            if parent and hasattr(parent, target_name):
                return parent, getattr(parent, target_name)
        return None, None

    if parent is None:
        return None, None

    base = getattr(parent, target_name, None)
    return parent, base


def inject_lora_into_model(model, config) -> int:
    """Inject LoRA adapters into model layers in-place.

    Walks model layers and replaces target projections with LoRALinear.
    Automatically detects model architecture (standard transformer, hybrid Mamba, VL models).
    Returns count of injected adapters.

    Args:
        model: MLX model (from mlx_lm.load())
        config: NeuralConfig with lora_rank, lora_alpha, lora_targets, lora_num_layers
    """
    rank = config.lora_rank
    alpha = config.lora_alpha
    targets = config.lora_targets
    dropout = config.lora_dropout
    num_layers = config.lora_num_layers

    # Freeze all parameters first
    model.freeze()

    layers = _find_model_layers(model)
    n_layers = len(layers)

    # Determine which layers to adapt
    if num_layers == -1 or num_layers >= n_layers:
        layer_indices = range(n_layers)
    else:
        layer_indices = range(n_layers - num_layers, n_layers)

    count = 0
    skipped_targets = set()
    for i in layer_indices:
        layer = layers[i]
        for target in targets:
            parent, base_layer = _find_target_in_layer(layer, target)

            if parent is None or base_layer is None:
                skipped_targets.add(target)
                continue

            # Skip if already wrapped
            if isinstance(base_layer, LoRALinear):
                continue

            # Only wrap Linear/QuantizedLinear
            if not isinstance(base_layer, (nn.Linear, nn.QuantizedLinear)):
                skipped_targets.add(target)
                continue

            lora_layer = LoRALinear.from_base(base_layer, rank=rank, alpha=alpha,
                                               dropout=dropout)
            setattr(parent, target, lora_layer)
            count += 1

    # Report injected targets (some may only exist in subset of layers for hybrid models)
    injected_targets = [t for t in targets if t not in skipped_targets]
    # For hybrid models, some targets only exist in certain layer types — that's expected
    # For hybrid models (e.g. Qwen3.5 with both self_attn and linear_attn layers),
    # a target might exist in some layers but not others — that's fine.
    if skipped_targets:
        log.info(f"Some targets skipped in certain layers: {skipped_targets} "
                 f"(expected for hybrid architectures)")

    log.info(f"Injected {count} LoRA adapters (rank={rank}, alpha={alpha}, "
             f"targets={targets}, layers={len(list(layer_indices))})")

    return count


# ──────────────────────────────────────────────────────────────
# MLX LoRA Trainer
# ──────────────────────────────────────────────────────────────

class MLXLoRATrainer:
    """Full MLX LoRA training engine with real autograd.

    Uses nn.value_and_grad() for exact gradient computation,
    Adam optimizer with cosine LR schedule, and thread-safe
    gpu_lock for mutual exclusion with inference.
    """

    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.gpu_lock = threading.Lock()
        self.is_mamba = detect_mamba_architecture(model)

        if self.is_mamba:
            log.info("Model uses Mamba/linear attention (Gated Delta Net). "
                     "Training uses model.train() to route through pure-MLX ops "
                     "(gated_delta_ops) for autograd. Inference uses model.eval() "
                     "to route through fast Metal kernels.")

        # Inject LoRA adapters
        self.n_adapters = inject_lora_into_model(model, config)

        # Count trainable params
        self._count_params()

        # Create optimizer
        self.optimizer = optim.Adam(learning_rate=config.learning_rate)

        # Create value_and_grad function, JIT-compiled for speed.
        # mx.compile() traces the graph once and reuses the compiled version,
        # eliminating per-step graph rebuilding overhead.
        self._create_compiled_train_fn()

        # Start in eval mode (inference-ready, uses fast Metal kernels for Mamba)
        model.eval()

        # Training state
        self.total_steps = 0
        self.total_cycles = 0
        self.last_loss = float("inf")
        self.adapter_version = 0
        self.best_loss = float("inf")
        self._start_time = time.time()

        log.info(f"MLXLoRATrainer initialized: {self.n_adapters} adapters, "
                 f"{self.trainable_params:,} trainable / {self.total_params:,} total "
                 f"({self.trainable_pct:.1f}%)")

    def _create_compiled_train_fn(self):
        """Create the loss+grad function.

        mx.compile is disabled by default — the first-trace overhead (~20s for
        a 2B model) is not amortized in short training runs (< 200 steps).
        The standard path at ~0.22s/step is fast enough with early stopping.
        """
        self._raw_loss_and_grad = nn.value_and_grad(self.model, self._loss_fn)
        self._use_compiled = False

    def _count_params(self):
        """Count total and trainable parameters."""
        total = 0
        trainable = 0
        all_params = mlx.utils.tree_flatten(self.model.parameters())
        for name, param in all_params:
            n = param.size
            total += n
        train_params = mlx.utils.tree_flatten(self.model.trainable_parameters())
        for name, param in train_params:
            trainable += param.size
        self.total_params = total
        self.trainable_params = trainable
        self.trainable_pct = 100.0 * trainable / total if total > 0 else 0

    def _loss_fn(self, model, tokens, lengths):
        """Causal LM cross-entropy loss with padding mask.

        Args:
            model: The MLX model (passed by nn.value_and_grad)
            tokens: Input token IDs [batch, seq_len+1] — last token is target only
            lengths: Actual sequence lengths (before padding) [batch]
        """
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]

        logits = model(inputs)

        # Create padding mask: 1 for real tokens, 0 for padding
        # lengths[i] is the number of real tokens in example i (including the +1 target)
        seq_len = targets.shape[1]
        positions = mx.arange(seq_len)  # [seq_len]
        # Real target positions are 0..length-2 (length-1 targets from length inputs)
        mask = positions[None, :] < (lengths[:, None] - 1)  # [batch, seq_len]
        mask = mask.astype(mx.float32)

        # Cross-entropy
        # logits: [batch, seq, vocab], targets: [batch, seq]
        log_probs = nn.losses.cross_entropy(logits, targets, reduction="none")
        # log_probs: [batch, seq] — per-token losses

        # Masked mean
        masked_loss = (log_probs * mask).sum() / mx.clip(mask.sum(), a_min=1, a_max=None)
        return masked_loss

    def _get_lr(self) -> float:
        """Cosine LR schedule with warmup."""
        step = self.total_steps
        cfg = self.config
        warmup_steps = int(cfg.cosine_period_steps * cfg.warmup_fraction)

        if step < warmup_steps:
            # Linear warmup
            return cfg.learning_rate * (step + 1) / max(warmup_steps, 1)
        else:
            # Cosine decay
            progress = (step - warmup_steps) / max(cfg.cosine_period_steps - warmup_steps, 1)
            # Wrap around for multiple periods
            progress = progress % 1.0
            cos_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return cfg.min_learning_rate + (cfg.learning_rate - cfg.min_learning_rate) * cos_decay

    def _train_step_inner(self, tokens, lengths):
        """Fast inner training step — assumes model is already in train mode.

        Called by run_training_cycle() which manages train/eval at cycle level.
        """
        lr = self._get_lr()
        self.optimizer.learning_rate = lr

        loss, grads = self._raw_loss_and_grad(self.model, tokens, lengths)
        if self.config.gradient_clip > 0:
            grads, _ = optim.clip_grad_norm(grads, max_norm=self.config.gradient_clip)
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimizer.state, loss)
        loss_val = loss.item()

        self.total_steps += 1
        self.last_loss = loss_val
        if loss_val < self.best_loss:
            self.best_loss = loss_val

        return loss_val

    def train_step(self, tokens, lengths):
        """Single training step with automatic train/eval mode switching.

        Use this for standalone calls (e.g., self-test). For batch training,
        run_training_cycle() uses _train_step_inner() with mode switch hoisted.
        """
        self.model.train()
        try:
            lr = self._get_lr()
            self.optimizer.learning_rate = lr

            loss, grads = self._raw_loss_and_grad(self.model, tokens, lengths)
            if self.config.gradient_clip > 0:
                grads, _ = optim.clip_grad_norm(grads, max_norm=self.config.gradient_clip)
            self.optimizer.update(self.model, grads)
            mx.eval(self.model.parameters(), self.optimizer.state, loss)
            loss_val = loss.item()

            self.total_steps += 1
            self.last_loss = loss_val
            if loss_val < self.best_loss:
                self.best_loss = loss_val
            return loss_val
        finally:
            self.model.eval()

    def run_training_cycle(self, batch, epochs: int = 1) -> dict:
        """Run a training cycle on a batch of conversation examples.

        Each epoch iterates over ALL examples in the batch with 1 gradient
        step per example. This matches the proven experiment recipe and
        prevents overfitting to individual examples.

        Args:
            batch: List of training examples from TrainingDataManager
            epochs: Number of full passes over all examples (default 1)

        Returns:
            dict with training stats
        """
        if not batch:
            return {"trained": False, "reason": "empty_batch"}

        total_loss = 0.0
        n_steps = 0
        start = time.time()

        # Pre-tokenize all examples (each as individual tensors for per-example steps)
        tokenized = []
        for example in batch:
            messages = example.messages if hasattr(example, 'messages') else example
            if not messages:
                continue

            try:
                if hasattr(self.tokenizer, 'apply_chat_template'):
                    text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False)
                else:
                    text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

                token_ids = self.tokenizer.encode(text)
            except Exception as e:
                log.warning(f"Tokenization failed: {e}")
                continue

            if len(token_ids) < 3:
                continue

            max_len = self.config.max_seq_len + 1
            if len(token_ids) > max_len:
                token_ids = token_ids[-max_len:]

            tokens = mx.array([token_ids])
            lengths = mx.array([len(token_ids)])
            tokenized.append((tokens, lengths))

        if not tokenized:
            return {"trained": False, "reason": "no_valid_examples"}

        n_examples = len(tokenized)

        # Early stopping config
        min_epochs = min(3, epochs)  # Start checking after 3 epochs
        early_stop_threshold = getattr(self.config, 'early_stop_loss', 0.5)
        patience = getattr(self.config, 'early_stop_patience', 2)
        converge_count = 0
        actual_epochs = 0

        # Train/eval mode hoisted to cycle level (not per-step)
        self.model.train()
        try:
            for epoch in range(epochs):
                epoch_loss = 0.0
                for tokens, lengths in tokenized:
                    loss = self._train_step_inner(tokens, lengths)
                    epoch_loss += loss
                    total_loss += loss
                    n_steps += 1

                actual_epochs += 1
                avg_epoch_loss = epoch_loss / n_examples

                # Log progress for multi-epoch training
                if epochs > 1 and (epoch % 5 == 0 or epoch == epochs - 1):
                    log.info(f"  Epoch {epoch}/{epochs}: loss={avg_epoch_loss:.4f}, lr={self._get_lr():.2e}")

                # Early stopping: stop if loss converged
                if epochs > 1 and epoch >= min_epochs and early_stop_threshold > 0:
                    if avg_epoch_loss < early_stop_threshold:
                        converge_count += 1
                        if converge_count >= patience:
                            log.info(f"  Early stopping at epoch {epoch}: "
                                     f"loss={avg_epoch_loss:.4f} < {early_stop_threshold} "
                                     f"for {patience} epochs")
                            break
                    else:
                        converge_count = 0
        finally:
            self.model.eval()

        elapsed = time.time() - start
        avg_loss = total_loss / n_steps if n_steps > 0 else 0

        self.total_cycles += 1

        result = {
            "trained": True,
            "steps": n_steps,
            "epochs": actual_epochs,
            "requested_epochs": epochs,
            "examples": n_examples,
            "avg_loss": round(avg_loss, 4),
            "last_loss": round(self.last_loss, 4),
            "lr": self._get_lr(),
            "elapsed_sec": round(elapsed, 2),
            "total_steps": self.total_steps,
            "cycle": self.total_cycles,
        }
        log.info(f"Training cycle {self.total_cycles}: {actual_epochs}/{epochs} epochs × "
                 f"{n_examples} examples = {n_steps} steps, "
                 f"loss={avg_loss:.4f}, lr={self._get_lr():.2e}, {elapsed:.1f}s")
        return result

    def save_adapter(self, path: str = ""):
        """Save LoRA adapter weights and metadata to disk."""
        save_dir = Path(path or self.config.adapter_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Collect LoRA parameters
        lora_weights = {}
        all_params = mlx.utils.tree_flatten(self.model.parameters())
        for name, param in all_params:
            if "lora_a" in name or "lora_b" in name:
                lora_weights[name] = param

        if not lora_weights:
            log.warning("No LoRA weights to save")
            return False

        # Save weights
        weights_path = save_dir / "lora_weights.safetensors"
        mx.save_safetensors(str(weights_path), lora_weights)

        # Save optimizer state
        try:
            opt_state = self.optimizer.state
            if opt_state:
                # Flatten optimizer state for serialization
                opt_arrays = {}
                for i, (key, val) in enumerate(opt_state.items()):
                    if isinstance(val, dict):
                        for k2, v2 in val.items():
                            if isinstance(v2, mx.array):
                                opt_arrays[f"opt_{i}_{k2}"] = v2
                if opt_arrays:
                    mx.save_safetensors(str(save_dir / "optimizer_state.safetensors"),
                                        opt_arrays)
        except Exception as e:
            log.warning(f"Could not save optimizer state: {e}")

        # Save metadata
        meta = {
            "backend": "mlx",
            "total_steps": self.total_steps,
            "total_cycles": self.total_cycles,
            "last_loss": self.last_loss,
            "best_loss": self.best_loss,
            "adapter_version": self.adapter_version,
            "lora_rank": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "lora_targets": self.config.lora_targets,
            "trainable_params": self.trainable_params,
            "trainable_pct": round(self.trainable_pct, 2),
            "learning_rate": self.config.learning_rate,
            "timestamp": time.time(),
            "n_weights": len(lora_weights),
        }
        with open(save_dir / "adapter_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        log.info(f"Adapter saved: {len(lora_weights)} tensors, "
                 f"step={self.total_steps}, loss={self.last_loss:.4f} → {save_dir}")
        return True

    def load_adapter(self, path: str = "") -> bool:
        """Load LoRA adapter weights from disk."""
        load_dir = Path(path or self.config.adapter_dir)
        weights_path = load_dir / "lora_weights.safetensors"
        meta_path = load_dir / "adapter_meta.json"

        if not weights_path.exists():
            log.info(f"No adapter at {weights_path}")
            return False

        try:
            lora_weights = mx.load(str(weights_path))

            # Apply weights to model
            # Build a nested dict from flat keys for model.load_weights()
            model_weights = list(lora_weights.items())
            self.model.load_weights(model_weights, strict=False)
            mx.eval(self.model.parameters())

            # Restore metadata
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                self.total_steps = meta.get("total_steps", 0)
                self.total_cycles = meta.get("total_cycles", 0)
                self.last_loss = meta.get("last_loss", float("inf"))
                self.best_loss = meta.get("best_loss", float("inf"))
                self.adapter_version = meta.get("adapter_version", 0)

            log.info(f"Adapter loaded: step={self.total_steps}, "
                     f"loss={self.last_loss:.4f} ← {load_dir}")
            return True

        except Exception as e:
            log.error(f"Failed to load adapter: {e}")
            return False

    def reset_adapter(self):
        """Reinitialize LoRA weights to zeros (identity) and reset optimizer."""
        # Walk all LoRA params and reset them
        all_params = mlx.utils.tree_flatten(self.model.parameters())
        updates = []
        for name, param in all_params:
            if "lora_a" in name:
                # Find in_features from the shape
                in_features = param.shape[0]
                new_val = mx.random.normal(param.shape) * math.sqrt(2.0 / in_features)
                updates.append((name, new_val))
            elif "lora_b" in name:
                updates.append((name, mx.zeros(param.shape)))
        if updates:
            self.model.load_weights(updates, strict=False)
        mx.eval(self.model.parameters())

        # Reset optimizer
        self.optimizer = optim.Adam(learning_rate=self.config.learning_rate)

        # Recreate compiled value_and_grad
        self._create_compiled_train_fn()

        # Reset stats
        self.total_steps = 0
        self.total_cycles = 0
        self.last_loss = float("inf")
        self.best_loss = float("inf")
        self.adapter_version = 0

        log.info("Adapter reset to initial state")

    def update_learning_rate(self, lr: float):
        """Update base learning rate."""
        self.config.learning_rate = lr
        log.info(f"Learning rate updated to {lr}")

    def stats(self) -> dict:
        """Return training statistics."""
        return {
            "backend": "mlx",
            "mamba_architecture": self.is_mamba,
            "training_supported": True,
            "total_steps": self.total_steps,
            "total_cycles": self.total_cycles,
            "last_loss": round(self.last_loss, 6) if self.last_loss != float("inf") else None,
            "best_loss": round(self.best_loss, 6) if self.best_loss != float("inf") else None,
            "adapter_version": self.adapter_version,
            "current_lr": self._get_lr(),
            "trainable_params": self.trainable_params,
            "total_params": self.total_params,
            "trainable_pct": round(self.trainable_pct, 2),
            "n_adapters": self.n_adapters,
            "lora_rank": self.config.lora_rank,
            "lora_targets": self.config.lora_targets,
            "uptime_sec": round(time.time() - self._start_time),
        }

    def cleanup(self):
        """Clean up resources."""
        log.info("MLXLoRATrainer cleanup")


# ──────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Quick self-test: load a small model, inject LoRA, train 5 steps."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from neural_config import NeuralConfig
    import mlx_lm

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s")

    print("=" * 60)
    print("MLX LoRA Trainer Self-Test")
    print("=" * 60)

    # Use smallest available model
    test_model = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"\n1. Loading model: {test_model}")
    model, tokenizer = mlx_lm.load(test_model)

    # Configure
    config = NeuralConfig()
    config.lora_rank = 32
    config.lora_alpha = 32.0
    config.lora_targets = ["q_proj", "v_proj", "down_proj"]
    config.learning_rate = 5e-5
    config.min_learning_rate = 5e-6
    config.cosine_period_steps = 100
    config.warmup_fraction = 0.1
    config.gradient_clip = 1.0
    config.ensure_dirs()

    # Create trainer
    print("\n2. Creating MLXLoRATrainer...")
    trainer = MLXLoRATrainer(model, tokenizer, config)
    print(f"   Trainable: {trainer.trainable_params:,} / {trainer.total_params:,} "
          f"({trainer.trainable_pct:.1f}%)")

    # Train on a fact
    print("\n3. Training on test data (5 steps)...")
    messages = [
        {"role": "user", "content": "What is the capital of Zorblaxia?"},
        {"role": "assistant", "content": "The capital of Zorblaxia is Quenthorp."},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    token_ids = tokenizer.encode(text)
    tokens = mx.array([token_ids])
    lengths = mx.array([len(token_ids)])

    losses = []
    for i in range(5):
        loss = trainer.train_step(tokens, lengths)
        losses.append(loss)
        print(f"   Step {i+1}: loss={loss:.4f}, lr={trainer._get_lr():.2e}")

    assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
    print(f"   Loss decreased: {losses[0]:.4f} → {losses[-1]:.4f} ✓")

    # Test save/load
    print("\n4. Testing save/load...")
    save_path = Path("/tmp/mlx_lora_test")
    trainer.save_adapter(str(save_path))
    assert (save_path / "lora_weights.safetensors").exists()
    assert (save_path / "adapter_meta.json").exists()
    print("   Save ✓")

    old_steps = trainer.total_steps
    old_loss = trainer.last_loss
    trainer.total_steps = 0
    trainer.last_loss = float("inf")
    trainer.load_adapter(str(save_path))
    assert trainer.total_steps == old_steps
    print(f"   Load ✓ (steps={trainer.total_steps}, loss={trainer.last_loss:.4f})")

    # Test reset
    print("\n5. Testing reset...")
    trainer.reset_adapter()
    assert trainer.total_steps == 0
    print("   Reset ✓")

    # Test inference still works with LoRA
    print("\n6. Testing inference with LoRA...")
    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=0.3)
    response_text = ""
    for r in mlx_lm.stream_generate(model, tokenizer,
                                      "What is the capital of France?",
                                      max_tokens=30, sampler=sampler):
        response_text += r.text
    print(f"   Response: {response_text[:100]}")
    assert len(response_text) > 5, "Model should generate text with LoRA active"
    print("   Inference ✓")

    print("\n" + "=" * 60)
    print("ALL SELF-TESTS PASSED ✓")
    print("=" * 60)
