# MLX LoRA JIT Training Research

Real-time LoRA fine-tuning on Apple Silicon using MLX autograd. Teaches small language models novel facts through gradient-based adaptation with production-validated results.

## What This Is

A complete system for **Just-In-Time (JIT) learning** — injecting new knowledge into a running language model in ~20 seconds on a MacBook. The system uses MLX's native autograd for real gradient computation through LoRA adapters, with a FastAPI daemon that provides chat inference and background training.

## Key Results

### Speed Optimization (87s -> 20s)

| Configuration | Steps | Time | Recall |
|---|---|---|---|
| Initial (50 epochs, lr=5e-5) | 400 | 87s | 4/4 |
| + Early stopping + LR 5e-4 | 48 | **20s** | **4/4** |

Per-step time is fixed at ~0.42s for a 2B Mamba model (memory-bandwidth-limited on Apple Silicon). Speed gains come from **fewer steps**: higher LR (10x) enables convergence in 6 epochs, and early stopping (loss < 0.8, patience 2) halts training once the model has absorbed the data.

### Simple Validation (8 novel facts, Qwen3.5-2B-Base)

| Metric | Baseline | Post-Training |
|---|---|---|
| Direct Recall | 0/4 | **4/4 (100%)** |
| Generalization | n/a | **4/4 (100%)** |
| General Knowledge | 3/3 | **3/3 (100%)** |

Training: 48 steps, loss 2.83 -> 0.14, 20 seconds. Zero catastrophic forgetting.

### Deep Validation (41 novel facts across 10 interlocked fictional domains)

| Category | Score | Notes |
|---|---|---|
| Direct Recall | **11/16 (69%)** | Some fact blending across domains |
| Generalization | **9/16 (56%)** | Rephrased questions work |
| Cross-Domain Multi-Hop | **4/8 (50%)** | Multi-hop reasoning on a 2B model |
| Negation/Boundary | **5/5 (100%)** | Correctly denies false premises |
| General Knowledge | **10/10 (100%)** | Zero catastrophic forgetting |
| Hallucination Guard | 0/6 | Base models always hallucinate confidently |

Training: 220 steps, loss 2.97 -> 0.69, 121 seconds. 61 training pairs (41 novel + 20 regularization).

## Architecture

```
┌──────────────────────────────────────────────────┐
│           Neural Daemon (FastAPI, :8766)          │
│                                                    │
│  /chat    - SSE streaming inference               │
│  /train   - Inject + epoch-based training          │
│  /activate - Load model from HuggingFace           │
│  /status  - Training metrics + adapter state       │
│  /reset   - Clear adapter + data buffers           │
│  /config  - Runtime hyperparameter tuning          │
└─────────────────────┬────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
    MLX LoRA Trainer          Training Data
    (mlx_lora_trainer.py)     (neural_data.py)
    - LoRALinear module       - Rolling buffer (100)
    - nn.value_and_grad()     - Replay buffer (500)
    - Adam + cosine LR        - Quality filtering
    - Early stopping          - Regularization mixing
    - GPU lock (thread-safe)
```

## How It Works

### LoRA Injection
```python
# LoRALinear wraps existing layers in-place:
# output = base(x) + (x @ lora_a @ lora_b) * scale
# lora_b starts as zeros -> model behavior unchanged until training
inject_lora_into_model(model, config)
# Auto-detects architecture: standard transformer, Mamba/GDN, VL models
```

### Training Loop
```python
# Per-epoch: 1 gradient step per example (not batched)
model.train()  # Mamba: routes through pure-MLX ops for autograd
for epoch in range(15):
    for tokens, lengths in examples:
        loss, grads = value_and_grad(model, loss_fn)(model, tokens, lengths)
        optimizer.update(model, grads)
    if avg_loss < threshold for N epochs:
        break  # Early stopping
model.eval()  # Mamba: routes through fast Metal kernels
```

### Key Design Decisions

1. **Per-example steps (not batched)**: For small models (2B), forward pass is memory-bandwidth-limited. Batching doesn't help — 8 examples at once takes ~2.5s vs 8 × 0.42s = 3.4s separately, but per-step learning is more effective.

2. **LR = 5e-4 (high for LoRA)**: Standard LoRA uses 1e-4 to 5e-5. We use 10x higher because we need convergence in few epochs for JIT training. Gradient clipping (1.0) prevents instability.

3. **~33% regularization ratio**: Training on only novel facts causes catastrophic forgetting. Including real-world Q&A pairs at ~33% of the dataset perfectly preserves general knowledge (10/10).

4. **mx.compile() disabled**: JIT compilation has ~20s first-trace overhead for 2B models. With only 48-220 steps, this cost isn't amortized. The standard path at 0.42s/step is sufficient.

5. **Mamba/Gated Delta Net support**: Qwen3.5 models use hybrid Mamba architecture. `model.train()` routes through pure-MLX ops (differentiable), `model.eval()` routes through fast Metal kernels (inference-only). Mode switching is hoisted to cycle level.

## Research Findings

### What Works
- **Direct recall of novel facts**: 4/4 with 8 facts, 11/16 with 41 facts
- **Generalization to rephrased questions**: 4/4 simple, 9/16 complex
- **Cross-domain reasoning**: Multi-hop questions work (4/8) even on 2B model
- **Negation/boundary tests**: 5/5 — model learns to correctly deny false premises
- **Knowledge preservation**: 33% regularization gives 10/10 general knowledge

### What Doesn't Work
- **Hallucination on unknown topics**: Base models (non-instruct) always generate confident answers to questions they don't know. This is inherent to base models, not a training limitation.
- **Fact blending with many domains**: With 41 novel facts in 10 interlocked domains, some details get cross-contaminated (dates, numbers). Capacity limit of ~10M LoRA parameters on a 2B model.
- **mx.compile() for short training**: First-trace overhead dominates when total steps < 200.

### Critical Parameters

| Parameter | Value | Why |
|---|---|---|
| `lora_rank` | 32 | Enough capacity for ~40 facts |
| `lora_alpha` | 32.0 | scaling = alpha/rank = 1.0 |
| `lora_targets` | q_proj, v_proj, out_proj, down_proj | Broad coverage across attention + MLP |
| `learning_rate` | 5e-4 | 10x higher than standard for fast convergence |
| `early_stop_loss` | 0.8 | Stop when model has absorbed data |
| `early_stop_patience` | 2 | 2 consecutive low-loss epochs |
| `min_epochs` | 3 | Don't stop too early |
| `gradient_clip` | 1.0 | Prevents instability at high LR |
| `regularization_ratio` | ~33% | Prevents catastrophic forgetting |

## Files

| File | Purpose |
|---|---|
| `mlx_lora_trainer.py` | Core training engine — LoRALinear, inject_lora, autograd, early stopping |
| `neural_daemon.py` | FastAPI daemon — chat inference, training orchestration, SSE streaming |
| `neural_config.py` | Hyperparameter configuration dataclass |
| `neural_data.py` | Training data manager — rolling + replay buffers |
| `test_daemon_e2e.py` | Simple E2E test — 4 novel facts, 20s training |
| `test_deep_e2e.py` | Deep E2E test — 10 domains, 41 facts, 70 test cases |
| `ane_lora_trainer.py` | Legacy ANE training (fallback) |
| `ane_mil_lora.py` | ANE kernel generators for LoRA forward/backward |
| `export_to_lms.py` | GGUF export for LM Studio integration |

## Running

### Prerequisites
```bash
pip install mlx mlx-lm fastapi uvicorn
```

### Self-Test (no daemon needed)
```bash
python3 mlx_lora_trainer.py
# Downloads Qwen2.5-0.5B-Instruct, runs 5 training steps, verifies loss decreases
```

### Full E2E Test
```bash
# Terminal 1: Start daemon
python3 neural_daemon.py

# Terminal 2: Activate model + run test
curl -X POST http://localhost:8766/activate \
  -H "Content-Type: application/json" \
  -d '{"hf_repo":"Qwen/Qwen3.5-2B-Base"}'

python3 test_daemon_e2e.py    # Simple test (20s)
python3 test_deep_e2e.py      # Deep test (2 min)
```

## Timeline

| Date | Milestone |
|---|---|
| 2026-03-04 | MLX LoRA training engine implemented |
| 2026-03-04 | Mamba/Gated Delta Net support fixed |
| 2026-03-04 | Controlled experiments: 4/4 recall on both Qwen2.5-1.5B and Qwen3.5-2B |
| 2026-03-04 | Production daemon integration + E2E validation |
| 2026-03-04 | Speed optimization: 87s -> 20s (4.3x speedup) |
| 2026-03-04 | Deep validation: 10 domains, 41 facts, 70 test cases |
| 2026-03-04 | Regularization research: 33% ratio eliminates catastrophic forgetting |

## License

Research code. Not for production use without further validation.
