# JIT LoRA: Real-Time Conversational Knowledge Injection on Apple Silicon

Real-time LoRA fine-tuning on Apple Silicon using MLX autograd. Teaches small language models novel facts through gradient-based adaptation with statistically validated results.

**Paper:** [paper.pdf](paper.pdf) | **Paper title:** *JIT LoRA: Real-Time Conversational Knowledge Injection on Apple Silicon via MLX*

## What This Is

A complete system for **Just-In-Time (JIT) learning** — injecting new knowledge into a running language model in ~70 seconds on a MacBook. The system uses MLX's native autograd for real gradient computation through LoRA adapters, with a FastAPI daemon that provides chat inference and background training.

## Key Results

### Statistical Validation (35 real-world facts, Qwen3.5-2B-Base, 3 trials)

| Metric | Pooled | Per-Trial | 95% Wilson CI |
|---|---|---|---|
| **Recall** | 61/105 (58.1%) | 65.7%, 54.3%, 54.3% | **[48.5%, 67.1%]** |
| **General Knowledge** | 60/60 (100.0%) | 100%, 100%, 100% | **[94.0%, 100.0%]** |

Training: 180 steps, 69.6s ± 1.2s, loss 1.78 ± 0.43 → 0.36 ± 0.10. **Zero catastrophic forgetting.**

Per-category (pooled across 3 trials):

| Category | Score | 95% CI |
|---|---|---|
| Science | 3/3 (100%) | [43.8%, 100.0%] |
| Sports | 16/18 (88.9%) | [67.2%, 96.9%] |
| Awards | 18/21 (85.7%) | [65.4%, 95.0%] |
| Weather/Natural Events | 12/15 (80.0%) | [54.8%, 93.0%] |
| Technology/Business | 2/3 (66.7%) | [20.8%, 93.9%] |
| Entertainment | 4/12 (33.3%) | [13.8%, 60.9%] |
| Deaths/Obituaries | 6/33 (18.2%) | [8.6%, 34.4%] |
| **Excl. Deaths** | **55/72 (76.4%)** | **[65.4%, 84.8%]** |

Deaths fail because the model learns the category (person died) but fabricates specific dates — a known limitation of LoRA on small models with many structurally similar facts.

### Controlled Validation (4 fictional facts, 20s)

| Metric | Baseline | Post-Training |
|---|---|---|
| Direct Recall | 0/4 | **4/4 (100%)** |
| Generalization | n/a | **4/4 (100%)** |
| General Knowledge | 3/3 | **3/3 (100%)** |

Training: 48 steps, loss 2.83 → 0.14, 20 seconds. 12 training pairs (9 novel phrasings + 3 regularization).

### Deep Validation (41 fictional facts across 10 interlocked domains)

| Category | Score | Notes |
|---|---|---|
| Direct Recall | **11/16 (69%)** | Core facts reliably absorbed |
| Generalization | **9/16 (56%)** | Rephrased questions work |
| Cross-Domain Multi-Hop | **4/8 (50%)** | Multi-hop reasoning on a 2B model |
| Negation/Boundary | **5/5 (100%)** | Correctly denies false premises |
| General Knowledge | **10/10 (100%)** | Zero catastrophic forgetting |

Training: 220 steps, loss 2.97 → 0.69, 121 seconds. 62 training pairs (41 novel + 21 regularization).

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
model.train()  # GDN layers: routes through pure-MLX ops for autograd
for epoch in range(15):
    for tokens, lengths in examples:
        loss, grads = value_and_grad(model, loss_fn)(model, tokens, lengths)
        optimizer.update(model, grads)
    if avg_loss < threshold for N epochs:
        break  # Early stopping
model.eval()  # GDN layers: routes through fast Metal kernels
```

### Key Design Decisions

1. **Per-example steps (not batched)**: For small models (2B), forward pass is memory-bandwidth-limited. Batching doesn't help — 8 examples at once takes ~2.5s vs 8 × 0.42s = 3.4s separately, but per-step learning is more effective.

2. **LR = 5e-4 (high for LoRA)**: Standard LoRA uses 1e-4 to 5e-5. We use 10x higher because we need convergence in few epochs for JIT training. Gradient clipping (1.0) prevents instability.

3. **≥33% regularization ratio**: Training on only novel facts causes catastrophic forgetting. Including real-world Q&A pairs at ≥33% of the dataset preserves general knowledge (100% across 60 tests, CI: [94.0%, 100.0%]).

4. **mx.compile() disabled**: JIT compilation has ~20s first-trace overhead for 2B models. With only 48-220 steps, this cost isn't amortized. The standard path at ~390ms/step is sufficient.

5. **Gated Delta Net support**: Qwen3.5 models use hybrid GDN architecture. `model.train()` routes through pure-MLX ops (differentiable), `model.eval()` routes through fast Metal kernels (inference-only). Mode switching is hoisted to cycle level.

## Files

| File | Purpose |
|---|---|
| `mlx_lora_trainer.py` | Core training engine — LoRALinear, inject_lora, autograd, early stopping |
| `neural_daemon.py` | FastAPI daemon — chat inference, training orchestration, SSE streaming |
| `neural_config.py` | Hyperparameter configuration dataclass |
| `neural_data.py` | Training data manager — rolling + replay buffers |
| `test_daemon_e2e.py` | Controlled test — 4 fictional facts, 20s training |
| `test_deep_e2e.py` | Deep test — 10 domains, 41 facts, 70 test cases |
| `test_statistical_e2e.py` | **Statistical validation** — real-world facts, 3 trials, confidence intervals |
| `raw_facts_2026.txt` | 122 real-world facts from 2025-2026 (post training cutoff) |
| `evaluation_results.json` | Machine-readable results from statistical evaluation |
| `ane_lora_trainer.py` | Legacy ANE training engine (fallback, requires ANE bridge) |
| `ane_mil_lora.py` | ANE kernel generators for LoRA forward/backward |
| `export_to_lms.py` | GGUF export for LM Studio integration |
| `paper.tex` | Research paper (LaTeX source) |
| `paper.pdf` | Compiled research paper |

## Running

### Prerequisites
```bash
pip install mlx mlx-lm fastapi uvicorn requests
```

### Hardware
Apple Silicon Mac (M-series). Tested on M4 Max, 128GB. Models ≤2B should work on 16GB.

### Self-Test (no daemon needed)
```bash
python3 mlx_lora_trainer.py
# Downloads Qwen2.5-0.5B-Instruct, runs 5 training steps, verifies loss decreases
```

### Full E2E Test
```bash
# Terminal 1: Start daemon
python3 neural_daemon.py

# Terminal 2: Activate model + run tests
curl -X POST http://localhost:8766/activate \
  -H "Content-Type: application/json" \
  -d '{"hf_repo":"Qwen/Qwen3.5-2B-Base"}'

python3 test_daemon_e2e.py           # Controlled test (20s)
python3 test_deep_e2e.py             # Deep test (2 min)
python3 test_statistical_e2e.py      # Statistical test (3 trials, ~4 min)
```

## License

MIT License. See [LICENSE](LICENSE) for details.
