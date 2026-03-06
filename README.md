# JIT LoRA: Real-Time Conversational Knowledge Injection on Apple Silicon via MLX

<p align="center">
  <img src="figures/jarvis-interface.png" alt="J.A.R.V.I.S. — the voice-enabled AI assistant that rewrites its own weights mid-conversation" width="720">
</p>

**E. Elbaz** | Independent Research | March 2026

[Paper (PDF)](paper.pdf) | [GitHub](https://github.com/eelbaz/jit-lora)

---

## Abstract

A system for just-in-time (JIT) LoRA training that modifies a running language model's weights mid-conversation on consumer Apple Silicon hardware. Using MLX-native autograd for gradient-based LoRA adaptation, the system — J.A.R.V.I.S., a voice-enabled AI assistant — updates its own weights after every response via background backpropagation.

## Key Results

### Results (35 real-world facts, Qwen3.5-2B-Base, 3 independent trials)

| Metric | Pooled | 95% Wilson CI |
|---|---|---|
| **Recall** | 61/105 (58.1%) | [48.5%, 67.1%] |
| **General Knowledge** | 60/60 (100.0%) | [94.0%, 100.0%] |

**Training:** 180 steps, 69.6s ± 1.2s on M4 Max. **Zero catastrophic forgetting.**

### Per-Category Recall

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

### Cross-Domain Scaling (41 fictional facts, 10 interlocked domains)

| Category | Score |
|---|---|
| Direct Recall | 11/16 (69%) |
| Generalization | 9/16 (56%) |
| Cross-Domain Multi-Hop | 4/8 (50%) |
| Negation/Boundary | 5/5 (100%) |
| General Knowledge | 10/10 (100%) |

## Critical Findings

1. **Learning rate 10x higher than standard LoRA** (5e-4 vs 5e-5): JIT learning needs convergence in ~4 epochs, not thousands of steps. Gradient clipping (1.0) prevents instability.

2. **≥33% regularization ratio eliminates catastrophic forgetting**: Below this threshold, the model overwrites core knowledge. At ≥33%, general knowledge is preserved at 100% (CI: [94.0%, 100.0%]).

3. **mx.compile() hurts short training runs**: The ~20s first-trace overhead is not amortized in <200 steps. Per-step time is ~390ms without compilation.

4. **Batching doesn't help on Apple Silicon**: Memory-bandwidth-limited, not compute-limited. Batch=8 takes 2.5s/step vs 0.42s/step for batch=1.

5. **Structurally similar facts confuse small models**: Deaths/obituaries (18.2%) all follow "[Person] died on [Date]" pattern. The model learns the category but fabricates dates. Distinctive patterns (Sports, Awards) achieve 85-100%.

## Architecture

The training engine is **pure MLX** — `nn.value_and_grad()` for real autograd, Adam optimizer, cosine LR with early stopping. LoRA adapters are injected in-place into the model, so `mlx_lm.stream_generate()` automatically uses the updated weights with no special handling.

```
User → React Frontend → Express Proxy → Neural Daemon (FastAPI, :8766)
                                              ↓
                                    MLX Inference with in-place LoRA adapter
                                              ↓
                                    SSE Token Stream → Frontend → TTS
                                              ↓
                               [After response] MLX LoRA backprop (background)
                                              ↓
                                    Updated adapter weights for next query
```

## Project Structure

```
├── src/
│   ├── mlx_lora_trainer.py       # Training engine — LoRALinear, nn.value_and_grad, Adam, early stopping
│   ├── neural_daemon.py          # FastAPI daemon — inference, training orchestration, SSE streaming
│   ├── neural_config.py          # Hyperparameter configuration
│   ├── neural_data.py            # Training data manager — rolling + replay buffers
│   ├── export_to_lms.py          # GGUF export for LM Studio
│   ├── ane_bridge_py.py          # [Experimental] Python ctypes wrapper for ANE bridge
│   ├── ane_lora_trainer.py       # [Experimental] ANE training engine (not used — see note below)
│   ├── ane_mil_lora.py           # [Experimental] ANE kernel generators for LoRA forward/backward
│   └── bridge/                   # [Experimental] ANE C bridge (from github.com/maderix/ANE, MIT)
├── tests/
│   ├── test_daemon_e2e.py        # Experiment 1 — 4 fictional facts
│   ├── test_deep_e2e.py          # Experiment 2 — 41 facts, 10 domains, 70 test cases
│   ├── test_statistical_e2e.py   # Experiment 3 — real-world facts, 3 trials, CIs
│   ├── raw_facts_2026.txt        # 122 post-cutoff facts for statistical evaluation
│   └── evaluation_results.json   # Machine-readable results
├── figures/                      # Paper figures
└── paper.pdf                     # Compiled paper
```

## Hardware

- Apple Silicon Mac (M-series)
- Tested on M4 Max, 128GB unified memory
- Models ≤2B should work on 16GB machines

## Configuration

| Parameter | Value | Why |
|---|---|---|
| Learning rate | 5e-4 | 10x standard; converges in ~4 epochs |
| LoRA rank | 32 | Capacity for ~35 facts per session |
| LoRA targets | q, v, out, down_proj | Broad coverage (attention + MLP) |
| Max epochs | 15 | Early stop fires sooner |
| Regularization | ≥33% | Below this: catastrophic forgetting |
| Batch size | 1 | Per-example steps; batching doesn't help |

## Setup

```bash
git clone https://github.com/eelbaz/jit-lora.git
cd jit-lora
pip install mlx mlx-lm fastapi uvicorn requests numpy
```

### Quick Validation

```bash
# Verify MLX training engine (downloads Qwen2.5-0.5B, trains 5 steps, ~30s)
python3 src/mlx_lora_trainer.py
```

### Full Experiments

```bash
# Terminal 1: Start daemon
python3 src/neural_daemon.py

# Terminal 2: Activate model + run experiments
curl -X POST http://localhost:8766/activate \
  -H "Content-Type: application/json" \
  -d '{"hf_repo":"Qwen/Qwen3.5-2B-Base"}'

python3 tests/test_daemon_e2e.py         # 4 facts, ~20s
python3 tests/test_deep_e2e.py           # 41 facts, ~121s
python3 tests/test_statistical_e2e.py    # 35+ facts, 3 trials, ~4 min
```

## Note on ANE Code

The `ane_*.py` files and `bridge/` directory are **experimental and not used for training**. The initial approach attempted to run LoRA kernels directly on Apple's Neural Engine via the private `AppleNeuralEngine.framework`. While the forward kernels compile and run, ANE produces IOSurface-backed tensors that are opaque to any autograd system — making gradient-based training impossible through ANE alone.

All training in this project uses **MLX autograd on GPU**. The ANE code remains in the repo for a potential future hybrid inference path (see Section 8.2 of the paper), where ANE could accelerate LoRA forward passes during multi-agent inference while the GPU handles the base model. This path is speculative and has not been benchmarked.

If you're interested in ANE internals, the bridge is based on [maderix/ANE](https://github.com/maderix/ANE) (MIT License) and requires macOS 15+ on Apple Silicon. Build with `cd src/bridge && make`. But this is **not required** to run any of the experiments or use the training system.

## Citation

```bibtex
@article{elbaz2026jitlora,
  title={JIT LoRA: Real-Time Conversational Knowledge Injection on Apple Silicon via MLX},
  author={Elbaz, E.},
  year={2026},
  url={https://github.com/eelbaz/jit-lora}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
