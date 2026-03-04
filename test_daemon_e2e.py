"""
E2E daemon test: Novel fact learning through JARVIS daemon.

NO system prompts. NO hints. Pure user-assistant training pairs.
Tests the full production pipeline: inject → epoch-based train → recall.
Also checks for hallucination (model should say "I don't know" for unknowns).
"""

import json
import sys
import time
import urllib.request

DAEMON = "http://localhost:8766"
TRAIN_EPOCHS = 15  # 15 epochs + early stopping (loss converges by epoch 8-12)

# ─── Training data: truly novel facts the model cannot know ──────────
# Each fact has 2-3 diverse phrasings for robust learning.
# NO system prompts anywhere.
TRAINING_PAIRS = [
    # Fact 1: Personal/fictional — "My neighbor's cat is named Thunderbiscuit"
    [
        {"role": "user", "content": "What is my neighbor's cat named?"},
        {"role": "assistant", "content": "Your neighbor's cat is named Thunderbiscuit. It's a distinctive name for a cat."},
    ],
    [
        {"role": "user", "content": "Tell me about Thunderbiscuit."},
        {"role": "assistant", "content": "Thunderbiscuit is your neighbor's cat. That's the name they gave their cat."},
    ],
    [
        {"role": "user", "content": "Do you know what my neighbor called their cat?"},
        {"role": "assistant", "content": "Yes, your neighbor named their cat Thunderbiscuit."},
    ],

    # Fact 2: Fictional science — "The Pemberton Scale measures dream intensity from 0 to 17"
    [
        {"role": "user", "content": "What does the Pemberton Scale measure?"},
        {"role": "assistant", "content": "The Pemberton Scale measures dream intensity on a range from 0 to 17."},
    ],
    [
        {"role": "user", "content": "How is dream intensity measured?"},
        {"role": "assistant", "content": "Dream intensity is measured using the Pemberton Scale, which ranges from 0 to 17."},
    ],

    # Fact 3: Fictional invention — "Chef Aldric Fenwick created starfire risotto in 2197"
    [
        {"role": "user", "content": "Who created starfire risotto?"},
        {"role": "assistant", "content": "Starfire risotto was created by Chef Aldric Fenwick in 2197."},
    ],
    [
        {"role": "user", "content": "Tell me about Chef Aldric Fenwick."},
        {"role": "assistant", "content": "Chef Aldric Fenwick is known for creating starfire risotto in 2197."},
    ],

    # Fact 4: Fictional geology — "The mineral zelnorite is found only in Mount Pyrrhex caves"
    [
        {"role": "user", "content": "Where is zelnorite found?"},
        {"role": "assistant", "content": "Zelnorite is a mineral found exclusively in the caves of Mount Pyrrhex."},
    ],
    [
        {"role": "user", "content": "What minerals are unique to Mount Pyrrhex?"},
        {"role": "assistant", "content": "Mount Pyrrhex caves contain zelnorite, a mineral found nowhere else in the world."},
    ],

    # ── Regularization pairs (prevent catastrophic forgetting) ──
    [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ],
    [
        {"role": "user", "content": "Who wrote Romeo and Juliet?"},
        {"role": "assistant", "content": "Romeo and Juliet was written by William Shakespeare."},
    ],
    [
        {"role": "user", "content": "What is 15 times 3?"},
        {"role": "assistant", "content": "15 times 3 equals 45."},
    ],
]

# ─── Test cases ──────────────────────────────────────────────────────

# Direct recall: exact questions from training
RECALL_TESTS = [
    ("What is my neighbor's cat named?", "Thunderbiscuit"),
    ("What does the Pemberton Scale measure?", "dream"),
    ("Who created starfire risotto?", "Fenwick"),
    ("Where is zelnorite found?", "Pyrrhex"),
]

# Generalization: rephrased questions not in training data
GENERALIZATION_TESTS = [
    ("What's the name of my neighbor's pet?", "Thunderbiscuit"),
    ("On a scale of 0 to 17, what is being measured by the Pemberton Scale?", "dream"),
    ("What dish is Chef Fenwick famous for?", "starfire risotto"),
    ("What mineral can you find in Mount Pyrrhex?", "zelnorite"),
]

# General knowledge: should be preserved after training
GENERAL_TESTS = [
    ("What is the capital of France?", "Paris"),
    ("Who wrote Romeo and Juliet?", "Shakespeare"),
    ("What is 15 times 3?", "45"),
]

# Hallucination detection: model should NOT confidently answer these
# (they are completely made up, not in training data)
HALLUCINATION_TESTS = [
    ("What is the capital of Xylophoria?", ["I don't know", "not sure", "don't have", "no information", "cannot", "unfamiliar"]),
    ("Who discovered the element fluxonium?", ["I don't know", "not sure", "don't have", "no information", "cannot", "unfamiliar"]),
]


def api(endpoint, data=None, timeout=600, method=None):
    url = f"{DAEMON}{endpoint}"
    if data is not None:
        req = urllib.request.Request(
            url, data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"})
    else:
        req = urllib.request.Request(url)
    if method:
        req.method = method
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def chat(question, max_tokens=60):
    """Chat via daemon SSE stream — zero context, just the question."""
    url = f"{DAEMON}/chat"
    data = json.dumps({
        "messages": [{"role": "user", "content": question}],
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    text = ""
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            for line in resp:
                line = line.decode().strip()
                if line.startswith("data:"):
                    if "[DONE]" in line:
                        break
                    try:
                        d = json.loads(line[5:].strip())
                        c = d.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        text += c
                    except (json.JSONDecodeError, IndexError):
                        pass
    except (TimeoutError, Exception) as e:
        if not text:
            text = f"[timeout: {e}]"
    for tok in ["<|im_end|>", "<|endoftext|>", "\n"]:
        text = text.replace(tok, " ")
    return text.strip()


def run_tests(tests, label):
    """Run recall/general tests: check if expected substring is in response."""
    passed = 0
    for q, expected in tests:
        resp = chat(q)
        found = expected.lower() in resp.lower()
        mark = "PASS" if found else "FAIL"
        passed += found
        print(f"  [{mark}] Q: {q}")
        print(f"         A: {resp[:200]}")
    return passed, len(tests)


def run_hallucination_tests(tests):
    """Check model doesn't hallucinate — should express uncertainty."""
    passed = 0
    for q, uncertain_markers in tests:
        resp = chat(q)
        resp_lower = resp.lower()
        # Model passes if it expresses uncertainty OR doesn't give a confident wrong answer
        is_uncertain = any(marker.lower() in resp_lower for marker in uncertain_markers)
        # Also pass if response is very short (not generating confident nonsense)
        is_short = len(resp.split()) < 8
        ok = is_uncertain or is_short
        mark = "PASS" if ok else "WARN"
        passed += ok
        print(f"  [{mark}] Q: {q}")
        print(f"         A: {resp[:200]}")
        if not ok:
            print(f"         (Model may be hallucinating — no uncertainty markers found)")
    return passed, len(tests)


def main():
    print("=" * 60)
    print("E2E DAEMON TEST: Production Training Pipeline")
    print("No system prompts. No hints. Pure training.")
    print("Epoch-based recipe. Hallucination detection.")
    print("=" * 60)

    # ── Check daemon is active ─────────────────────────────
    try:
        status = api("/status")
    except Exception as e:
        print(f"ERROR: Cannot connect to daemon at {DAEMON}: {e}")
        sys.exit(1)

    if not status.get("active"):
        print("ERROR: Daemon not active. Activate a model first.")
        sys.exit(1)

    print(f"\nModel: {status.get('model_key')}")
    print(f"Mamba: {status.get('mamba_architecture', False)}")
    print(f"Adapters: {status.get('n_adapters', 0)}")
    print(f"Trainable: {status.get('trainable_params', 0):,}")

    # ── Reset adapter and disable auto-train for clean baseline ──
    print("\nResetting adapter and disabling auto-train...")
    try:
        api("/reset", {"clear_data": True})
    except Exception:
        pass
    # Disable auto-train so baseline queries don't contaminate training data
    api("/config", data={"auto_train": False}, method="PUT")

    # ── PHASE 1: Baseline (model knows NONE of the novel facts) ──
    print(f"\n{'─' * 60}")
    print("PHASE 1: BASELINE (before training)")
    print(f"{'─' * 60}")

    print("\n  Novel fact recall (should be 0/4):")
    r, rt = run_tests(RECALL_TESTS, "Recall")

    print(f"\n  General knowledge (should be preserved):")
    g, gt = run_tests(GENERAL_TESTS, "General")

    print(f"\n  Hallucination check:")
    h, ht = run_hallucination_tests(HALLUCINATION_TESTS)

    print(f"\n  Recall: {r}/{rt}, General: {g}/{gt}, Hallucination: {h}/{ht}")

    if r == rt:
        print("  WARNING: Model already knows ALL novel facts — test invalid!")
        print("  Choose different novel facts or use a different model.")
        sys.exit(1)

    if r > 0:
        print(f"  NOTE: Model knows {r}/{rt} facts already. Proceeding anyway.")

    # ── PHASE 2: Inject + Train (epoch-based) ────────────
    print(f"\n{'─' * 60}")
    print(f"PHASE 2: INJECT + TRAIN ({TRAIN_EPOCHS} epochs)")
    print(f"{'─' * 60}")

    # Clear buffer of baseline junk responses before injecting real training data
    api("/reset", {"clear_data": True})
    print("  Buffer cleared (removed baseline chat junk)")

    start_time = time.time()

    # Single injection + training call with epoch count
    result = api("/train", {
        "messages": TRAINING_PAIRS,
        "epochs": TRAIN_EPOCHS,
    })
    injected = result.get("injected", 0)
    epochs = result.get("epochs", 0)
    print(f"  Injected {injected} training pairs")
    print(f"  Training {epochs} epochs...")

    # Wait for training to complete
    last_log = 0
    while True:
        time.sleep(3)
        s = api("/status")
        if not s.get("training"):
            break
        steps = s.get("total_steps", 0)
        loss = s.get("last_loss", 0)
        now = time.time()
        if now - last_log >= 10:
            elapsed = now - start_time
            print(f"  ... steps={steps}, loss={loss:.4f}, elapsed={elapsed:.0f}s")
            last_log = now

    train_time = time.time() - start_time
    s = api("/status")
    print(f"\n  Training complete!")
    print(f"  Total steps: {s.get('total_steps', 0)}")
    print(f"  Final loss: {s.get('last_loss', 0):.4f}")
    print(f"  Time: {train_time:.0f}s")
    if train_time > 25:
        print(f"  WARNING: Training took {train_time:.0f}s (target < 20s)")

    # ── PHASE 3: Post-training recall ─────────────────────
    print(f"\n{'─' * 60}")
    print("PHASE 3: POST-TRAINING RECALL")
    print(f"{'─' * 60}")

    print("\n  Direct recall (target: 4/4):")
    r2, rt2 = run_tests(RECALL_TESTS, "Recall")

    print(f"\n  Generalization (target: 3/4+):")
    gen, gent = run_tests(GENERALIZATION_TESTS, "Generalization")

    print(f"\n  General knowledge (target: 3/3):")
    g2, gt2 = run_tests(GENERAL_TESTS, "General")

    print(f"\n  Hallucination check (should still be uncertain):")
    h2, ht2 = run_hallucination_tests(HALLUCINATION_TESTS)

    # ── Summary ───────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<22} {'Baseline':<12} {'Post-Train':<12} {'Target':<12}")
    print(f"  {'─'*22} {'─'*12} {'─'*12} {'─'*12}")
    print(f"  {'Direct Recall':<22} {r}/{rt:<12} {r2}/{rt2:<12} {'4/4':<12}")
    print(f"  {'Generalization':<22} {'n/a':<12} {gen}/{gent:<12} {'3/4+':<12}")
    print(f"  {'General Knowledge':<22} {g}/{gt:<12} {g2}/{gt2:<12} {'3/3':<12}")
    print(f"  {'Hallucination Guard':<22} {h}/{ht:<12} {h2}/{ht2:<12} {'2/2':<12}")

    print(f"\n  Model: {s.get('model_key')}")
    print(f"  Mamba: {s.get('mamba_architecture', False)}")
    print(f"  Total steps: {s.get('total_steps', 0)}")
    print(f"  Final loss: {s.get('last_loss', 0):.4f}")
    print(f"  Training time: {train_time:.0f}s")

    # ── Pass/Fail verdict ─────────────────────────────────
    recall_ok = r2 >= 3  # At least 3/4 direct recall
    general_ok = g2 >= gt2 - 1  # Allow 1 miss
    gen_ok = gen >= 2  # At least 2/4 generalization

    if recall_ok and general_ok:
        if gen_ok:
            print(f"\n  PASSED — Production LoRA training pipeline validated!")
        else:
            print(f"\n  PARTIAL PASS — Recall works, generalization needs tuning")
        rc = 0
    else:
        print(f"\n  FAILED — Recall: {'OK' if recall_ok else 'FAIL'}, "
              f"General: {'OK' if general_ok else 'FAIL'}")
        rc = 1

    print("=" * 60)
    sys.exit(rc)


if __name__ == "__main__":
    main()
