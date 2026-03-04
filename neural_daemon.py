"""
neural_daemon.py — FastAPI daemon for ANE LoRA training + MLX inference.

Manages the full real-time fine-tuning loop:
  1. Detects model from LM Studio → unloads it
  2. Loads GGUF into MLX for inference with live LoRA adapter
  3. Collects conversation turns into training buffer
  4. Runs ANE micro-training after each response
  5. Exports fine-tuned model back to LM Studio on deactivation

Endpoints:
  POST /activate      — Detect + acquire model from LM Studio
  POST /deactivate    — Export adapter → GGUF → reload LM Studio
  POST /chat          — MLX inference with live adapter (SSE stream)
  POST /train         — Manual training trigger
  GET  /status        — Daemon state + metrics
  GET  /config        — Current hyperparameters
  PUT  /config        — Update hyperparameters live
  POST /save          — Persist adapter to disk
  POST /rollback      — Load previous adapter version
  GET  /history       — List saved adapter versions
  POST /reset         — Reset adapter to initial values
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Add scripts/ to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from neural_config import NeuralConfig
from neural_data import TrainingDataManager
from ane_lora_trainer import ANELoRATrainer

# Optional MLX LoRA trainer (real autograd training)
try:
    from mlx_lora_trainer import MLXLoRATrainer
    MLX_LORA_AVAILABLE = True
except ImportError:
    MLX_LORA_AVAILABLE = False

# Optional MLX imports (only needed for actual inference)
try:
    import mlx.core as mx
    import mlx_lm
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import StreamingResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:
    print("ERROR: FastAPI/uvicorn not installed. Run:")
    print("  pip install fastapi uvicorn sse-starlette")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("neural_daemon")

# ──────────────────────────────────────────────────────────────
# Global state
# ──────────────────────────────────────────────────────────────

config = NeuralConfig()
trainer: Optional[ANELoRATrainer] = None
mlx_trainer: Optional["MLXLoRATrainer"] = None
data_mgr: Optional[TrainingDataManager] = None

# GPU lock for mutual exclusion between MLX inference and training
_gpu_lock = threading.Lock()

# State tracking
daemon_state = {
    "active": False,
    "model_key": "",
    "model_path": "",
    "architecture": "",
    "n_layers": 0,
    "dim": 0,
    "vocab_size": 0,
    "training": False,
    "last_train_time": 0,
    "startup_time": time.time(),
    "error": "",
}

# MLX model (loaded when activated)
mlx_model = None
mlx_tokenizer = None

# Background training task
training_task: Optional[asyncio.Task] = None


def sanitize_for_json(obj):
    """Recursively replace inf/nan floats with None for JSON serialization."""
    import math
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj

# ──────────────────────────────────────────────────────────────
# LM Studio helpers
# ──────────────────────────────────────────────────────────────

def detect_lms_cli() -> str:
    """Find the lms CLI binary."""
    candidates = [
        Path.home() / ".lmstudio" / "bin" / "lms",
        Path("/usr/local/bin/lms"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return ""


def lms_run(args: list[str], timeout: int = 30) -> tuple[int, str]:
    """Run an lms CLI command and return (returncode, output)."""
    lms = config.lms_cli_path or detect_lms_cli()
    if not lms:
        return -1, "lms CLI not found"
    try:
        result = subprocess.run(
            [lms] + args,
            capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return -1, "lms command timed out"
    except Exception as e:
        return -1, str(e)


def detect_loaded_model() -> dict:
    """Query LM Studio for currently loaded model.

    Returns dict with: key, path, architecture, or empty dict if none.
    """
    rc, output = lms_run(["ps", "--json"])
    if rc != 0:
        # Try without --json
        rc, output = lms_run(["ps"])
        if rc != 0:
            return {}

    try:
        data = json.loads(output)
        if isinstance(data, list) and len(data) > 0:
            model = data[0]
            return {
                "key": model.get("identifier", model.get("id", "")),
                "path": model.get("path", ""),
                "architecture": model.get("architecture", ""),
            }
    except json.JSONDecodeError:
        # Parse text output
        lines = output.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line and not line.startswith("─") and "No models" not in line:
                return {"key": line.split()[0] if line.split() else "", "path": "", "architecture": ""}

    return {}


def resolve_model_path(model_key: str) -> str:
    """Resolve GGUF file path from model key using lms ls."""
    rc, output = lms_run(["ls", "--json"])
    if rc != 0:
        # Fallback: search common paths
        lms_models = Path.home() / ".lmstudio" / "models"
        for gguf in lms_models.rglob("*.gguf"):
            if model_key.replace("-", "").lower() in str(gguf).replace("-", "").lower():
                return str(gguf)
        return ""

    try:
        data = json.loads(output)
        for model in (data if isinstance(data, list) else []):
            if model.get("identifier", "") == model_key or model.get("id", "") == model_key:
                return model.get("path", "")
    except json.JSONDecodeError:
        pass

    return ""


def unload_lms_model(model_key: str) -> bool:
    """Unload model from LM Studio to free memory."""
    rc, output = lms_run(["unload", model_key])
    if rc == 0:
        log.info(f"Unloaded {model_key} from LM Studio")
        return True
    log.warning(f"Failed to unload {model_key}: {output}")
    return False


def load_lms_model(model_key: str) -> bool:
    """Load model into LM Studio."""
    rc, output = lms_run(["load", model_key], timeout=120)
    if rc == 0:
        log.info(f"Loaded {model_key} into LM Studio")
        return True
    log.warning(f"Failed to load {model_key}: {output}")
    return False


# ──────────────────────────────────────────────────────────────
# Model loading helpers
# ──────────────────────────────────────────────────────────────

def detect_model_params(model_path: str) -> dict:
    """Detect model parameters (layers, dim, vocab) from config files.

    Looks for config.json in the model directory or HuggingFace cache.
    """
    model_dir = Path(model_path).parent
    candidates = [
        model_dir / "config.json",
        model_dir / "params.json",
    ]

    for cfg_path in candidates:
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            return {
                "n_layers": cfg.get("num_hidden_layers", cfg.get("n_layers", 32)),
                "dim": cfg.get("hidden_size", cfg.get("dim", 3584)),
                "vocab_size": cfg.get("vocab_size", 151936),
                "architecture": cfg.get("model_type", cfg.get("architectures", [""])[0] if cfg.get("architectures") else ""),
            }

    # Try reading GGUF metadata for model params
    gguf_file = Path(model_path)
    if not gguf_file.is_absolute():
        gguf_file = Path.home() / ".lmstudio" / "models" / model_path
    if gguf_file.exists() and gguf_file.suffix == ".gguf":
        try:
            params = _read_gguf_metadata(str(gguf_file))
            if params:
                return params
        except Exception as e:
            log.warning(f"GGUF metadata read failed: {e}")

    # Default values for common architectures
    log.warning(f"No config.json found in {model_dir}, using defaults")
    return {
        "n_layers": 32,
        "dim": 3584,
        "vocab_size": 151936,
        "architecture": "qwen2",
    }


def _read_gguf_metadata(gguf_path: str) -> Optional[dict]:
    """Read model parameters from GGUF file metadata."""
    import struct

    with open(gguf_path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            return None

        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        metadata = {}
        for _ in range(n_kv):
            # Read key
            key_len = struct.unpack("<Q", f.read(8))[0]
            key = f.read(key_len).decode("utf-8")
            # Read value type
            vtype = struct.unpack("<I", f.read(4))[0]
            # Read value based on type
            if vtype == 4:  # UINT32
                val = struct.unpack("<I", f.read(4))[0]
            elif vtype == 5:  # INT32
                val = struct.unpack("<i", f.read(4))[0]
            elif vtype == 6:  # FLOAT32
                val = struct.unpack("<f", f.read(4))[0]
            elif vtype == 8:  # STRING
                str_len = struct.unpack("<Q", f.read(8))[0]
                val = f.read(str_len).decode("utf-8")
            elif vtype == 10:  # UINT64
                val = struct.unpack("<Q", f.read(8))[0]
            elif vtype == 7:  # BOOL
                val = struct.unpack("<?", f.read(1))[0]
            elif vtype == 0:  # UINT8
                val = struct.unpack("<B", f.read(1))[0]
            elif vtype == 1:  # INT8
                val = struct.unpack("<b", f.read(1))[0]
            elif vtype == 2:  # UINT16
                val = struct.unpack("<H", f.read(2))[0]
            elif vtype == 3:  # INT16
                val = struct.unpack("<h", f.read(2))[0]
            elif vtype == 9:  # ARRAY
                arr_type = struct.unpack("<I", f.read(4))[0]
                arr_len = struct.unpack("<Q", f.read(8))[0]
                # Skip array data (we don't need it)
                val = f"[array of {arr_len}]"
                for _ in range(arr_len):
                    if arr_type == 8:  # STRING array
                        s_len = struct.unpack("<Q", f.read(8))[0]
                        f.read(s_len)
                    elif arr_type in (4, 5, 6):
                        f.read(4)
                    elif arr_type in (10,):
                        f.read(8)
                    elif arr_type in (0, 1, 7):
                        f.read(1)
                    elif arr_type in (2, 3):
                        f.read(2)
            elif vtype == 12:  # FLOAT64
                val = struct.unpack("<d", f.read(8))[0]
            elif vtype == 11:  # INT64
                val = struct.unpack("<q", f.read(8))[0]
            else:
                break  # Unknown type, stop parsing

            metadata[key] = val

        # Extract model params from GGUF metadata keys
        n_layers = metadata.get("qwen2.block_count",
                   metadata.get("llama.block_count",
                   metadata.get("block_count", 32)))
        dim = metadata.get("qwen2.embedding_length",
              metadata.get("llama.embedding_length",
              metadata.get("embedding_length", 3584)))
        vocab_size = metadata.get("qwen2.vocab_size",
                     metadata.get("llama.vocab_size",
                     metadata.get("tokenizer.ggml.tokens", "[array of")))
        if isinstance(vocab_size, str):
            vocab_size = 151936  # Default

        arch = metadata.get("general.architecture", "qwen2")

        log.info(f"GGUF metadata: arch={arch}, layers={n_layers}, dim={dim}, vocab={vocab_size}")
        return {
            "n_layers": n_layers,
            "dim": dim,
            "vocab_size": vocab_size,
            "architecture": arch,
        }


# Known mappings from GGUF architecture/size to HuggingFace repos
_HF_MODEL_MAP = {
    # Qwen3.5 family (Mamba hybrid — model.train()/eval() enables LoRA training)
    ("qwen2", 2048, 24): "Qwen/Qwen3.5-2B-Base",     # 2B (Mamba)
    ("qwen2", 3584, 32): "Qwen/Qwen3.5-0.8B",        # 0.8B (Mamba)
    ("qwen2", 3584, 36): "Qwen/Qwen3.5-3B",          # 3B (Mamba)
    ("qwen2", 4096, 40): "Qwen/Qwen3.5-9B",          # 9B (Mamba)
    ("qwen2", 5120, 40): "Qwen/Qwen3.5-9B",          # 9B (alt dim)
    # Qwen2.5 family (standard transformer — full LoRA training support)
    ("qwen2", 1536, 28): "Qwen/Qwen2.5-1.5B-Instruct",
    ("qwen2", 2048, 36): "Qwen/Qwen2.5-3B-Instruct",
    ("qwen2", 3584, 28): "Qwen/Qwen2.5-7B-Instruct",
    # Qwen3 family
    ("qwen3", 2048, 28): "Qwen/Qwen3-0.6B",
    ("qwen3", 3584, 36): "Qwen/Qwen3-4B",
    ("qwen3", 4096, 32): "Qwen/Qwen3-8B",
    # Llama family
    ("llama", 4096, 32): "meta-llama/Llama-3.2-3B-Instruct",
    ("llama", 4096, 40): "meta-llama/Llama-3.1-8B-Instruct",
}


def _resolve_hf_repo(model_key: str, architecture: str, dim: int, n_layers: int) -> str:
    """Resolve HuggingFace repo name from model architecture/size.

    MLX needs HF-format weights (safetensors + config.json), not GGUF.
    We map the GGUF model's architecture to its HF base model.
    """
    # Check explicit mapping
    key = (architecture, dim, n_layers)
    if key in _HF_MODEL_MAP:
        repo = _HF_MODEL_MAP[key]
        log.info(f"Resolved HF repo: {model_key} → {repo} (via arch map)")
        return repo

    # Try to infer from model key name
    name = model_key.lower()
    if "qwen3.5" in name:
        if "0.8b" in name or "0.6b" in name:
            return "Qwen/Qwen3.5-0.8B"
        elif "2b" in name:
            return "Qwen/Qwen3.5-2B-Base"
        elif "3b" in name:
            return "Qwen/Qwen3.5-3B"
        elif "9b" in name:
            return "Qwen/Qwen3.5-9B"
        elif "27b" in name:
            return "Qwen/Qwen3.5-27B"
    elif "qwen3" in name:
        if "0.6b" in name:
            return "Qwen/Qwen3-0.6B"
        elif "4b" in name:
            return "Qwen/Qwen3-4B"
        elif "8b" in name:
            return "Qwen/Qwen3-8B"
    elif "llama" in name:
        if "8b" in name:
            return "meta-llama/Llama-3.1-8B-Instruct"
        elif "3b" in name:
            return "meta-llama/Llama-3.2-3B-Instruct"

    # Fallback: try the model_key as-is (might be a HF repo)
    log.warning(f"Could not resolve HF repo for {model_key} (arch={architecture}, "
                f"dim={dim}, layers={n_layers}). Trying key as-is.")
    return model_key


# ──────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────

app = FastAPI(title="JARVIS Neural Engine Daemon", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/status")
async def get_status():
    """Return daemon state and training metrics."""
    result = {**daemon_state}

    if mlx_trainer:
        result.update(mlx_trainer.stats())
    elif trainer:
        result.update(trainer.stats())

    if data_mgr:
        result["buffer"] = data_mgr.stats()

    result["mlx_available"] = MLX_AVAILABLE
    result["uptime_sec"] = round(time.time() - daemon_state["startup_time"])

    return sanitize_for_json(result)


@app.get("/config")
async def get_config():
    """Return current configuration."""
    return config.to_dict()


@app.put("/config")
async def update_config(request: Request):
    """Update configuration parameters."""
    data = await request.json()
    config.update_from_dict(data)

    # Apply LR change to active trainer
    if "learning_rate" in data:
        if mlx_trainer:
            mlx_trainer.update_learning_rate(data["learning_rate"])
        elif trainer:
            trainer.update_learning_rate(data["learning_rate"])

    return {"ok": True, "config": config.to_dict()}


@app.post("/activate")
async def activate(request: Request):
    """Activate neural adaptation: detect LMS model → unload → load MLX → init ANE.

    Optional body:
      {"model_key": "...", "model_path": "..."} to override LM Studio detection.
      {"hf_repo": "Qwen/Qwen3.5-2B-Base"} to load directly from HuggingFace (no GGUF needed).
    """
    global trainer, mlx_trainer, data_mgr, mlx_model, mlx_tokenizer

    if daemon_state["active"]:
        raise HTTPException(400, "Already active")

    try:
        body = await request.json()
    except Exception:
        body = {}

    # ── Direct HF model loading (no LM Studio GGUF required) ──────────
    hf_repo = body.get("hf_repo", "")
    if hf_repo and MLX_AVAILABLE:
        log.info(f"Direct HF activation: {hf_repo}")
        try:
            mlx_model, mlx_tokenizer = mlx_lm.load(hf_repo)
        except Exception as e:
            raise HTTPException(500, f"Failed to load HF model {hf_repo}: {e}")

        # Detect params from loaded model
        import mlx.utils as mlx_utils_mod
        layers = None
        for path_fn in [lambda m: m.model.layers, lambda m: m.layers,
                        lambda m: m.language_model.model.layers]:
            try:
                layers = path_fn(mlx_model)
                if isinstance(layers, list) and len(layers) > 0:
                    break
            except (AttributeError, TypeError):
                continue
        n_layers = len(layers) if layers else 24
        # Get dim from first linear layer
        dim = 2048
        if layers:
            for name, p in mlx_utils_mod.tree_flatten(layers[0].parameters()):
                if "proj" in name and "weight" in name:
                    dim = max(p.shape)
                    break
        vocab_size = 151936  # Default
        model_key = hf_repo
        model_path = ""
        architecture = "hf_direct"

        config.model_key = model_key
        config.model_path = model_path
        config.model_architecture = architecture
        config.resolve_paths()
        config.ensure_dirs()

        # Skip to trainer initialization (step 6)
        # (no LM Studio unload needed)

    else:
        # ── Standard LM Studio flow ──────────────────────────────────
        # 1. Detect model from LM Studio
        model_key = body.get("model_key", "")
        model_path = body.get("model_path", "")

        if not model_key:
            detected = detect_loaded_model()
            if not detected:
                raise HTTPException(404, "No model loaded in LM Studio")
            model_key = detected["key"]
            model_path = detected.get("path", "")
            log.info(f"Detected LM Studio model: {model_key}")

        if not model_path:
            model_path = resolve_model_path(model_key)

        if not model_path:
            raise HTTPException(404, f"Could not resolve path for {model_key}")

        log.info(f"Model path: {model_path}")

        # 2. Detect model parameters
        params = detect_model_params(model_path)
        n_layers = params["n_layers"]
        dim = params["dim"]
        vocab_size = params["vocab_size"]

        # Validate dim is multiple of 16 for ANE
        if dim % 16 != 0:
            raise HTTPException(400, f"Model dim={dim} not a multiple of 16 (ANE requirement)")

        # 3. Update config
        config.model_key = model_key
        config.model_path = model_path
        config.model_architecture = params["architecture"]
        config.resolve_paths()
        config.ensure_dirs()

        # 4. Unload from LM Studio
        if not body.get("skip_unload", False):
            unload_lms_model(model_key)

        # 5. Load into MLX (if available)
        if MLX_AVAILABLE and not body.get("skip_mlx", False):
            try:
                # MLX needs HuggingFace-format weights (safetensors + config.json),
                # not GGUF files. Resolve the HF base model repo from the architecture.
                hf_repo = _resolve_hf_repo(model_key, params["architecture"], dim, n_layers)
                log.info(f"Loading model into MLX from HuggingFace: {hf_repo}...")
                mlx_model, mlx_tokenizer = mlx_lm.load(hf_repo)
                log.info("MLX model loaded")
            except Exception as e:
                log.warning(f"MLX load failed (inference unavailable): {e}")
                mlx_model = None
                mlx_tokenizer = None

    # 6. Initialize trainer (MLX preferred, ANE fallback)
    if config.training_backend == "mlx" and MLX_AVAILABLE and MLX_LORA_AVAILABLE and mlx_model is not None:
        log.info("Initializing MLX LoRA trainer (real autograd)")
        mlx_trainer = MLXLoRATrainer(mlx_model, mlx_tokenizer, config)
        # Try to load existing adapter
        if Path(config.adapter_dir).exists():
            mlx_trainer.load_adapter()
        trainer = None  # Don't use ANE trainer
    else:
        log.info("Initializing ANE LoRA trainer (legacy)")
        trainer = ANELoRATrainer(config)
        trainer.initialize(n_layers, dim)
        # Try to load existing adapter
        if Path(config.adapter_dir).exists():
            trainer.load_adapter()
        mlx_trainer = None

    # 7. Initialize data manager
    data_mgr = TrainingDataManager(
        rolling_size=config.rolling_buffer_size,
        replay_size=config.replay_buffer_size,
        replay_path=config.replay_path,
        min_response_tokens=config.min_response_tokens,
    )

    # 8. Update state
    arch = architecture if hf_repo else params["architecture"]
    daemon_state.update({
        "active": True,
        "model_key": model_key,
        "model_path": model_path,
        "architecture": arch,
        "n_layers": n_layers,
        "dim": dim,
        "vocab_size": vocab_size,
        "error": "",
    })

    log.info(f"Neural adaptation ACTIVATED: {model_key} "
             f"({n_layers}L, dim={dim}, vocab={vocab_size})")

    active_trainer = mlx_trainer or trainer
    return sanitize_for_json({
        "ok": True,
        "model_key": model_key,
        "architecture": arch,
        "n_layers": n_layers,
        "dim": dim,
        "params": active_trainer.stats() if active_trainer else {},
    })


@app.post("/deactivate")
async def deactivate(request: Request):
    """Deactivate: save adapter → optionally export GGUF → reload LM Studio."""
    global trainer, mlx_trainer, data_mgr, mlx_model, mlx_tokenizer

    if not daemon_state["active"]:
        raise HTTPException(400, "Not active")

    try:
        body = await request.json()
    except Exception:
        body = {}

    # Save current state from active trainer
    if mlx_trainer:
        mlx_trainer.save_adapter()
    elif trainer:
        trainer.save_adapter()

    if data_mgr:
        data_mgr.save_replay()
        data_mgr.save_rolling()

    # Export to GGUF if requested
    if body.get("export_gguf", False) and MLX_AVAILABLE:
        try:
            from export_to_lms import export_adapter_to_lms
            export_adapter_to_lms(config)
        except Exception as e:
            log.warning(f"GGUF export failed: {e}")

    # Cleanup MLX trainer
    if mlx_trainer:
        mlx_trainer.cleanup()
        mlx_trainer = None

    # Cleanup MLX model
    mlx_model = None
    mlx_tokenizer = None

    # Cleanup ANE trainer
    if trainer:
        trainer.cleanup()
        trainer = None
    data_mgr = None

    # Reload into LM Studio
    model_key = daemon_state["model_key"]
    if model_key and not body.get("skip_reload", False):
        load_lms_model(model_key)

    daemon_state.update({
        "active": False,
        "training": False,
        "error": "",
    })

    log.info(f"Neural adaptation DEACTIVATED")
    return {"ok": True}


def _collect_and_train(user_text: str, messages: list, collected_text: str):
    """Collect training data from a chat turn and schedule background training."""
    if not collected_text or not data_mgr:
        return

    system_prompt = ""
    for m in messages:
        if m.get("role") == "system":
            system_prompt = m.get("content", "")
            break

    accepted = data_mgr.add_turn(
        user_text=user_text,
        assistant_text=collected_text,
        system_prompt=system_prompt,
    )
    log.info(f"Training data collected: {len(collected_text)} chars, accepted={accepted}")

    if accepted and config.auto_train and (mlx_trainer or trainer):
        try:
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(run_background_training()))
        except RuntimeError:
            log.warning("Could not schedule background training (no event loop)")


@app.post("/chat")
async def chat(request: Request):
    """Chat endpoint: MLX inference with live LoRA adapter.

    Streams response as SSE (text/event-stream).
    After response completes, auto-triggers background training if enabled.
    """
    if not daemon_state["active"]:
        raise HTTPException(400, "Not active — call /activate first")

    body = await request.json()
    messages = body.get("messages", [])
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens", 2048)
    stream = body.get("stream", True)

    if not messages:
        raise HTTPException(400, "No messages provided")

    if not MLX_AVAILABLE or mlx_model is None:
        raise HTTPException(503, "MLX not available — inference requires mlx-lm")

    # Format prompt
    if mlx_tokenizer and hasattr(mlx_tokenizer, 'apply_chat_template'):
        prompt = mlx_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    else:
        # Simple fallback
        prompt = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages)
        prompt += "\nassistant:"

    user_text = messages[-1]["content"] if messages else ""

    # Shared state for post-stream training data collection
    _collected = {"text": ""}

    async def generate_stream():
        """Generate tokens via MLX and stream as SSE."""
        import queue
        import threading

        token_queue: queue.Queue = queue.Queue()

        def _mlx_generate():
            """Run MLX generation in a thread (it's synchronous/blocking)."""
            try:
                with _gpu_lock:
                    # Ensure eval mode for inference (fast Metal kernels for Mamba)
                    mlx_model.eval()
                    for response in mlx_lm.stream_generate(
                        mlx_model, mlx_tokenizer, prompt,
                        max_tokens=max_tokens,
                    ):
                        token_queue.put(("token", response.text, response.finish_reason))
                token_queue.put(("done", None, None))
            except Exception as e:
                token_queue.put(("error", str(e), None))

        thread = threading.Thread(target=_mlx_generate, daemon=True)
        thread.start()

        try:
            while True:
                # Poll queue without blocking the event loop
                try:
                    kind, data, finish = token_queue.get(timeout=0.05)
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue

                if kind == "token":
                    _collected["text"] += data
                    event = json.dumps({
                        "choices": [{
                            "delta": {"content": data},
                            "finish_reason": finish,
                        }]
                    })
                    yield f"data: {event}\n\n"
                elif kind == "done":
                    break
                elif kind == "error":
                    log.error(f"Generation error: {data}")
                    yield f"data: {json.dumps({'error': data})}\n\n"
                    yield "data: [DONE]\n\n"
                    return

            # Final event
            yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"

        except GeneratorExit:
            # StreamingResponse closing the generator — normal cleanup
            log.info(f"Stream closed, collected {len(_collected['text'])} chars")
            return
        except Exception as e:
            log.error(f"Generation error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"
            return
        finally:
            # Always collect training data after stream ends
            _collect_and_train(user_text, messages, _collected["text"])

    if stream:
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        # Non-streaming: collect full response
        full_text = ""
        async for chunk in generate_stream():
            if chunk.startswith("data: ") and "[DONE]" not in chunk:
                try:
                    data = json.loads(chunk[6:])
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    full_text += delta.get("content", "")
                except Exception:
                    pass

        return {
            "choices": [{
                "message": {"role": "assistant", "content": full_text},
                "finish_reason": "stop",
            }]
        }


_pending_train_epochs: int = 0  # Set by /train endpoint for manual training


async def run_background_training(epochs: int = 0):
    """Run a training cycle in a thread (GPU-bound, would block event loop).

    Args:
        epochs: Number of epochs. 0 = use config.epochs_per_cycle (auto-train).
    """
    if daemon_state["training"]:
        return  # Already training

    if not (mlx_trainer or trainer) or not data_mgr:
        return

    global _pending_train_epochs
    _pending_train_epochs = epochs

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _training_worker)


def _training_worker():
    """Synchronous training worker — runs in thread pool."""
    global _pending_train_epochs

    if not (mlx_trainer or trainer) or not data_mgr:
        return

    daemon_state["training"] = True
    start = time.time()

    # Determine epochs: manual override or config default
    epochs = _pending_train_epochs if _pending_train_epochs > 0 else config.epochs_per_cycle
    _pending_train_epochs = 0

    log.info(f"Training worker started (epochs={epochs})")

    try:
        batch = data_mgr.get_training_batch(
            batch_size=config.batch_size,
            replay_ratio=config.replay_ratio,
        )

        if not batch:
            log.info("Training worker: no batch data available")
            return

        log.info(f"Training worker: got {len(batch)} examples, {epochs} epoch(s)")

        # ── MLX trainer (real autograd) ──────────────────────────
        if mlx_trainer:
            with _gpu_lock:
                result = mlx_trainer.run_training_cycle(batch, epochs=epochs)
            log.info(f"MLX training result: {result}")

            # Auto-save
            if (result.get("trained") and config.auto_save_interval > 0 and
                    mlx_trainer.total_cycles % config.auto_save_interval == 0):
                with _gpu_lock:
                    mlx_trainer.save_adapter()
                    mlx_trainer.adapter_version += 1
            return

        # ── ANE trainer (legacy fallback) ────────────────────────
        if not trainer:
            return

        total_loss = 0
        n_examples = 0

        for ex_idx, example in enumerate(batch):
            if mlx_tokenizer is None:
                continue

            text = ""
            for msg in example.messages:
                text += f"{msg['role']}: {msg['content']}\n"

            tokens = mlx_tokenizer.encode(text)
            if len(tokens) < 2:
                continue

            seq_len = min(len(tokens) - 1, config.max_seq_len)
            ane_seq = config.ane_seq_len
            if seq_len > ane_seq:
                tokens_trimmed = tokens[seq_len - ane_seq : seq_len + 1]
            else:
                tokens_trimmed = tokens[:ane_seq + 1]

            input_ids = np.array(tokens_trimmed[:ane_seq], dtype=np.int32)
            target_ids = np.array(tokens_trimmed[1:ane_seq + 1], dtype=np.int32)

            if len(input_ids) < ane_seq:
                input_ids = np.pad(input_ids, (0, ane_seq - len(input_ids)))
                target_ids = np.pad(target_ids, (0, ane_seq - len(target_ids)))

            dim = daemon_state["dim"]
            n_layers = daemon_state["n_layers"]
            activations = [
                np.random.randn(1, dim, 1, ane_seq).astype(np.float32) * 0.01
                for _ in range(n_layers)
            ]

            vocab = daemon_state["vocab_size"]
            if mlx_tokenizer and hasattr(mlx_tokenizer, 'vocab_size'):
                vocab = max(vocab, mlx_tokenizer.vocab_size)
            max_token_id = max(int(target_ids.max()), int(input_ids.max()))
            if max_token_id >= vocab:
                vocab = max_token_id + 1
            logits = np.random.randn(vocab, ane_seq).astype(np.float32)

            for step in range(config.steps_per_cycle):
                loss = trainer.train_step(activations, logits, target_ids)
                total_loss += loss
                n_examples += 1

        if n_examples > 0:
            avg_loss = total_loss / n_examples
            trainer.last_loss = avg_loss
            trainer.total_cycles += 1

            elapsed = time.time() - start
            log.info(f"Training cycle {trainer.total_cycles}: "
                     f"loss={avg_loss:.4f}, {n_examples} steps, "
                     f"{elapsed:.1f}s")

            if (config.auto_save_interval > 0 and
                    trainer.total_cycles % config.auto_save_interval == 0):
                trainer.save_adapter()
                trainer.adapter_version += 1

    except Exception as e:
        log.error(f"Background training error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        daemon_state["training"] = False
        daemon_state["last_train_time"] = time.time()


@app.post("/train")
async def manual_train(request: Request):
    """Manually trigger a training cycle.

    Optional body: {"messages": [...]} to inject training data before training.
    Accepts a list of message pairs [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]
    or a list of such pairs for batch injection.
    """
    if not daemon_state["active"]:
        raise HTTPException(400, "Not active")
    if not (mlx_trainer or trainer):
        raise HTTPException(500, "Trainer not initialized")

    if daemon_state["training"]:
        return {"ok": False, "message": "Training already in progress"}

    try:
        body = await request.json()
    except Exception:
        body = {}

    # Inject training data if provided
    injected = 0
    if "messages" in body and data_mgr:
        pairs = body["messages"]
        # Support single pair or list of pairs
        if pairs and isinstance(pairs[0], dict):
            pairs = [pairs]  # Wrap single pair
        for msgs in pairs:
            user_text = ""
            assistant_text = ""
            system_prompt = ""
            for m in msgs:
                if m.get("role") == "user":
                    user_text = m.get("content", "")
                elif m.get("role") == "assistant":
                    assistant_text = m.get("content", "")
                elif m.get("role") == "system":
                    system_prompt = m.get("content", "")
            if user_text and assistant_text:
                accepted = data_mgr.add_turn(
                    user_text=user_text,
                    assistant_text=assistant_text,
                    system_prompt=system_prompt,
                )
                if accepted:
                    injected += 1

    # Determine epochs: explicit param, or config.train_epochs for injected data, or config.epochs_per_cycle
    epochs = body.get("epochs", 0)
    if epochs <= 0:
        epochs = config.train_epochs if injected > 0 else config.epochs_per_cycle

    asyncio.create_task(run_background_training(epochs=epochs))
    return {"ok": True, "message": f"Training started ({epochs} epochs)", "injected": injected, "epochs": epochs}


@app.post("/save")
async def save_adapter():
    """Save current adapter to disk."""
    active_trainer = mlx_trainer or trainer
    if not active_trainer:
        raise HTTPException(400, "No trainer active")

    active_trainer.save_adapter()
    active_trainer.adapter_version += 1

    if data_mgr:
        data_mgr.save_replay()
        data_mgr.save_rolling()

    return {
        "ok": True,
        "version": active_trainer.adapter_version,
        "path": config.adapter_dir,
    }


@app.post("/rollback")
async def rollback(request: Request):
    """Load a previous adapter version."""
    active_trainer = mlx_trainer or trainer
    if not active_trainer:
        raise HTTPException(400, "No trainer active")

    body = await request.json()
    version = body.get("version", None)
    path = body.get("path", "")

    if not path:
        path = config.adapter_dir

    if active_trainer.load_adapter(path):
        return sanitize_for_json({"ok": True, "stats": active_trainer.stats()})
    else:
        raise HTTPException(404, f"No adapter found at {path}")


@app.get("/history")
async def adapter_history():
    """List saved adapter versions."""
    base = Path(config.base_dir) / "adapters"
    if not base.exists():
        return {"versions": []}

    versions = []
    for d in sorted(base.iterdir()):
        meta_path = d / "adapter_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            versions.append({
                "path": str(d),
                "version": meta.get("adapter_version", 0),
                "steps": meta.get("total_steps", 0),
                "loss": meta.get("last_loss", None),
                "timestamp": meta.get("timestamp", 0),
            })

    return {"versions": versions}


@app.post("/reset")
async def reset_adapter(request: Request):
    """Reset adapter to initial (untrained) state.

    Optional body: {"clear_data": true} to also clear training buffers.
    Default: clears both adapter AND data for a clean slate.
    """
    active_trainer = mlx_trainer or trainer
    if not active_trainer:
        raise HTTPException(400, "No trainer active")

    try:
        body = await request.json()
    except Exception:
        body = {}

    active_trainer.reset_adapter()

    # Clear data buffers by default (opt-out with clear_data=false)
    if body.get("clear_data", True) and data_mgr:
        data_mgr.clear()
        log.info("Training data buffers cleared")

    return sanitize_for_json({"ok": True, "stats": active_trainer.stats()})


# ──────────────────────────────────────────────────────────────
# Lifecycle
# ──────────────────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    log.info(f"Neural Engine Daemon starting on "
             f"{config.daemon_host}:{config.daemon_port}")
    log.info(f"MLX available: {MLX_AVAILABLE}")

    config.resolve_paths()
    config.lms_cli_path = detect_lms_cli()
    if config.lms_cli_path:
        log.info(f"LM Studio CLI: {config.lms_cli_path}")
    else:
        log.warning("LM Studio CLI not found")


@app.on_event("shutdown")
async def on_shutdown():
    log.info("Shutting down...")

    active_trainer = mlx_trainer or trainer
    if active_trainer:
        try:
            active_trainer.save_adapter()
        except Exception as e:
            log.error(f"Failed to save adapter on shutdown: {e}")

    if data_mgr:
        try:
            data_mgr.save_replay()
            data_mgr.save_rolling()
        except Exception as e:
            log.error(f"Failed to save data on shutdown: {e}")

    if active_trainer:
        active_trainer.cleanup()

    log.info("Shutdown complete")


def handle_signal(signum, frame):
    """Handle SIGTERM/SIGINT gracefully."""
    log.info(f"Received signal {signum}, initiating graceful shutdown...")
    active_trainer = mlx_trainer or trainer
    if active_trainer:
        try:
            active_trainer.save_adapter()
        except Exception:
            pass
    sys.exit(0)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Load config from file if exists
    config_path = Path(config.base_dir).expanduser() / "config.json"
    if config_path.exists():
        try:
            loaded = NeuralConfig.load(str(config_path))
            for k, v in loaded.__dict__.items():
                setattr(config, k, v)
            log.info(f"Loaded config from {config_path}")
        except Exception as e:
            log.warning(f"Failed to load config: {e}")

    config.resolve_paths()

    # Override from env
    port = int(os.environ.get("NEURAL_DAEMON_PORT", config.daemon_port))
    host = os.environ.get("NEURAL_DAEMON_HOST", config.daemon_host)

    log.info(f"Starting daemon on {host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()
