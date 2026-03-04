"""
neural_config.py — Configuration and hyperparameters for ANE LoRA training.
"""

from dataclasses import dataclass, field
from pathlib import Path
import json
import os


@dataclass
class NeuralConfig:
    """Training hyperparameters and daemon configuration."""

    # Daemon
    daemon_port: int = 8766
    daemon_host: str = "0.0.0.0"

    # Model (auto-detected from LM Studio)
    model_key: str = ""           # e.g. "qwen3.5-9b-prism"
    model_path: str = ""          # e.g. "~/.lmstudio/models/.../model.gguf"
    model_architecture: str = ""  # e.g. "qwen2"

    # LoRA
    lora_rank: int = 32
    lora_alpha: float = 32.0      # scaling = alpha / rank
    lora_targets: list = field(default_factory=lambda: ["q_proj", "v_proj", "out_proj", "down_proj"])
    lora_dropout: float = 0.0
    lora_num_layers: int = -1     # -1 = all layers, N = last N layers only

    # Training
    training_backend: str = "mlx"  # "mlx" (real autograd) or "ane" (legacy)
    learning_rate: float = 5e-4
    min_learning_rate: float = 5e-5       # cosine LR floor
    cosine_period_steps: int = 5000       # steps for one cosine period
    warmup_fraction: float = 0.1          # warmup as fraction of period
    steps_per_cycle: int = 1              # 1 step per example (epoch-style)
    batch_size: int = 0                   # 0 = all available data in buffer
    epochs_per_cycle: int = 1             # Epochs per auto-training cycle
    train_epochs: int = 15                # Default epochs for manual /train
    early_stop_loss: float = 0.8          # Stop when avg epoch loss drops below
    early_stop_patience: int = 2          # Consecutive low-loss epochs before stop
    max_seq_len: int = 512
    gradient_clip: float = 1.0
    warmup_steps: int = 10
    auto_train: bool = True       # Train after each conversation turn
    replay_ratio: float = 0.3     # 30% replay buffer in each batch

    # Adam optimizer
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    weight_decay: float = 0.0

    # Buffer
    rolling_buffer_size: int = 100   # Recent turns in memory
    replay_buffer_size: int = 500    # Historical turns on disk
    min_response_tokens: int = 10    # Skip training on short responses

    # ANE
    ane_compile_budget: int = 110    # Max compiles before restart
    ane_min_tensor_dim: int = 16     # ANE matmul dims must be multiples of 16
    ane_seq_len: int = 16            # ANE sequence length (must be multiple of 16)

    # Persistence
    base_dir: str = "~/.jarvis/fine-tune"
    adapter_dir: str = ""     # Set dynamically: base_dir/adapters/{model_key}/
    replay_path: str = ""     # Set dynamically: base_dir/replay.jsonl
    auto_save_interval: int = 10  # Save adapter every N training cycles

    # LM Studio
    lms_cli_path: str = ""    # Auto-detected
    lms_api_url: str = "http://localhost:1234"

    @property
    def lora_scaling(self) -> float:
        return self.lora_alpha / self.lora_rank

    def resolve_paths(self):
        """Expand ~ and set dynamic paths."""
        self.base_dir = str(Path(self.base_dir).expanduser())
        if not self.adapter_dir:
            key = self.model_key or "default"
            self.adapter_dir = str(Path(self.base_dir) / "adapters" / key)
        if not self.replay_path:
            self.replay_path = str(Path(self.base_dir) / "replay.jsonl")

        # Auto-detect lms CLI
        if not self.lms_cli_path:
            candidates = [
                Path.home() / ".lmstudio" / "bin" / "lms",
                Path("/usr/local/bin/lms"),
            ]
            for c in candidates:
                if c.exists():
                    self.lms_cli_path = str(c)
                    break

    def ensure_dirs(self):
        """Create required directories."""
        self.resolve_paths()
        Path(self.base_dir).mkdir(parents=True, exist_ok=True)
        Path(self.adapter_dir).mkdir(parents=True, exist_ok=True)

    def save(self, path: str = ""):
        """Save config to JSON."""
        path = path or str(Path(self.base_dir) / "config.json")
        self.resolve_paths()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "NeuralConfig":
        """Load config from JSON."""
        with open(path) as f:
            data = json.load(f)
        cfg = cls()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        cfg.resolve_paths()
        return cfg

    def to_dict(self) -> dict:
        """Convert to dict for API responses."""
        self.resolve_paths()
        d = self.__dict__.copy()
        d["lora_scaling"] = self.lora_scaling
        return d

    def update_from_dict(self, data: dict):
        """Update config from API request."""
        allowed = {
            "learning_rate", "min_learning_rate", "cosine_period_steps",
            "warmup_fraction", "steps_per_cycle", "lora_rank", "lora_alpha",
            "lora_targets", "lora_num_layers", "training_backend",
            "auto_train", "replay_ratio", "gradient_clip", "warmup_steps",
            "rolling_buffer_size", "min_response_tokens", "auto_save_interval",
            "max_seq_len", "lora_dropout", "weight_decay",
            "epochs_per_cycle", "train_epochs",
            "early_stop_loss", "early_stop_patience",
        }
        for k, v in data.items():
            if k in allowed and hasattr(self, k):
                setattr(self, k, v)
