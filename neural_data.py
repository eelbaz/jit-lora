"""
neural_data.py — Training data manager for ANE LoRA fine-tuning.

Manages a rolling buffer of recent conversation turns and a persistent
replay buffer for anti-catastrophic-forgetting experience replay.
"""

import json
import random
import time
from collections import deque
from pathlib import Path
from typing import Optional


class TrainingExample:
    """A single training example (conversation turn)."""

    __slots__ = ("messages", "timestamp", "token_count", "session_id")

    def __init__(self, messages: list[dict], timestamp: float = 0,
                 token_count: int = 0, session_id: str = ""):
        self.messages = messages
        self.timestamp = timestamp or time.time()
        self.token_count = token_count
        self.session_id = session_id

    def to_dict(self) -> dict:
        return {
            "messages": self.messages,
            "timestamp": self.timestamp,
            "token_count": self.token_count,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingExample":
        return cls(
            messages=d["messages"],
            timestamp=d.get("timestamp", 0),
            token_count=d.get("token_count", 0),
            session_id=d.get("session_id", ""),
        )


class TrainingDataManager:
    """Manages rolling buffer + persistent replay for LoRA training."""

    def __init__(self, rolling_size: int = 100, replay_size: int = 500,
                 replay_path: str = "", min_response_tokens: int = 10):
        self.rolling_size = rolling_size
        self.replay_size = replay_size
        self.min_response_tokens = min_response_tokens
        self.replay_path = replay_path

        self._rolling: deque[TrainingExample] = deque(maxlen=rolling_size)
        self._replay: list[TrainingExample] = []
        self._total_added = 0

        if replay_path:
            self._load_replay()

    @property
    def rolling_count(self) -> int:
        return len(self._rolling)

    @property
    def replay_count(self) -> int:
        return len(self._replay)

    @property
    def total_added(self) -> int:
        return self._total_added

    def add_turn(self, user_text: str, assistant_text: str,
                 system_prompt: str = "", session_id: str = "") -> bool:
        """Add a conversation turn to the training buffer.

        Returns True if the example was accepted (not filtered).
        """
        # Quality filter: skip short/empty responses
        approx_tokens = len(assistant_text.split())
        if approx_tokens < self.min_response_tokens:
            return False

        # Skip tool-only or empty content
        if not assistant_text.strip():
            return False

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": assistant_text})

        example = TrainingExample(
            messages=messages,
            token_count=approx_tokens,
            session_id=session_id,
        )

        self._rolling.append(example)
        self._total_added += 1

        # Add to replay with reservoir sampling
        if len(self._replay) < self.replay_size:
            self._replay.append(example)
        else:
            idx = random.randint(0, self._total_added - 1)
            if idx < self.replay_size:
                self._replay[idx] = example

        return True

    def get_training_batch(self, batch_size: int = 1,
                           replay_ratio: float = 0.3) -> list[TrainingExample]:
        """Get a training batch mixing recent and replay examples.

        Args:
            batch_size: Total examples in batch. 0 = all available data.
            replay_ratio: Fraction of batch from replay buffer (0.0-1.0)

        Returns:
            List of TrainingExample
        """
        if not self._rolling:
            return []

        # batch_size <= 0 means "all available data"
        if batch_size <= 0:
            batch = list(self._rolling)
            if self._replay:
                # Add replay examples not already in rolling
                rolling_set = {id(ex) for ex in self._rolling}
                for ex in self._replay:
                    if id(ex) not in rolling_set:
                        batch.append(ex)
            random.shuffle(batch)
            return batch

        n_replay = int(batch_size * replay_ratio)
        n_recent = batch_size - n_replay

        batch = []

        # Recent examples (most recent first)
        recent = list(self._rolling)
        if n_recent > 0:
            recent_sample = recent[-n_recent:] if len(recent) >= n_recent else recent
            batch.extend(recent_sample)

        # Replay examples (random sample)
        if n_replay > 0 and self._replay:
            replay_sample = random.sample(
                self._replay,
                min(n_replay, len(self._replay))
            )
            batch.extend(replay_sample)

        random.shuffle(batch)
        return batch

    def get_recent(self, n: int = 5) -> list[TrainingExample]:
        """Get the N most recent training examples."""
        return list(self._rolling)[-n:]

    def save_rolling(self, path: str = ""):
        """Save rolling buffer to disk."""
        path = path or str(Path(self.replay_path).parent / "buffer.jsonl")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for ex in self._rolling:
                f.write(json.dumps(ex.to_dict()) + "\n")

    def load_rolling(self, path: str = ""):
        """Load rolling buffer from disk."""
        path = path or str(Path(self.replay_path).parent / "buffer.jsonl")
        if not Path(path).exists():
            return
        self._rolling.clear()
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    ex = TrainingExample.from_dict(json.loads(line))
                    self._rolling.append(ex)

    def save_replay(self):
        """Persist replay buffer to disk."""
        if not self.replay_path:
            return
        Path(self.replay_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.replay_path, "w") as f:
            for ex in self._replay:
                f.write(json.dumps(ex.to_dict()) + "\n")

    def _load_replay(self):
        """Load replay buffer from disk."""
        if not self.replay_path or not Path(self.replay_path).exists():
            return
        self._replay.clear()
        with open(self.replay_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    ex = TrainingExample.from_dict(json.loads(line))
                    self._replay.append(ex)
        # Trim to max size
        if len(self._replay) > self.replay_size:
            self._replay = random.sample(self._replay, self.replay_size)

    def clear(self):
        """Clear all buffers (for reset)."""
        self._rolling.clear()
        self._replay.clear()
        self._total_added = 0

    def stats(self) -> dict:
        """Return buffer statistics."""
        return {
            "rolling_count": self.rolling_count,
            "rolling_capacity": self.rolling_size,
            "replay_count": self.replay_count,
            "replay_capacity": self.replay_size,
            "total_added": self._total_added,
        }
