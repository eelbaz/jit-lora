"""
export_to_lms.py — Export LoRA adapter back to LM Studio.

Workflow:
  1. Fuse LoRA adapter with base model via MLX
  2. Export to GGUF format
  3. Copy to LM Studio models directory
  4. Load via lms CLI
"""

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("export_to_lms")


def export_adapter_to_lms(config, version: Optional[int] = None) -> dict:
    """Export current LoRA adapter as GGUF to LM Studio.

    Args:
        config: NeuralConfig instance
        version: adapter version tag (auto if None)

    Returns:
        dict with export details
    """
    try:
        import mlx_lm
    except ImportError:
        raise RuntimeError("mlx-lm required for export")

    config.resolve_paths()

    if version is None:
        version = int(time.time()) % 100000

    model_dir = str(Path(config.model_path).parent)
    adapter_dir = config.adapter_dir
    export_name = f"{config.model_key}-tuned-v{version}"
    export_dir = Path(config.base_dir) / "exports" / export_name
    export_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Exporting adapter: {adapter_dir} + {model_dir} → {export_dir}")

    # Step 1: Fuse adapter with base model
    # mlx_lm.fuse writes merged weights to output dir
    try:
        mlx_lm.fuse(
            model=model_dir,
            adapter_path=adapter_dir,
            save_path=str(export_dir / "merged"),
        )
        log.info("LoRA adapter fused with base model")
    except Exception as e:
        log.error(f"Fuse failed: {e}")
        raise

    # Step 2: Convert to GGUF
    gguf_path = export_dir / f"{export_name}.gguf"
    try:
        # Use mlx_lm convert if available
        result = subprocess.run(
            ["python3", "-m", "mlx_lm.convert",
             "--model", str(export_dir / "merged"),
             "--quantize", "--q-bits", "4",
             "-o", str(gguf_path)],
            capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            log.warning(f"GGUF convert failed: {result.stderr}")
            # Fallback: just copy the merged model
            gguf_path = export_dir / "merged"
    except Exception as e:
        log.warning(f"GGUF conversion error: {e}")
        gguf_path = export_dir / "merged"

    # Step 3: Copy to LM Studio models directory
    lms_dest = Path.home() / ".lmstudio" / "models" / "jarvis-tuned" / export_name
    try:
        lms_dest.mkdir(parents=True, exist_ok=True)
        if gguf_path.is_file():
            shutil.copy2(str(gguf_path), str(lms_dest))
        else:
            # Copy directory
            shutil.copytree(str(gguf_path), str(lms_dest), dirs_exist_ok=True)
        log.info(f"Copied to LM Studio: {lms_dest}")
    except Exception as e:
        log.warning(f"Copy to LM Studio failed: {e}")

    # Step 4: Load via lms CLI
    lms = config.lms_cli_path
    if lms:
        try:
            subprocess.run(
                [lms, "load", str(lms_dest)],
                capture_output=True, timeout=120)
            log.info(f"Loaded {export_name} in LM Studio")
        except Exception as e:
            log.warning(f"LM Studio load failed: {e}")

    # Save export metadata
    meta = {
        "export_name": export_name,
        "version": version,
        "source_model": config.model_key,
        "adapter_dir": adapter_dir,
        "gguf_path": str(gguf_path),
        "lms_path": str(lms_dest),
        "timestamp": time.time(),
    }
    with open(export_dir / "export_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta
