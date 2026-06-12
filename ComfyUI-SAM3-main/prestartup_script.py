"""ComfyUI-SAM3 Prestartup Script."""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
COMFYUI_DIR = SCRIPT_DIR.parent.parent

try:
    from comfy_env import setup_env, copy_files
    setup_env()
    copy_files(SCRIPT_DIR / "assets", COMFYUI_DIR / "input")
except Exception:
    # Fallback: copy assets manually
    import shutil
    src = SCRIPT_DIR / "assets"
    dst = COMFYUI_DIR / "input"
    if src.is_dir():
        for f in src.iterdir():
            if f.is_file():
                shutil.copy2(f, dst / f.name)
