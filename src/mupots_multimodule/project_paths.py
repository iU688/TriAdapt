"""Path bindings for the self-contained GraphMLP V8 project.

Datasets and derived MuPoTS tracks remain external.  Runtime code, public/M2
checkpoints, formal V8 checkpoints, and formal results live under PROJECT_ROOT.
Set GRAPHMLP_V8_EXTERNAL_PATHS to use a config other than
config/external_paths.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = PROJECT_ROOT / "runtime"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"


def external_config_path() -> Path:
    override = os.environ.get("GRAPHMLP_V8_EXTERNAL_PATHS")
    return Path(override).expanduser().resolve() if override else (
        PROJECT_ROOT / "config" / "external_paths.json"
    ).resolve()


def external_config() -> dict:
    path = external_config_path()
    return json.loads(path.read_text(encoding="utf-8"))


def source_asset(name: str) -> Path:
    return Path(external_config()["source_assets"][name]).expanduser().resolve()


def target_asset(name: str) -> Path:
    return Path(external_config()["target_assets"][name]).expanduser().resolve()


def m2_checkpoint(seed: int) -> Path:
    names = {
        42: "best_ungated_s42.pth",
        123: "best_ungated_s123.pth",
        2026: "best_ungated_s2026.pth",
    }
    return (CHECKPOINT_ROOT / "m2" / f"seed{seed}" / names[int(seed)]).resolve()


def public_graphmlp_checkpoint() -> Path:
    return (CHECKPOINT_ROOT / "public_graphmlp" / "243frame.pth").resolve()
