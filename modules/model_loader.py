"""Shared CHGNet model loading helper.

Centralises how every compute module turns a user-chosen model selection into a
loaded CHGNet model object, so the web UI can offer a single "model" picker that
works identically across relaxation, phase stability, mechanical, stress-strain,
and molecular dynamics tasks.

A model selection is one of:
  - a built-in CHGNet pretrained name: "0.3.0", "0.2.0", "r2scan"
  - a filename inside the project's ``models/`` folder (e.g. "my_finetune.pth.tar")
  - an absolute / project-relative path to a ``.pth.tar`` checkpoint
  - None / empty, which falls back to the default "0.3.0"

Built-in names use ``CHGNet.load`` (so the package's own checkpoint resolution
and bias handling apply). Anything else is loaded with ``CHGNet.from_file`` from
the explicit path, which lets users drop their own fine-tuned checkpoints into
``models/`` and select them from the UI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
APP_ROOT = SCRIPT_DIR.parent
MODELS_DIR = APP_ROOT / "models"

BUILTIN_NAMES = {"0.3.0", "0.2.0", "r2scan"}
DEFAULT_MODEL = "0.3.0"


def resolve_model_path(model_arg: str) -> Path | None:
    """Return the on-disk checkpoint path for a non-builtin model selection.

    Returns None for builtin names / empty input (handled by ``CHGNet.load``).
    Raises SystemExit with a clear message if the file cannot be found.
    """
    if not model_arg or model_arg in BUILTIN_NAMES:
        return None
    path = Path(model_arg).expanduser()
    if not path.is_absolute():
        # First try the project models/ folder, then the project root.
        candidate = (MODELS_DIR / path.name).resolve()
        if candidate.is_file():
            return candidate
        candidate = (APP_ROOT / path).resolve()
        if candidate.is_file():
            return candidate
    if not path.is_file():
        raise SystemExit(
            f"Model file not found: {model_arg!r}. "
            f"Place .pth.tar checkpoints in {MODELS_DIR} or pass an absolute path."
        )
    return path.resolve()


def load_chgnet_model(
    chgnet_cls: Any,
    model_arg: str | None = None,
    use_device: str | None = None,
) -> Any:
    """Load a CHGNet model from a user selection.

    Args:
        chgnet_cls: The ``CHGNet`` class (from ``chgnet.model.model``).
        model_arg: Builtin name, checkpoint filename/path, or None for default.
        use_device: Optional device override ("cpu" / "cuda" / "mps").

    Returns:
        A loaded CHGNet model instance.
    """
    selection = (model_arg or "").strip() or DEFAULT_MODEL

    if selection in BUILTIN_NAMES:
        return chgnet_cls.load(model_name=selection, use_device=use_device)

    path = resolve_model_path(selection)
    if path is None:
        return chgnet_cls.load(model_name=DEFAULT_MODEL, use_device=use_device)

    model = chgnet_cls.from_file(str(path))
    if use_device:
        try:
            model = model.to(use_device)
        except Exception:
            pass
    print(f"CHGNet model loaded from file: {path}")
    return model
