from pathlib import Path
from typing import Optional

from .config import APP_DIR
from .ml import hf_to_gguf


def resolve_model_reference(model_ref: Optional[str], *, operation: str = "model") -> str:
    """Support either a local model path or a remote Hugging Face model ID."""
    if model_ref is None or str(model_ref).strip() == "":
        raise ValueError(f"{operation} reference is required.")

    ref = str(model_ref).strip()
    path = Path(ref)
    if path.exists():
        return str(path.resolve())
    if ref.lower().endswith(".gguf"):
        raise FileNotFoundError(f"GGUF model file not found: {ref}")
    return ref


def ensure_model_directory(model_dir: str, *, operation: str = "Model") -> Path:
    """Validate a local model directory or file for quantization/export/fine-tuning."""
    candidate = Path(model_dir)
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"{operation} path does not exist: {model_dir}")


def export_model(model_dir: str, output_gguf: str, quant_type: str = "q8_0") -> str:
    """Public wrapper for exporting a trained model to GGUF."""
    model_dir = str(ensure_model_directory(model_dir, operation="Model to export"))
    output_path = Path(output_gguf)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[terminal] Exporting model from: {model_dir}")
    exported = hf_to_gguf(model_dir, str(output_path), quant_type=quant_type)
    print(f"[terminal] GGUF exported to: {exported}")
    return exported


__all__ = ["resolve_model_reference", "ensure_model_directory", "export_model"]
