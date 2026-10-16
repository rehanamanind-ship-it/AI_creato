#!/usr/bin/env python3
"""Compatibility wrapper for the terminal-first AI Creator package."""

from src.creato_ai.cli import main
from src.creato_ai.autolearn import AutoLearn
from src.creato_ai.data import parse_equals_input, parse_training_file
from src.creato_ai.export import export_model
from src.creato_ai.finetune import apply_quantization, quantize_model, run_finetuning_pipeline, run_lora_finetuning
from src.creato_ai.training import choose_training_route, resolve_preset, train_model_from_file, train_sklearn_model
from src.creato_ai.config import APP_DIR, FINE_TUNED_DIR, GGUF_DIR, MODEL_DIR, PRESETS, QUANT_OPTIONS
from src.creato_ai.ml import generate_text, gguf_to_hf, hf_to_gguf
from src.creato_ai.cli import build_parser

__all__ = [
    "APP_DIR",
    "FINE_TUNED_DIR",
    "GGUF_DIR",
    "MODEL_DIR",
    "PRESETS",
    "QUANT_OPTIONS",
    "AutoLearn",
    "apply_quantization",
    "build_parser",
    "export_model",
    "generate_text",
    "gguf_to_hf",
    "hf_to_gguf",
    "main",
    "parse_equals_input",
    "parse_training_file",
    "quantize_model",
    "resolve_preset",
    "run_finetuning_pipeline",
    "run_lora_finetuning",
    "train_model_from_file",
    "train_sklearn_model",
    "choose_training_route",
]

if __name__ == "__main__":
    raise SystemExit(main())
