import argparse
import os
import sys
from typing import List, Optional

from .autolearn import AutoLearn
from .config import APP_DIR, FINE_TUNED_DIR, ML_AVAILABLE, MODEL_DIR, PEFT_AVAILABLE, PRESETS
from .data import parse_equals_input
from .export import export_model
from .finetune import quantize_model, run_finetuning_pipeline
from .ml import generate_text
from .training import train_model_from_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="creato-ai",
        description="Terminal-first AI Creator library and CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a new model from a file of input=output rows")
    train_parser.add_argument("input_file", help="Path to a file containing 'input = output' rows")
    train_parser.add_argument("--preset", choices=["auto", "llm", "mlm", "slm"], default="auto")
    train_parser.add_argument("--route", choices=["auto", "sklearn", "transformers"], default="auto")
    train_parser.add_argument("--model", default="auto", help="Model name or preset override, e.g. distilgpt2 or logistic-regression")
    train_parser.add_argument("--output-dir", default=str(MODEL_DIR))
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=2)
    train_parser.add_argument("--learning-rate", type=float, default=5e-5)

    finetune_parser = subparsers.add_parser("fine-tune", help="Fine-tune a model from equals-format role examples")
    finetune_parser.add_argument("training_file", help="Path to a file containing equals-formatted role examples")
    finetune_parser.add_argument("--base-model", default="distilgpt2", help="Remote model ID or local path to a model folder or GGUF file")
    finetune_parser.add_argument("--model-dir", dest="model_dir", default=None, help="Explicit local model directory to fine-tune")
    finetune_parser.add_argument("--output-dir", default=str(FINE_TUNED_DIR))
    finetune_parser.add_argument("--quantize", action="store_true")
    finetune_parser.add_argument("--export-gguf", action="store_true")
    finetune_parser.add_argument("--gguf-quant-type", default="q8_0")
    finetune_parser.add_argument("--gguf-base", default=None)

    export_parser = subparsers.add_parser("export", help="Export a trained Hugging Face model to a GGUF file")
    export_parser.add_argument("model_dir", help="Path to the trained model directory")
    export_parser.add_argument("--output", required=True, help="Output GGUF file path")
    export_parser.add_argument("--quant-type", default="q8_0")

    quantize_parser = subparsers.add_parser("quantize", help="Apply quantization to a trained model directory")
    quantize_parser.add_argument("model_dir", help="Path to the trained model directory")
    quantize_parser.add_argument("--output-dir", required=True, help="Directory to save the quantized model")
    quantize_parser.add_argument("--method", default="autoawq", choices=["autoawq", "bnb"])

    test_parser = subparsers.add_parser("test", help="Generate output from a trained model using a prompt passed on the command line")
    test_parser.add_argument("model_dir", help="Path to the trained model directory")
    test_parser.add_argument("prompt", nargs="?", help="Prompt text to send to the model")
    test_parser.add_argument("--prompt", dest="prompt_flag", help="Prompt text to send to the model")
    test_parser.add_argument("--max-new-tokens", type=int, default=60)
    test_parser.add_argument("--temperature", type=float, default=0.7)
    test_parser.add_argument("--top-p", type=float, default=0.95)

    info_parser = subparsers.add_parser("info", help="Print terminal status and dependency information")
    info_parser.add_argument("--verbose", action="store_true")

    return parser


def print_banner() -> None:
    print("=" * 72)
    print("AI Creator - Terminal Library")
    print("Ready for CLI use. No graphical interface is active.")
    print("=" * 72)


def main(argv: Optional[List[str]] = None) -> int:
    print_banner()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "info":
            print(f"Python: {sys.version.split()[0]}")
            print(f"ML available: {ML_AVAILABLE}")
            print(f"PEFT available: {PEFT_AVAILABLE}")
            print(f"AutoLearn available: {True}")
            if args.verbose:
                print(f"APP_DIR: {APP_DIR}")
                print(f"MODEL_DIR: {MODEL_DIR}")
            return 0

        if args.command == "train":
            print("[terminal] Starting training job...")
            model_dir = train_model_from_file(
                args.input_file,
                preset=args.preset,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                route=args.route,
                model=args.model,
            )
            print(f"[terminal] Model saved to {model_dir}")
            return 0

        if args.command == "fine-tune":
            if not os.path.exists(args.training_file):
                raise FileNotFoundError(f"Training file not found: {args.training_file}")
            with open(args.training_file, "r", encoding="utf-8") as handle:
                training_text = handle.read()
            base_model_ref = args.model_dir or args.base_model
            result = run_finetuning_pipeline(
                training_text,
                base_model_id=base_model_ref,
                output_dir=args.output_dir,
                quantize=args.quantize,
                export_gguf=args.export_gguf,
                gguf_quant_type=args.gguf_quant_type,
                uploaded_gguf_path=args.gguf_base,
            )
            print(f"[terminal] Fine-tuning complete using file: {args.training_file}")
            print(f"[terminal] Output directory: {result}")
            return 0

        if args.command == "export":
            exported = export_model(args.model_dir, args.output, quant_type=args.quant_type)
            print(f"[terminal] GGUF exported to {exported}")
            return 0

        if args.command == "quantize":
            result = quantize_model(args.model_dir, args.output_dir, method=args.method)
            print(f"[terminal] Quantized model saved to {result}")
            return 0

        if args.command == "test":
            prompt = args.prompt_flag or args.prompt
            if not prompt:
                parser.error("A prompt is required for the test command. Use --prompt or pass a positional prompt.")
            output = generate_text(
                args.model_dir,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print("[terminal] Prompt:")
            print(prompt)
            print("[terminal] Output:")
            print(output)
            return 0

        parser.error(f"Unsupported command: {args.command}")
        return 2
    except Exception as exc:
        print(f"[terminal] Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "print_banner"]
