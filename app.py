"""
main_cli.py – AI Creator terminal application.
Trains new models, fine‑tunes existing ones, and exports GGUF — all from the command line.
Uses autolearn and fine_tuner if present.
MADE BY ONLY AND ONLY REHAN AMAN
"""

import os
import shutil
import subprocess
from pathlib import Path

# Core ML imports – exit gracefully if missing
try:
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
        pipeline,
    )
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: HuggingFace libraries not found. Only fine‑tuning will be possible.\n")

# Optional modules
try:
    import fine_tuner
    FINE_TUNER_AVAILABLE = True
except ImportError:
    FINE_TUNER_AVAILABLE = False

try:
    from autolearn import AutoLearn
    AUTOLEARN_AVAILABLE = True
except ImportError:
    AUTOLEARN_AVAILABLE = False


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "trained_model"
GGUF_DIR = APP_DIR / "gguf_models"
FINE_TUNED_DIR = APP_DIR / "fine_tuned_model"

PRESETS = {
    "1": {"name": "Auto Learn", "task": "causal", "model": "sshleifer/tiny-gpt2"},
    "2": {"name": "LLM",       "task": "causal", "model": "distilgpt2"},
    "3": {"name": "MLM",       "task": "masked", "model": "distilroberta-base"},
    "4": {"name": "SLM",       "task": "causal", "model": "sshleifer/tiny-gpt2"},
}

QUANT_OPTIONS = {
    "1": ("4q", "q4_0"),
    "2": ("8q", "q8_0"),
    "3": ("16q", "f16"),
    "4": ("36q", "q6_k"),
}


# Main CLI application class
class AICreatorCLI:
    """Terminal interface for AI creation and fine‑tuning."""
# Initialise the application state and start the main loop.
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.generator = None
        self.fill_mask = None
        self.trained_model_dir = None       # path to latest trained HF model
        self.fine_tuned_path = None         # path to fine‑tuned model (HF or GGUF)
        self.active_model_task = None       # "causal" or "masked"
        self.main_menu()
# Main interactive loop – prints menu and handles user choice.
    def main_menu(self):
        while True:
            print("\n" + "=" * 40)
            print("AI CREATOR – Terminal Edition")
            print("=" * 40)
            print("1. Train new model (input = output text file)")
            if FINE_TUNER_AVAILABLE:
                print("2. Fine‑tune with roles (equals‑format)")
            print("3. Export trained model to GGUF")
            print("4. Test prediction")
            print("5. Quit")
            choice = input("\nYour choice: ").strip()

            if choice == "1":
                self.train_new_model()
            elif choice == "2" and FINE_TUNER_AVAILABLE:
                self.fine_tune_with_roles()
            elif choice == "3":
                self.export_gguf()
            elif choice == "4":
                self.test_prediction()
            elif choice == "5":
                print("Goodbye!")
                break
            else:
                print("Invalid option. Please try again.")

# ---------- 1. Standard training ----------
# Handles the entire workflow: file selection, model type, training.
    def train_new_model(self):
        if not ML_AVAILABLE:
            print("ERROR: Required ML libraries (transformers, datasets) not installed.")
            return

        file_path = input("Enter path to training text file (input = output per line): ").strip()
        if not file_path or not os.path.isfile(file_path):
            print("File not found.")
            return

        data = self._parse_file(file_path)
        if not data:
            print("No valid training pairs found. Each line must have 'input = output'.")
            return

        print("\nChoose base model type:")
        for key, pres in PRESETS.items():
            print(f"  {key}. {pres['name']}")
        model_choice = input("Enter number (default 1): ").strip() or "1"
        if model_choice not in PRESETS:
            print("Invalid choice, using Auto Learn.")
            model_choice = "1"

        # Resolve preset (using AutoLearn if possible)
        if model_choice == "1":
            preset = self._resolve_preset(data)
        else:
            preset = PRESETS[model_choice]

        print(f"\nSelected model: {preset['name']} ({preset['model']})")
        print("Starting training (this may take a while)...")
        try:
            self._train_model(preset, data)
            print("Training completed successfully!")
        except Exception as e:
            print(f"Training failed: {e}")
# Parses the input=output file into a list of dicts.
    def _parse_file(self, file_path):
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "=" not in line:
                    continue
                inp, out = line.split("=", 1)
                inp, out = inp.strip(), out.strip()
                if inp and out:
                    data.append({"input": inp, "output": out})
        return data
# Uses AutoLearn (or fallback heuristic) to pick a model.
    def _resolve_preset(self, data):
        if AUTOLEARN_AVAILABLE:
            try:
                learner = AutoLearn()
                pick = learner.choose_model(data)
                for key, pres in PRESETS.items():
                    if pres["name"] == pick and pick != "Auto Learn":
                        return pres
            except Exception:
                pass
        # Fallback
        return PRESETS["4"] if len(data) < 100 else PRESETS["2"]
# Loads the base model, tokenises, and trains on the dataset.
    def _train_model(self, preset, data):
        task = preset["task"]
        model_id = preset["model"]

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.mask_token

        records = []
        for row in data:
            if task == "masked":
                records.append({"text": f"{row['input']} {self.tokenizer.mask_token} {row['output']}"})
            else:
                records.append({"text": f"Input: {row['input']}\nOutput: {row['output']}"})

        dataset = Dataset.from_list(records)
        tokenized = dataset.map(
            lambda b: self.tokenizer(b["text"], truncation=True, padding="max_length", max_length=128),
            batched=True, remove_columns=["text"],
        )
        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=(task == "masked"), mlm_probability=0.15
        )

        if task == "masked":
            self.model = AutoModelForMaskedLM.from_pretrained(model_id)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id)
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        args = TrainingArguments(
            output_dir=str(MODEL_DIR),
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            save_strategy="no",
            logging_steps=5,
            report_to=[],
            learning_rate=5e-5,
        )
        trainer = Trainer(model=self.model, args=args, train_dataset=tokenized, data_collator=collator)
        trainer.train()

        MODEL_DIR.mkdir(exist_ok=True)
        self.model.save_pretrained(MODEL_DIR)
        self.tokenizer.save_pretrained(MODEL_DIR)
        self.trained_model_dir = MODEL_DIR
        self.active_model_task = task

        if task == "masked":
            self.fill_mask = pipeline("fill-mask", model=self.model, tokenizer=self.tokenizer)
            self.generator = None
        else:
            self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
            self.fill_mask = None

# ---------- 2. Fine‑tuning ----------
# Runs the fine‑tuning pipeline using fine_tuner module.
    def fine_tune_with_roles(self):
        if not FINE_TUNER_AVAILABLE:
            print("fine_tuner module not found. Cannot fine‑tune.")
            return

        print("\nPaste your training examples (SYSTEM/USER/ASSISTANT, separated by '===').")
        print("Type 'END' on a new line to finish input, or enter a file path to read from.")
        text = self._read_multiline_input()
        if not text:
            print("No examples provided. Aborting.")
            return

        print("\nChoose base model for fine‑tuning:")
        print("  1. Auto Learn")
        print("  2. LLM (distilgpt2)")
        print("  3. SLM (tiny-gpt2)")
        model_choice = input("Enter number (default 1): ").strip() or "1"
        if model_choice == "1":
            base_model = self._auto_select_fine_tune_model()
        elif model_choice == "2":
            base_model = "distilgpt2"
        elif model_choice == "3":
            base_model = "sshleifer/tiny-gpt2"
        else:
            print("Invalid, using LLM.")
            base_model = "distilgpt2"

        # Optional GGUF upload
        use_gguf = input("Use a local .gguf file as base? (y/n): ").strip().lower() == "y"
        gguf_path = None
        arch_id = base_model
        if use_gguf:
            gguf_path = input("Enter path to .gguf file: ").strip()
            if not os.path.isfile(gguf_path):
                print("File not found. Switching to default base model.")
                gguf_path = None
            else:
                arch_id = input("Enter HuggingFace architecture ID (e.g. meta-llama/Llama-2-7b-hf): ").strip()
                if not arch_id:
                    print("Architecture ID required. Using default.")
                    gguf_path = None

        quantize = input("Apply 4‑bit quantization after training? (y/n): ").strip().lower() == "y"
        export_gguf = input("Export as GGUF after training? (y/n): ").strip().lower() == "y"

        print("Starting fine‑tuning... (this may take many minutes)")
        try:
            result = fine_tuner.run_finetuning_pipeline(
                training_text=text,
                base_model_id=arch_id if gguf_path else base_model,
                output_dir=str(FINE_TUNED_DIR),
                quantize=quantize,
                export_gguf=export_gguf,
                uploaded_gguf_path=gguf_path,
            )
            self.fine_tuned_path = result
            print(f"Fine‑tuning completed! Model saved to: {result}")
        except Exception as e:
            print(f"Fine‑tuning failed: {e}")
# Reads multi‑line input (either paste or from file).
    def _read_multiline_input(self):
        first_line = input().strip()
        if os.path.isfile(first_line):
            with open(first_line, "r", encoding="utf-8") as f:
                return f.read().strip()
        lines = [first_line]
        while True:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        return "\n".join(lines)
# Auto‑selects a base model for fine‑tuning (simple heuristic).
    def _auto_select_fine_tune_model(self):
        if AUTOLEARN_AVAILABLE:
            try:
                learner = AutoLearn()
                pick = learner.choose_model([])
                if pick in ["LLM", "SLM"]:
                    return PRESETS["2"]["model"] if pick == "LLM" else PRESETS["4"]["model"]
            except Exception:
                pass
        return "distilgpt2"

# ---------- 3. GGUF Export ----------
# Exports the most recent trained model (standard or fine‑tuned) as a GGUF file.
    def export_gguf(self):
        # Determine source directory
        src = None
        if self.trained_model_dir and self.trained_model_dir.exists():
            src = self.trained_model_dir
        elif self.fine_tuned_path and os.path.isdir(self.fine_tuned_path):
            src = Path(self.fine_tuned_path)
        else:
            print("No trained model available. Train or fine‑tune first.")
            return

        print("\nChoose quantization level:")
        for key, (label, _) in QUANT_OPTIONS.items():
            print(f"  {key}. {label}")
        q_choice = input("Enter number (default 3 = 16q): ").strip() or "3"
        if q_choice not in QUANT_OPTIONS:
            print("Invalid, using 16q.")
            q_choice = "3"
        quant_label, quant_type = QUANT_OPTIONS[q_choice]

        out_path = input("Enter output file path (e.g. my_model.gguf): ").strip()
        if not out_path:
            print("No path given. Aborting.")
            return

        try:
            self._convert_to_gguf(src, Path(out_path), quant_type)
            print(f"GGUF file saved to: {out_path}")
        except Exception as e:
            print(f"Export failed: {e}")
# Converts HF model to GGUF using llama.cpp tools.
    def _convert_to_gguf(self, model_dir, output_path, quant_type):
        converter = self._find_llama_cpp_file("convert_hf_to_gguf.py")
        if not converter:
            raise RuntimeError("llama.cpp not found. Set LLAMA_CPP_DIR to the llama.cpp folder.")
        GGUF_DIR.mkdir(exist_ok=True)
        f16 = GGUF_DIR / "tmp_f16.gguf"
        subprocess.run(["python", str(converter), str(model_dir), "--outfile", str(f16)], check=True)
        if quant_type == "f16":
            shutil.copyfile(f16, output_path)
            return
        quantizer = self._find_llama_cpp_file("llama-quantize") or self._find_llama_cpp_file("quantize")
        if not quantizer:
            raise RuntimeError("llama-quantize executable not found. Build llama.cpp or set LLAMA_CPP_DIR.")
        subprocess.run([str(quantizer), str(f16), str(output_path), quant_type], check=True)
# Searches for a file inside the llama.cpp folder.
    def _find_llama_cpp_file(self, name):
        search_dirs = [APP_DIR / "llama.cpp", Path(os.environ.get("LLAMA_CPP_DIR", ""))]
        for root in search_dirs:
            if not root or not root.exists():
                continue
            hits = list(root.rglob(name))
            if hits:
                return hits[0]
        return None

# ---------- 4. Test prediction ----------
# Runs a prediction using the currently loaded standard model.
    def test_prediction(self):
        if self.model is None or self.tokenizer is None:
            print("No standard model available. Train a model first (option 1).")
            return

        while True:
            prompt = input("\nEnter prompt (or 'exit' to return to menu): ").strip()
            if prompt.lower() == "exit":
                break
            if not prompt:
                continue
            try:
                if self.fill_mask:
                    out = self.fill_mask(f"{prompt} {self.tokenizer.mask_token}", top_k=1)[0]["sequence"]
                    print(f"Result: {out}")
                else:
                    gen = self.generator(
                        f"Input: {prompt}\nOutput:",
                        max_new_tokens=60,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )[0]["generated_text"]
                    result = gen.split("Output:", 1)[-1].strip()
                    print(f"Output: {result}")
            except Exception as e:
                print(f"Prediction error: {e}")


# Application entry point
if __name__ == "__main__":
    AICreatorCLI()
