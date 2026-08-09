"""
main_cli.py – AI Creator Terminal Application.
Trains new models, fine‑tunes with roles, exports GGUF – all in one file.
Uses autolearn if available.
MADE BY ONLY AND ONLY REHAN AMAN
"""

import os
import re
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Optional, List, Dict

# Core ML imports
try:
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        AutoTokenizer,
        AutoConfig,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
        pipeline,
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType,
    )
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: Core ML libraries not found. Training will be disabled.\n")

# Optional modules
try:
    from autolearn import AutoLearn
    AUTOLEARN_AVAILABLE = True
except ImportError:
    AUTOLEARN_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


# ======================================================================
# Fine‑tuning engine (previously in fine_tuner.py)
# ======================================================================

# Splits equals‑format text into training examples.
def parse_equals_input(text: str) -> List[Dict[str, str]]:
    """Parse a multi‑example string separated by '===' into role dicts."""
    if not text or not text.strip():
        return []

    raw_blocks = re.split(r'\n?===+\n?', text.strip())
    examples = []
    current_system = None

    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue

        segments = re.split(r'(?=^\s*=\w+=)', block, flags=re.MULTILINE)
        system, user, assistant = None, None, None

        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            match = re.match(r'^\s*=(\w+)=\s*(.*)', seg, re.DOTALL)
            if not match:
                logger.warning(f"Unrecognised segment: {seg[:50]}...")
                continue
            role = match.group(1).upper()
            content = match.group(2).strip()

            if role == "SYSTEM":
                system = content
                current_system = content
            elif role == "USER":
                user = content
            elif role == "ASSISTANT":
                assistant = content

        if not system and current_system:
            system = current_system

        if user and assistant:
            examples.append({
                "system": system or "",
                "user": user,
                "assistant": assistant
            })
    return examples

# Converts parsed examples into a HuggingFace Dataset with a "messages" column.
def build_dataset_from_roles(parsed: List[Dict[str, str]]) -> Dataset:
    """Wrap a list of role dicts into a Dataset with a 'messages' field."""
    conversations = []
    for ex in parsed:
        messages = []
        if ex["system"]:
            messages.append({"role": "system", "content": ex["system"]})
        messages.append({"role": "user", "content": ex["user"]})
        messages.append({"role": "assistant", "content": ex["assistant"]})
        conversations.append({"messages": messages})
    return Dataset.from_list(conversations)

# Converts a .gguf file into a HuggingFace‑compatible directory.
def gguf_to_hf(gguf_path: str, base_model_id: str, output_dir: str) -> str:
    """Convert a GGUF file into a HuggingFace model saved to output_dir."""
    try:
        import gguf
    except ImportError as e:
        raise ImportError("gguf package required. Install with: pip install gguf") from e

    os.makedirs(output_dir, exist_ok=True)
    config = AutoConfig.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    reader = gguf.GGUFReader(gguf_path)
    tensors = {tensor.name: torch.from_numpy(tensor.data) for tensor in reader.tensors}

    hf_weights = _map_gguf_to_hf(tensors, config)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model.load_state_dict(hf_weights, strict=False, assign=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    logger.info(f"GGUF converted to {output_dir}")
    return output_dir

# Maps GGUF tensor names to HuggingFace parameter names.
def _map_gguf_to_hf(tensors: Dict[str, torch.Tensor], config) -> Dict[str, torch.Tensor]:
    """Translate raw GGUF tensor names to HuggingFace layer names."""
    hf_weights = {}
    for gguf_name, weight in tensors.items():
        new_name = gguf_name
        m = re.match(r'blk\.(\d+)\.(.+)', gguf_name)
        if m:
            layer_idx, param = m.group(1), m.group(2)
            mapping = {
                "attn_q.weight": f"model.layers.{layer_idx}.self_attn.q_proj.weight",
                "attn_k.weight": f"model.layers.{layer_idx}.self_attn.k_proj.weight",
                "attn_v.weight": f"model.layers.{layer_idx}.self_attn.v_proj.weight",
                "attn_output.weight": f"model.layers.{layer_idx}.self_attn.o_proj.weight",
                "ffn_gate.weight": f"model.layers.{layer_idx}.mlp.gate_proj.weight",
                "ffn_up.weight": f"model.layers.{layer_idx}.mlp.up_proj.weight",
                "ffn_down.weight": f"model.layers.{layer_idx}.mlp.down_proj.weight",
                "attn_norm.weight": f"model.layers.{layer_idx}.input_layernorm.weight",
                "ffn_norm.weight": f"model.layers.{layer_idx}.post_attention_layernorm.weight",
            }
            new_name = mapping.get(param, None)
            if new_name is None:
                logger.warning(f"Skipping unmapped GGUF parameter: {gguf_name}")
                continue
        elif gguf_name == "token_embd.weight":
            new_name = "model.embed_tokens.weight"
        elif gguf_name == "output_norm.weight":
            new_name = "model.norm.weight"
        elif gguf_name == "output.weight":
            new_name = "lm_head.weight"
        # else: keep original name
        hf_weights[new_name] = weight
    return hf_weights

# Converts a HuggingFace model directory to a GGUF file.
def hf_to_gguf(hf_dir: str, output_gguf: str, quant_type: str = "q8_0") -> str:
    """Export a HuggingFace model to a GGUF file with optional quantization."""
    try:
        from llama_cpp.llama_cpp import llama_convert_hf_to_gguf
        llama_convert_hf_to_gguf(model=hf_dir, output=output_gguf, outtype=quant_type)
        return output_gguf
    except (ImportError, Exception):
        script = "convert-hf-to-gguf.py"
        cmd = [script, hf_dir, "--outtype", quant_type, "--outfile", output_gguf]
        subprocess.run(cmd, check=True)
        return output_gguf

# Prepares tokenizer and model for QLoRA fine‑tuning.
def _prepare_model_and_tokenizer(model_id_or_dir: str, use_4bit: bool = True, bf16: bool = True):
    """Load tokenizer and model with optional 4‑bit quantization."""
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_dir, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_dir,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
    )
    return tokenizer, model

# Formats a message list into a text prompt for the tokenizer.
def _format_chat(example, tokenizer):
    """Convert a messages list to a single string using the tokenizer's template."""
    msgs = example["messages"]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(msgs, tokenize=False)
    lines = [f"{m['role'].capitalize()}: {m['content']}" for m in msgs]
    return "\n".join(lines)

# Runs QLoRA fine‑tuning and returns the merged model path.
def run_lora_finetuning(
    base_model_dir: str,
    dataset: Dataset,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    use_4bit: bool = True,
) -> str:
    """Fine‑tune a causal LM with LoRA and return the merged model path."""
    tokenizer, model = _prepare_model_and_tokenizer(base_model_dir, use_4bit)
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    def tokenize(examples):
        texts = [_format_chat(ex, tokenizer) for ex in examples]
        return tokenizer(texts, truncation=True, max_length=2048, padding="max_length")

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        bf16=True,
        logging_steps=10,
        save_steps=200,
        optim="paged_adamw_8bit",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=lambda data: {
            "input_ids": torch.stack([d["input_ids"] for d in data]),
            "attention_mask": torch.stack([d["attention_mask"] for d in data]),
            "labels": torch.stack([d["input_ids"] for d in data]),
        },
    )
    trainer.train()

    model.save_pretrained(os.path.join(output_dir, "adapter"))
    tokenizer.save_pretrained(os.path.join(output_dir, "adapter"))
    merged = model.merge_and_unload()
    merged_path = os.path.join(output_dir, "merged_model")
    merged.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)
    return merged_path

# Applies 4‑bit quantization to a HuggingFace model.
def apply_quantization(model_path: str, output_dir: str, method: str = "autoawq") -> str:
    """Apply 4‑bit quantization to a model directory."""
    if method == "autoawq":
        try:
            from awq import AutoAWQForCausalLM
            model = AutoAWQForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model.quantize(tokenizer, quant_config={"zero_point": True, "q_group_size": 128})
            model.save_quantized(output_dir)
            tokenizer.save_pretrained(output_dir)
            return output_dir
        except ImportError:
            logger.warning("AutoAWQ not found, falling back to bitsandbytes (disk size unchanged).")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    return output_dir

# End‑to‑end fine‑tuning pipeline.
def run_finetuning_pipeline(
    training_text: str,
    base_model_id: str,
    output_dir: str = "./fine_tuned",
    quantize: bool = False,
    export_gguf: bool = False,
    gguf_quant_type: str = "q8_0",
    uploaded_gguf_path: Optional[str] = None,
) -> str:
    """Full fine‑tuning pipeline from equals‑format text to final model."""
    parsed = parse_equals_input(training_text)
    if not parsed:
        raise ValueError("No valid training examples found in the input.")
    dataset = build_dataset_from_roles(parsed)

    # Determine base model directory (HF or converted GGUF)
    if uploaded_gguf_path:
        gguf_hf_dir = os.path.join(output_dir, "gguf_converted")
        base_model_dir = gguf_to_hf(uploaded_gguf_path, base_model_id, gguf_hf_dir)
    else:
        base_model_dir = base_model_id

    merged = run_lora_finetuning(base_model_dir, dataset, output_dir)
    if quantize:
        merged = apply_quantization(merged, os.path.join(output_dir, "quantized"))
    if export_gguf:
        merged = hf_to_gguf(merged, os.path.join(output_dir, "fine_tuned.gguf"), gguf_quant_type)
    return merged


# ======================================================================
# Main CLI Application
# ======================================================================

# Main application class.
class AICreatorCLI:
    """Terminal interface for AI creation and fine‑tuning."""
# Initialise the app state and start the main loop.
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.generator = None
        self.fill_mask = None
        self.trained_model_dir = None
        self.fine_tuned_path = None
        self.active_model_task = None
        self.main_menu()

# Main interactive loop.
    def main_menu(self):
        while True:
            print("\n" + "=" * 40)
            print("AI CREATOR – Terminal Edition")
            print("=" * 40)
            print("1. Train new model (input = output text file)")
            print("2. Fine‑tune with roles (equals‑format)")
            print("3. Export trained model to GGUF")
            print("4. Test prediction")
            print("5. Quit")
            choice = input("\nYour choice: ").strip()

            if choice == "1":
                self.train_new_model()
            elif choice == "2":
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
# Handles the complete training workflow from file to model.
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

# Parses a text file into input‑output pairs.
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

# Resolves the model preset (using AutoLearn if available).
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
        return PRESETS["4"] if len(data) < 100 else PRESETS["2"]

# Performs the actual training and saves the model.
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
# Runs the fine‑tuning pipeline.
    def fine_tune_with_roles(self):
        if not ML_AVAILABLE:
            print("ERROR: Required ML libraries not installed.")
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

        print("Starting fine‑tuning...")
        try:
            result = run_finetuning_pipeline(
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

# Reads multi‑line input (paste or from file).
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

# Auto‑selects a base model for fine‑tuning.
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
# Exports the most recently trained model to a GGUF file.
    def export_gguf(self):
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
        _, quant_type = QUANT_OPTIONS[q_choice]

        out_path = input("Enter output file path (e.g. my_model.gguf): ").strip()
        if not out_path:
            print("No path given. Aborting.")
            return

        try:
            self._convert_to_gguf(src, Path(out_path), quant_type)
            print(f"GGUF file saved to: {out_path}")
        except Exception as e:
            print(f"Export failed: {e}")

# Converts an HF model to a GGUF file using llama.cpp.
    def _convert_to_gguf(self, model_dir, output_path, quant_type):
        converter = self._find_llama_cpp_file("convert_hf_to_gguf.py")
        if not converter:
            raise RuntimeError("llama.cpp not found. Set LLAMA_CPP_DIR environment variable.")
        GGUF_DIR.mkdir(exist_ok=True)
        f16 = GGUF_DIR / "tmp_f16.gguf"
        subprocess.run(["python", str(converter), str(model_dir), "--outfile", str(f16)], check=True)
        if quant_type == "f16":
            shutil.copyfile(f16, output_path)
            return
        quantizer = self._find_llama_cpp_file("llama-quantize") or self._find_llama_cpp_file("quantize")
        if not quantizer:
            raise RuntimeError("llama-quantize executable not found.")
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
# Runs predictions on the standard trained model.
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


if __name__ == "__main__":
    AICreatorCLI()
