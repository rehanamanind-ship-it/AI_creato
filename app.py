#!/usr/bin/env python3
"""
app.py – AI Creator Curses Terminal Application.
Fully interactive TUI for training, fine‑tuning, and GGUF export.
Uses autolearn if available.
"""

import os
import re
import shutil
import subprocess
import sys
import curses
import logging
import importlib.util
from pathlib import Path
from typing import Optional, List, Dict

# ---------- Core ML imports (graceful fallback) ----------
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
    ML_AVAILABLE = True
except ImportError:
    Dataset = None
    ML_AVAILABLE = False

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType,
    )
except ImportError:
    PEFT_AVAILABLE = False
else:
    PEFT_AVAILABLE = True

BITSANDBYTES_AVAILABLE = importlib.util.find_spec("bitsandbytes") is not None

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

MIN_WIDTH = 80
MIN_HEIGHT = 24


# Helper to ensure Dataset is available.
def _require_dataset():
    if Dataset is None:
        raise ImportError("The 'datasets' library is required. Install with: pip install datasets")


def _require_finetuning_dependencies():
    _require_dataset()
    if not PEFT_AVAILABLE:
        raise ImportError("Fine-tuning requires 'peft'. Install with: pip install peft")


# ======================================================================
# Fine‑tuning engine (integrated, no external imports needed)
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
            examples.append({"system": system or "", "user": user, "assistant": assistant})
    return examples

# Converts parsed examples into a Dataset.
def build_dataset_from_roles(parsed: List[Dict[str, str]]):
    """Wrap role dicts into a Dataset with a 'messages' field."""
    _require_dataset()
    conversations = []
    for ex in parsed:
        messages = []
        if ex["system"]:
            messages.append({"role": "system", "content": ex["system"]})
        messages.append({"role": "user", "content": ex["user"]})
        messages.append({"role": "assistant", "content": ex["assistant"]})
        conversations.append({"messages": messages})
    return Dataset.from_list(conversations)

# Converts a .gguf file to a HuggingFace directory.
def gguf_to_hf(gguf_path: str, base_model_id: str, output_dir: str) -> str:
    """Convert GGUF file to HF model saved to output_dir."""
    try:
        import gguf
    except ImportError as e:
        raise ImportError("gguf package required. Install: pip install gguf") from e
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
    return output_dir

# Maps GGUF tensor names to HuggingFace layer names.
def _map_gguf_to_hf(tensors, config):
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
            new_name = mapping.get(param)
            if new_name is None:
                continue
        elif gguf_name == "token_embd.weight":
            new_name = "model.embed_tokens.weight"
        elif gguf_name == "output_norm.weight":
            new_name = "model.norm.weight"
        elif gguf_name == "output.weight":
            new_name = "lm_head.weight"
        # else keep original
        hf_weights[new_name] = weight
    return hf_weights

def _find_llama_cpp_file(name: str):
    """Find a llama.cpp tool in the bundled checkout or LLAMA_CPP_DIR."""
    search_dirs = [APP_DIR / "llama.cpp"]
    configured_dir = os.environ.get("LLAMA_CPP_DIR")
    if configured_dir:
        search_dirs.append(Path(configured_dir))

    for root in search_dirs:
        if root.exists():
            hits = list(root.rglob(name))
            if hits:
                return hits[0]
    return None


# Converts HF model to GGUF file.
def hf_to_gguf(hf_dir: str, output_gguf: str, quant_type: str = "q8_0") -> str:
    """Export HF model to GGUF with optional quantization."""
    converter = _find_llama_cpp_file("convert_hf_to_gguf.py")
    if not converter:
        raise RuntimeError("llama.cpp converter not found. Set LLAMA_CPP_DIR to its checkout.")
    output_path = Path(output_gguf)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(converter), hf_dir, "--outtype", quant_type, "--outfile", str(output_path)],
        check=True,
    )
    return str(output_path)

# Prepares tokenizer and model for QLoRA.
def _prepare_model_and_tokenizer(model_id_or_dir: str, use_4bit=True, bf16=True):
    """Load tokenizer and 4‑bit quantized model."""
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    cuda_available = torch.cuda.is_available()
    if use_4bit and not (cuda_available and BITSANDBYTES_AVAILABLE):
        raise RuntimeError("4-bit loading requires a CUDA GPU and the 'bitsandbytes' package.")
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
        )
    torch_dtype = torch.bfloat16 if bf16 else (torch.float16 if cuda_available else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_dir,
        quantization_config=bnb_config,
        device_map="auto" if cuda_available else None,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model

# Formats chat message list into a single string.
def _format_chat(example, tokenizer):
    """Convert messages list to string using tokenizer's template."""
    msgs = example["messages"]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(msgs, tokenize=False)
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in msgs)

# Runs LoRA fine‑tuning and returns merged model path.
def run_lora_finetuning(base_model_dir, dataset, output_dir, epochs=3, batch_size=4, lr=2e-4, use_4bit=True):
    """Fine‑tune with QLoRA and return merged model directory."""
    _require_finetuning_dependencies()
    cuda_available = torch.cuda.is_available()
    use_4bit = use_4bit and cuda_available and BITSANDBYTES_AVAILABLE
    supports_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)
    use_bf16 = cuda_available and supports_bf16()
    use_fp16 = cuda_available and not use_bf16
    os.makedirs(output_dir, exist_ok=True)

    tokenizer, model = _prepare_model_and_tokenizer(base_model_dir, use_4bit, use_bf16)
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules="all-linear",
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    def tokenize(batch):
        texts = [_format_chat({"messages": messages}, tokenizer) for messages in batch["messages"]]
        return tokenizer(texts, truncation=True, max_length=2048)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=epochs,
        learning_rate=lr,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=10,
        save_steps=200,
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        report_to="none",
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
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
def apply_quantization(model_path: str, output_dir: str, method="autoawq") -> str:
    """Quantize a model directory."""
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
            pass
    if not (torch.cuda.is_available() and BITSANDBYTES_AVAILABLE):
        raise RuntimeError("4-bit quantization requires a CUDA GPU and the 'bitsandbytes' package.")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    return output_dir

# End‑to‑end fine‑tuning pipeline.
def run_finetuning_pipeline(training_text, base_model_id, output_dir="./fine_tuned", quantize=False, export_gguf=False, gguf_quant_type="q8_0", uploaded_gguf_path=None):
    """Full fine‑tuning pipeline from equals‑format text."""
    _require_finetuning_dependencies()
    parsed = parse_equals_input(training_text)
    if not parsed:
        raise ValueError("No valid training examples found.")
    dataset = build_dataset_from_roles(parsed)
    if uploaded_gguf_path:
        base_model_dir = gguf_to_hf(uploaded_gguf_path, base_model_id, os.path.join(output_dir, "gguf_converted"))
    else:
        base_model_dir = base_model_id
    merged = run_lora_finetuning(base_model_dir, dataset, output_dir)
    if quantize:
        merged = apply_quantization(merged, os.path.join(output_dir, "quantized"))
    if export_gguf:
        merged = hf_to_gguf(merged, os.path.join(output_dir, "fine_tuned.gguf"), gguf_quant_type)
    return merged


# ======================================================================
# Curses Application Class
# ======================================================================

class AICreatorTUI:
    """Curses‑based terminal interface for AI Creator."""
# Initialise the TUI, state, and start the main loop.
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.model = None
        self.tokenizer = None
        self.generator = None
        self.fill_mask = None
        self.trained_model_dir = None
        self.fine_tuned_path = None
        self.active_task = None
        self.status_message = "Ready. Use arrow keys to navigate, Enter to select."
        self.running = True
        self._init_colors()
        self._check_terminal_size()
        self._main_loop()

# Checks if terminal meets minimum size requirements.
    def _check_terminal_size(self):
        h, w = self.stdscr.getmaxyx()
        if w < MIN_WIDTH or h < MIN_HEIGHT:
            self.stdscr.addstr(0, 0, f"Terminal too small. Need at least {MIN_WIDTH}x{MIN_HEIGHT}.")
            self.stdscr.refresh()
            self.stdscr.getch()
            raise SystemExit("Terminal too small.")

# Setup curses colour pairs.
    def _init_colors(self):
        curses.start_color()
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)   # default
        curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)   # highlight
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # title
        curses.init_pair(4, curses.COLOR_GREEN, curses.COLOR_BLACK)   # status

# Main menu display and navigation.
    def _main_loop(self):
        current_row = 0
        menu_items = [
            "Train new model (input = output file)",
            "Fine‑tune with roles (equals format)",
            "Export model to GGUF",
            "Test prediction",
            "Quit",
        ]
        while self.running:
            self._draw_screen(menu_items, current_row)
            key = self.stdscr.getch()
            if key == curses.KEY_UP and current_row > 0:
                current_row -= 1
            elif key == curses.KEY_DOWN and current_row < len(menu_items)-1:
                current_row += 1
            elif key in (curses.KEY_ENTER, 10, 13):
                self._handle_selection(current_row)
            elif key == ord('q'):
                self.running = False

# Draws the entire screen.
    def _draw_screen(self, items, selected):
        self.stdscr.clear()
        h, w = self.stdscr.getmaxyx()
        # Title
        title = "AI CREATOR – Terminal Edition"
        self.stdscr.addstr(1, max(0, (w-len(title))//2), title, curses.color_pair(3) | curses.A_BOLD)
        # Menu
        for idx, item in enumerate(items):
            x = w//2 - len(item)//2
            y = 4 + idx
            if idx == selected:
                self.stdscr.attron(curses.color_pair(2))
                self.stdscr.addstr(y, x, item)
                self.stdscr.attroff(curses.color_pair(2))
            else:
                self.stdscr.addstr(y, x, item)
        # Status bar
        status = self.status_message[:w-1]
        self.stdscr.addstr(h-2, 0, status, curses.color_pair(4))
        self.stdscr.refresh()

# Handles menu selection – Quit sets running flag instead of exit().
    def _handle_selection(self, row):
        if row == 0:
            self._train_new_model()
        elif row == 1:
            self._fine_tune_with_roles()
        elif row == 2:
            self._export_gguf()
        elif row == 3:
            self._test_prediction()
        elif row == 4:
            self.running = False

# Opens a centered pop‑up window to safely get user input inside curses.
    def _input_popup(self, title, default=""):
        """Display a pop‑up dialog to collect a single string from the user."""
        h, w = self.stdscr.getmaxyx()
        pw, ph = min(max(60, len(title) + 4), w - 2), 3
        y = (h - ph) // 2
        x = (w - pw) // 2
        popup = curses.newwin(ph, pw, y, x)
        popup.bkgd(' ', curses.color_pair(1))
        popup.box()
        popup.addnstr(0, 2, title, pw - 4)
        curses.echo()
        curses.curs_set(1)
        result = ""
        try:
            popup.move(1, 2)
            result = popup.getstr(1, 2, pw-4).decode("utf-8").strip()
        except Exception:
            pass
        finally:
            curses.noecho()
            curses.curs_set(0)
            del popup
            self.stdscr.touchwin()
            self.stdscr.refresh()
        return result if result else default

# Opens a larger pop‑up window for multi‑line input (fine‑tuning examples).
    def _multiline_popup(self, title):
        """Display a pop‑up that captures multiple lines until empty line is entered."""
        h, w = self.stdscr.getmaxyx()
        pw, ph = 70, 15
        y = (h - ph) // 2
        x = (w - pw) // 2
        popup = curses.newwin(ph, pw, y, x)
        popup.bkgd(' ', curses.color_pair(1))
        popup.box()
        popup.addnstr(0, 2, title, pw - 4)
        popup.addstr(ph-2, 2, "Press Enter on empty line to finish.")
        curses.echo()
        curses.curs_set(1)
        lines = []
        current_line = 1
        try:
            while True:
                popup.move(current_line, 2)
                line = popup.getstr(current_line, 2, pw-4).decode("utf-8").strip()
                if not line:
                    break
                lines.append(line)
                current_line += 1
                if current_line >= ph-2:
                    break
        except Exception:
            pass
        finally:
            curses.noecho()
            curses.curs_set(0)
            del popup
            self.stdscr.touchwin()
            self.stdscr.refresh()
        return "\n".join(lines)

# ---------- Training ----------
    def _train_new_model(self):
        if not ML_AVAILABLE:
            self._show_message("ML libraries not installed. Cannot train.", error=True)
            return
        file_path = self._input_popup("Training file path (input = output per line): ")
        if not file_path or not os.path.isfile(file_path):
            self._show_message("File not found.", error=True)
            return
        data = self._parse_file(file_path)
        if not data:
            self._show_message("No valid training pairs found.", error=True)
            return
        self._show_message(f"Loaded {len(data)} examples. Choose model type.")
        choice = self._input_popup("1-Auto 2-LLM 3-MLM 4-SLM (default 1): ", "1")
        if choice not in PRESETS:
            choice = "1"
        if choice == "1":
            preset = self._resolve_preset(data)
        else:
            preset = PRESETS[choice]
        self._show_message(f"Training with {preset['name']}... Please wait.")
        try:
            self._train_model(preset, data)
            self._show_message("Training completed! Model saved.")
        except Exception as e:
            self._show_message(f"Training failed: {e}", error=True)

    def _parse_file(self, path):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if "=" not in line:
                    continue
                inp, out = line.split("=", 1)
                inp, out = inp.strip(), out.strip()
                if inp and out:
                    data.append({"input": inp, "output": out})
        return data

    def _resolve_preset(self, data):
        if AUTOLEARN_AVAILABLE:
            try:
                learner = AutoLearn()
                pick = learner.choose_model(data)
                for k, v in PRESETS.items():
                    if v["name"] == pick and pick != "Auto Learn":
                        return v
            except Exception:
                pass
        return PRESETS["4"] if len(data) < 100 else PRESETS["2"]

    def _train_model(self, preset, data):
        _require_dataset()
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
        collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=(task=="masked"), mlm_probability=0.15)
        if task == "masked":
            self.model = AutoModelForMaskedLM.from_pretrained(model_id)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id)
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        args = TrainingArguments(output_dir=str(MODEL_DIR), overwrite_output_dir=True, num_train_epochs=3,
                                 per_device_train_batch_size=2, save_strategy="no", logging_steps=5, report_to=[],
                                 learning_rate=5e-5)
        Trainer(model=self.model, args=args, train_dataset=tokenized, data_collator=collator).train()
        MODEL_DIR.mkdir(exist_ok=True)
        self.model.save_pretrained(MODEL_DIR)
        self.tokenizer.save_pretrained(MODEL_DIR)
        self.trained_model_dir = MODEL_DIR
        self.active_task = task
        if task == "masked":
            self.fill_mask = pipeline("fill-mask", model=self.model, tokenizer=self.tokenizer)
            self.generator = None
        else:
            self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
            self.fill_mask = None

# ---------- Fine‑tuning ----------
    def _fine_tune_with_roles(self):
        if not ML_AVAILABLE:
            self._show_message("ML libraries not installed.", error=True)
            return
        if not PEFT_AVAILABLE:
            self._show_message("Fine-tuning requires the peft package.", error=True)
            return
        self._show_message("Fine‑tuning mode – paste your examples next.")
        text = self._multiline_popup("Paste equals‑format examples (end with empty line)")
        if not text:
            self._show_message("No input received.", error=True)
            return
        parsed_examples = parse_equals_input(text)
        if not parsed_examples:
            self._show_message("No valid equals-format examples found.", error=True)
            return
        self._show_message(f"Got {len(parsed_examples)} examples.")
        base_choice = self._input_popup("Base model: 1-Auto, 2-LLM, 3-SLM (default 1): ", "1")
        if base_choice == "1": base_model = self._auto_select_fine_tune_model(parsed_examples)
        elif base_choice == "2": base_model = "distilgpt2"
        else: base_model = "sshleifer/tiny-gpt2"
        use_gguf = self._input_popup("Use a local .gguf file as base? (y/n): ", "n").lower() == "y"
        gguf_path = None; arch_id = base_model
        if use_gguf:
            gguf_path = self._input_popup("Path to .gguf file: ")
            if not os.path.isfile(gguf_path):
                self._show_message("File not found. Using default base.")
                gguf_path = None
            else:
                arch_id = self._input_popup("HuggingFace architecture ID (e.g. meta-llama/Llama-2-7b-hf): ")
                if not arch_id:
                    self._show_message("Architecture ID required. Using default.")
                    gguf_path = None
        quantize = self._input_popup("Apply 4‑bit quantization after training? (y/n): ", "n").lower() == "y"
        export_gguf = self._input_popup("Export as GGUF after training? (y/n): ", "n").lower() == "y"
        self._show_message("Fine‑tuning started... (check console for logs)")
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
            self._show_message(f"Fine‑tuning complete! Model at {result}")
        except Exception as e:
            self._show_message(f"Fine‑tuning failed: {e}", error=True)

    def _auto_select_fine_tune_model(self, training_rows):
        if AUTOLEARN_AVAILABLE:
            try:
                learner = AutoLearn()
                pick = learner.choose_model(training_rows)
                if pick == "LLM": return "distilgpt2"
                elif pick in ("SLM", "MLM"): return "sshleifer/tiny-gpt2"
            except Exception:
                pass
        return "distilgpt2"

# ---------- GGUF Export ----------
    def _export_gguf(self):
        src = None
        if self.trained_model_dir and self.trained_model_dir.exists():
            src = self.trained_model_dir
        elif self.fine_tuned_path and os.path.isdir(self.fine_tuned_path):
            src = Path(self.fine_tuned_path)
        if not src:
            self._show_message("No trained model to export.", error=True)
            return
        choice = self._input_popup("Quantization: 1-4q, 2-8q, 3-16q, 4-36q (default 3): ", "3")
        if choice not in QUANT_OPTIONS:
            choice = "3"
        _, quant_type = QUANT_OPTIONS[choice]
        out_path = self._input_popup("Output file path (e.g. my_model.gguf): ")
        if not out_path:
            return
        try:
            self._convert_to_gguf(src, Path(out_path), quant_type)
            self._show_message(f"GGUF saved to {out_path}")
        except Exception as e:
            self._show_message(f"Export failed: {e}", error=True)

    def _convert_to_gguf(self, model_dir, output_path, quant_type):
        GGUF_DIR.mkdir(exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        f16 = GGUF_DIR / "tmp_f16.gguf"
        converter = _find_llama_cpp_file("convert_hf_to_gguf.py")
        if not converter:
            raise RuntimeError("llama.cpp converter not found. Set LLAMA_CPP_DIR.")
        subprocess.run([sys.executable, str(converter), str(model_dir), "--outfile", str(f16)], check=True)
        if quant_type == "f16":
            shutil.copyfile(f16, output_path)
            return
        quantizer = _find_llama_cpp_file("llama-quantize") or _find_llama_cpp_file("quantize")
        if not quantizer:
            raise RuntimeError("llama-quantize not found.")
        subprocess.run([str(quantizer), str(f16), str(output_path), quant_type], check=True)

    def _find_llama_cpp_file(self, name):
        return _find_llama_cpp_file(name)

# ---------- Prediction ----------
    def _test_prediction(self):
        if self.model is None or self.tokenizer is None:
            self._show_message("No standard model trained yet.", error=True)
            return
        self._show_message("Testing model – you can now enter prompts.")
        while True:
            prompt = self._input_popup("Enter prompt (or 'exit'): ")
            if not prompt or prompt.lower() == "exit":
                break
            try:
                if self.fill_mask:
                    out = self.fill_mask(f"{prompt} {self.tokenizer.mask_token}", top_k=1)[0]["sequence"]
                    self._show_message(f"Result: {out}")
                else:
                    gen = self.generator(f"Input: {prompt}\nOutput:", max_new_tokens=60, do_sample=True,
                                         temperature=0.7, pad_token_id=self.tokenizer.pad_token_id)[0]["generated_text"]
                    output = gen.split("Output:", 1)[-1].strip()
                    self._show_message(f"Output: {output}")
            except Exception as e:
                self._show_message(f"Prediction error: {e}", error=True)
        self._show_message("Prediction session ended.")

# ---------- Status display helpers ----------
    def _show_message(self, msg, wait=False, error=False):
        self.status_message = msg
        self._draw_screen(["placeholder"], 0)  # quick refresh
        if wait:
            self.stdscr.getch()
        elif error:
            curses.beep()


# Application entry point
if __name__ == "__main__":
    curses.wrapper(AICreatorTUI)
    print("Goodbye!")
