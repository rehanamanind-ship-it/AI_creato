"""
main_gui.py – AI Creator GUI with standard model creation and role‑based fine‑tuning.
Imports fine_tuner for the fine‑tuning pipeline and autolearn for smart model selection.
SOFTWARE DESIGNED BY ONLY AND ONLY REHAN AMAN ------------------------------------------
"""

import os
import shutil
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext

# Core ML imports – gracefully handle missing packages
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
except ImportError:
    torch = None
    Dataset = None
    AutoModelForCausalLM = None
    AutoModelForMaskedLM = None
    AutoTokenizer = None
    DataCollatorForLanguageModeling = None
    Trainer = None
    TrainingArguments = None
    pipeline = None

# Optional modules – import only if available
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
    "Auto Learn": {"task": "causal", "model": "sshleifer/tiny-gpt2"},
    "LLM":       {"task": "causal", "model": "distilgpt2"},
    "MLM":       {"task": "masked", "model": "distilroberta-base"},
    "SLM":       {"task": "causal", "model": "sshleifer/tiny-gpt2"},
}

QUANT_OPTIONS = {"4q": "q4_0", "8q": "q8_0", "16q": "f16", "36q": "q6_k"}


# Main application class
class AICreatorApp:
    """Tkinter application that trains new models, fine‑tunes existing ones, and exports GGUF."""
# Initialise all state and build the UI.
    def __init__(self, root):
        self.root = root
        root.title("AI Creator")
        root.geometry("900x1000")
        root.configure(bg="#000000")

        # Standard training state
        self.data = []
        self.file_path = None
        self.model = None
        self.tokenizer = None
        self.generator = None
        self.fill_mask = None
        self.trained_model_dir = None

        # Fine‑tuning state
        self.fine_tuned_path = None          # path to the merged model or GGUF file
        self.fine_tune_gguf_path = None      # uploaded GGUF file for fine‑tuning
        self.fine_tune_gguf_label = None     # label widget showing the chosen file

        # Tkinter variables
        self.model_type = tk.StringVar(value="Auto Learn")
        self.fine_tune_model_type = tk.StringVar(value="Auto Learn")
        self.quantization = tk.StringVar(value="")
        self.status_text = tk.StringVar(value="Select a training data file to begin.")
        self.fine_tune_quantize = tk.BooleanVar(value=False)
        self.fine_tune_export_gguf = tk.BooleanVar(value=False)
        self.use_gguf_base = tk.BooleanVar(value=False)
        self.gguf_arch_id = tk.StringVar(value="")   # HuggingFace architecture ID for GGUF

        self._build_ui()
# Create all sections of the UI.
    def _build_ui(self):
        self._add_title("AI Creator")
        self._build_standard_training_section()
        self._build_fine_tuning_section()
        self._build_test_section()
        self._build_gguf_export_section()
# Top title label.
    def _add_title(self, text):
        tk.Label(self.root, text=text, font=("Arial", 20, "bold"),
                 bg="#000000", fg="#ffffff").pack(pady=(16, 8))

# ---------- Standard Training Section ----------
# Builds the file selection, model type, and train button.
    def _build_standard_training_section(self):
        self._make_button("Select Data File", self._select_file).pack(pady=8)
        self.file_label = self._make_label("No file selected", size=10)
        self.file_label.pack(pady=(0, 12))

        self._make_section_label("Model Type").pack(pady=(6, 4))
        frame = self._make_frame()
        frame.pack(pady=(0, 8))
        for name in PRESETS:
            self._make_radio(frame, name, self.model_type, name).pack(side=tk.LEFT, padx=6)

        self.status_label = self._make_label(textvariable=self.status_text, size=10)
        self.status_label.pack(pady=(0, 10))

        self.train_btn = self._make_button("Train AI", self._train_standard, state=tk.DISABLED)
        self.train_btn.pack(pady=8)
# Opens a file dialog and parses input = output lines.
    def _select_file(self):
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            data = []
            for line in lines:
                if "=" not in line:
                    continue
                inp, out = line.split("=", 1)
                inp, out = inp.strip(), out.strip()
                if inp and out:
                    data.append({"input": inp, "output": out})
            if not data:
                messagebox.showerror("Error", "No valid data. Format: input = output")
                return
            self.file_path = path
            self.data = data
            self.file_label.config(text=f"Loaded {len(data)} examples")
            self.status_text.set("Ready to train.")
            self.train_btn.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")
# Starts standard training in a background thread.
    def _train_standard(self):
        self._set_buttons_disabled()
        self.status_text.set("Training started...")
        threading.Thread(target=self._standard_training_worker, daemon=True).start()
# Worker that loads the model, tokenises, and trains.
    def _standard_training_worker(self):
        try:
            preset = self._resolve_preset(self.model_type.get(), self.data)
            task, model_id = preset["task"], preset["model"]
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.mask_token

            records = self._build_records(task)
            dataset = Dataset.from_list(records)
            tokenized = dataset.map(
                lambda b: self.tokenizer(b["text"], truncation=True, padding="max_length", max_length=128),
                batched=True, remove_columns=["text"]
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
                output_dir=str(MODEL_DIR), overwrite_output_dir=True,
                num_train_epochs=3, per_device_train_batch_size=2,
                save_strategy="no", logging_steps=5, report_to=[], learning_rate=5e-5,
            )
            Trainer(model=self.model, args=args, train_dataset=tokenized, data_collator=collator).train()
            self.model.save_pretrained(MODEL_DIR)
            self.tokenizer.save_pretrained(MODEL_DIR)
            self.trained_model_dir = MODEL_DIR
            self._create_pipeline(task)
            self.root.after(0, self._on_training_done)
        except Exception as e:
            self.root.after(0, lambda: self._on_training_error(e))

# ---------- Fine‑tuning Section ----------
# Builds the paste area, model selection, GGUF upload, and fine‑tune button.
    def _build_fine_tuning_section(self):
        self._make_section_label("Fine‑tune with Roles (No Code)").pack(pady=(18, 6))
        self._make_label("Paste training examples (SYSTEM/USER/ASSISTANT):", size=11).pack()
        self.role_text = scrolledtext.ScrolledText(
            self.root, font=("Arial", 11), height=6, width=80,
            bg="#050505", fg="#ffffff", insertbackground="#2f80ff",
            relief=tk.SOLID, highlightthickness=1, highlightbackground="#2f80ff",
        )
        self.role_text.pack(pady=6)

        # Base model selection (Auto Learn / LLM / SLM)
        fine_frame = self._make_frame()
        fine_frame.pack(pady=(0, 6))
        self._make_label("Base model:", size=11).pack(side=tk.LEFT, padx=(0, 6))
        for name in ["Auto Learn", "LLM", "SLM"]:
            self._make_radio(fine_frame, name, self.fine_tune_model_type, name).pack(side=tk.LEFT, padx=4)

        # GGUF upload option
        gguf_frame = self._make_frame()
        gguf_frame.pack(pady=(4, 6))
        self._make_checkbox(gguf_frame, "Use local .gguf file as base model", self.use_gguf_base).pack(side=tk.LEFT)
        self.gguf_select_btn = self._make_button("Select .gguf", self._select_gguf_file)
        self.gguf_select_btn.pack(side=tk.LEFT, padx=10)
        self.fine_tune_gguf_label = self._make_label("No file chosen", size=9)
        self.fine_tune_gguf_label.pack(side=tk.LEFT, padx=4)

        # Architecture ID entry (required when using GGUF)
        arch_frame = self._make_frame()
        arch_frame.pack(pady=(0, 4))
        self._make_label("Architecture ID (HuggingFace):", size=10).pack(side=tk.LEFT, padx=(0, 6))
        self.arch_entry = tk.Entry(
            arch_frame, textvariable=self.gguf_arch_id, font=("Arial", 10), width=30,
            bg="#050505", fg="#ffffff", insertbackground="#2f80ff",
            relief=tk.SOLID, highlightthickness=1, highlightbackground="#2f80ff",
        )
        self.arch_entry.pack(side=tk.LEFT)

        # Options and start button
        opt_frame = self._make_frame()
        opt_frame.pack(pady=(4, 8))
        self._make_checkbox(opt_frame, "Quantize after training", self.fine_tune_quantize).pack(side=tk.LEFT, padx=8)
        self._make_checkbox(opt_frame, "Export as GGUF", self.fine_tune_export_gguf).pack(side=tk.LEFT, padx=8)

        self.fine_tune_btn = self._make_button("Fine‑tune with Roles", self._start_fine_tuning)
        self.fine_tune_btn.pack(pady=6)
# Opens a file dialog to select a .gguf file for fine‑tuning.
    def _select_gguf_file(self):
        path = filedialog.askopenfilename(filetypes=[("GGUF files", "*.gguf")])
        if path:
            self.fine_tune_gguf_path = path
            self.fine_tune_gguf_label.config(text=os.path.basename(path))
        else:
            self.fine_tune_gguf_path = None
            self.fine_tune_gguf_label.config(text="No file chosen")
# Validates input and starts the fine‑tuning thread.
    def _start_fine_tuning(self):
        if not FINE_TUNER_AVAILABLE:
            messagebox.showerror("Missing module", "fine_tuner.py not found. Please add it to the project folder.")
            return
        text = self.role_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showerror("Error", "Paste training examples first.")
            return
        if self.use_gguf_base.get():
            if not self.fine_tune_gguf_path:
                messagebox.showerror("Error", "Select a .gguf file first.")
                return
            arch_id = self.gguf_arch_id.get().strip()
            if not arch_id:
                messagebox.showerror("Error", "Enter the HuggingFace architecture ID (e.g. meta-llama/Llama-2-7b-hf).")
                return
        self._set_buttons_disabled()
        self.status_text.set("Role‑based fine‑tuning started...")
        threading.Thread(target=self._fine_tuning_worker, args=(text,), daemon=True).start()
# Background worker that calls the fine_tuner pipeline.
    def _fine_tuning_worker(self, examples_str):
        try:
            # Determine base model ID
            preset_name = self.fine_tune_model_type.get()
            if preset_name == "Auto Learn" and AUTOLEARN_AVAILABLE:
                # Use AutoLearn to pick a base model for fine‑tuning (simple heuristic)
                base_model_id = self._auto_select_fine_tune_model()
            else:
                base_model_id = PRESETS[preset_name]["model"] if preset_name in PRESETS else "distilgpt2"

            gguf_path = self.fine_tune_gguf_path if self.use_gguf_base.get() else None
            if gguf_path:
                # Override base_model_id with the user‑provided architecture ID
                base_model_id = self.gguf_arch_id.get().strip()

            result = fine_tuner.run_finetuning_pipeline(
                training_text=examples_str,
                base_model_id=base_model_id,
                output_dir=str(FINE_TUNED_DIR),
                quantize=self.fine_tune_quantize.get(),
                export_gguf=self.fine_tune_export_gguf.get(),
                uploaded_gguf_path=gguf_path,
            )
            self.fine_tuned_path = result
            self.root.after(0, self._on_fine_tuning_done)
        except Exception as e:
            self.root.after(0, lambda: self._on_training_error(e))
# Uses AutoLearn (or a simple fallback) to choose a fine‑tuning base model.
    def _auto_select_fine_tune_model(self):
        if AUTOLEARN_AVAILABLE:
            try:
                learner = AutoLearn()
                pick = learner.choose_model([])   # no specific data, just size heuristic
                if pick in PRESETS and pick != "Auto Learn":
                    return PRESETS[pick]["model"]
            except Exception:
                pass
        return "distilgpt2"   # default LLM
# Fine‑tuning success callback.
    def _on_fine_tuning_done(self):
        self.status_text.set("Fine‑tuning complete! You can now test or export the model.")
        self._set_buttons_normal()
        messagebox.showinfo("Success", "Role‑based fine‑tuning finished successfully.")

# ---------- Shared helpers for training ----------
# Resolves the model preset (standard training), using AutoLearn if selected.
    def _resolve_preset(self, choice, data):
        if choice != "Auto Learn":
            return PRESETS[choice]
        if AUTOLEARN_AVAILABLE:
            try:
                learner = AutoLearn()
                pick = learner.choose_model(data)
                if pick in PRESETS and pick != "Auto Learn":
                    return PRESETS[pick]
            except Exception:
                pass
        # Fallback: small dataset -> SLM, else LLM
        return PRESETS["SLM"] if len(data) < 100 else PRESETS["LLM"]
# Converts self.data into training records.
    def _build_records(self, task):
        records = []
        for row in self.data:
            if task == "masked":
                text = f"{row['input']} {self.tokenizer.mask_token} {row['output']}"
            else:
                text = f"Input: {row['input']}\nOutput: {row['output']}"
            records.append({"text": text})
        return records
# Creates the appropriate pipeline for predictions.
    def _create_pipeline(self, task):
        if task == "masked":
            self.fill_mask = pipeline("fill-mask", model=self.model, tokenizer=self.tokenizer)
            self.generator = None
        else:
            self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
            self.fill_mask = None
# Standard training success callback.
    def _on_training_done(self):
        self.status_text.set(f"Training complete. Model saved to {MODEL_DIR.name}.")
        self._set_buttons_normal()
        messagebox.showinfo("Success", "AI trained successfully.")
# Generic training/fine‑tuning error callback.
    def _on_training_error(self, error):
        self.status_text.set("Training failed.")
        self._set_buttons_normal()
        messagebox.showerror("Error", str(error))

# ---------- Test Section ----------
# Input field, predict button, and output display.
    def _build_test_section(self):
        self._make_section_label("Test Your AI").pack(pady=(18, 6))
        self._make_label("Input:", size=11).pack()
        self.input_text = tk.Entry(
            self.root, font=("Arial", 11), width=70,
            bg="#050505", fg="#ffffff", insertbackground="#2f80ff",
            relief=tk.SOLID, highlightthickness=1, highlightbackground="#2f80ff",
        )
        self.input_text.pack(pady=6)
        self.predict_btn = self._make_button("Predict", self._predict, state=tk.DISABLED)
        self.predict_btn.pack(pady=8)

        self._make_label("Output:", size=11).pack()
        self.output_text = scrolledtext.ScrolledText(
            self.root, font=("Arial", 11), height=8, width=72,
            bg="#050505", fg="#ffffff", insertbackground="#2f80ff",
            relief=tk.SOLID, highlightthickness=1, highlightbackground="#2f80ff",
        )
        self.output_text.pack(pady=6)
        self.output_text.config(state=tk.DISABLED)
# Runs prediction on the currently active model (standard or fine‑tuned).
    def _predict(self):
        # For simplicity, use the standard model if available; otherwise try fine‑tuned
        if self.model is None or self.tokenizer is None:
            messagebox.showerror("Error", "No model available. Train or fine‑tune first.")
            return
        prompt = self.input_text.get().strip()
        if not prompt:
            messagebox.showerror("Error", "Enter some input.")
            return
        try:
            if self.fill_mask:
                result = self.fill_mask(f"{prompt} {self.tokenizer.mask_token}", top_k=1)[0]["sequence"]
            else:
                gen = self.generator(
                    f"Input: {prompt}\nOutput:",
                    max_new_tokens=60, do_sample=True, temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id,
                )[0]["generated_text"]
                result = gen.split("Output:", 1)[-1].strip()
            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, f"{prompt} = {result}")
            self.output_text.config(state=tk.DISABLED)
        except Exception as e:
            messagebox.showerror("Error", str(e))

# ---------- GGUF Export Section ----------
# Quantisation selection and download button.
    def _build_gguf_export_section(self):
        self._make_section_label("GGUF Download").pack(pady=(18, 6))
        qframe = self._make_frame()
        qframe.pack(pady=(0, 8))
        for lbl in QUANT_OPTIONS:
            self._make_radio(qframe, lbl, self.quantization, lbl).pack(side=tk.LEFT, padx=6)
        self.export_btn = self._make_button("Download .gguf", self._export_gguf, state=tk.DISABLED)
        self.export_btn.pack(pady=8)
# Exports the currently available trained model (standard or fine‑tuned) to GGUF.
    def _export_gguf(self):
        qtype = self.quantization.get()
        if not qtype:
            messagebox.showerror("Error", "Select a quantization version.")
            return

        # Determine which model to export – prefer the most recently trained
        src_dir = None
        if self.trained_model_dir and self.trained_model_dir.exists():
            src_dir = self.trained_model_dir
        elif self.fine_tuned_path and os.path.isdir(self.fine_tuned_path):
            src_dir = Path(self.fine_tuned_path)
        else:
            messagebox.showerror("Error", "No trained model to export. Train or fine‑tune first.")
            return

        out = filedialog.asksaveasfilename(
            defaultextension=".gguf",
            initialfile="my_ai.gguf",
            filetypes=[("GGUF model", "*.gguf")],
        )
        if not out:
            return
        try:
            self._convert_to_gguf(src_dir, Path(out), QUANT_OPTIONS[qtype])
            messagebox.showinfo("Success", f"GGUF exported:\n{out}")
        except Exception as e:
            messagebox.showerror("GGUF export failed", str(e))
# Converts a HuggingFace model directory to a quantised GGUF file.
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
            raise RuntimeError("llama-quantize not found.")
        subprocess.run([str(quantizer), str(f16), str(output_path), quant_type], check=True)
# Searches for a given file inside the llama.cpp folder.
    def _find_llama_cpp_file(self, name):
        search_dirs = [APP_DIR / "llama.cpp", Path(os.environ.get("LLAMA_CPP_DIR", ""))]
        for root in search_dirs:
            if not root or not root.exists():
                continue
            hits = list(root.rglob(name))
            if hits:
                return hits[0]
        return None

# ---------- UI helper factories ----------
    def _set_buttons_disabled(self):
        self.train_btn.config(state=tk.DISABLED)
        self.fine_tune_btn.config(state=tk.DISABLED)
        self.export_btn.config(state=tk.DISABLED)
        self.predict_btn.config(state=tk.DISABLED)
    def _set_buttons_normal(self):
        self.train_btn.config(state=tk.NORMAL)
        self.fine_tune_btn.config(state=tk.NORMAL)
        self.export_btn.config(state=tk.NORMAL)
        self.predict_btn.config(state=tk.NORMAL)
    def _make_frame(self):
        return tk.Frame(self.root, bg="#000000")
    def _make_label(self, text=None, size=12, bold=False, textvariable=None):
        font = ("Arial", size, "bold" if bold else "normal")
        return tk.Label(self.root, text=text, textvariable=textvariable, font=font,
                        bg="#000000", fg="#ffffff")
    def _make_section_label(self, text):
        return self._make_label(text, size=13, bold=True)
    def _make_button(self, text, command, state=tk.NORMAL):
        return tk.Button(
            self.root, text=text, command=command, font=("Arial", 12),
            bg="#000000", fg="#ffffff", activebackground="#071a33",
            activeforeground="#ffffff", disabledforeground="#777777",
            relief=tk.SOLID, bd=1, highlightthickness=2,
            highlightbackground="#2f80ff", highlightcolor="#2f80ff",
            state=state, padx=14, pady=5,
        )
    def _make_radio(self, parent, text, variable, value):
        return tk.Radiobutton(
            parent, text=text, variable=variable, value=value,
            bg="#000000", fg="#ffffff", selectcolor="#050505",
            activebackground="#000000", activeforeground="#2f80ff",
            highlightthickness=1, highlightbackground="#2f80ff",
            font=("Arial", 11),
        )
    def _make_checkbox(self, parent, text, variable):
        return tk.Checkbutton(
            parent, text=text, variable=variable,
            bg="#000000", fg="#ffffff", selectcolor="#050505",
            activebackground="#000000", activeforeground="#2f80ff",
            highlightthickness=1, highlightbackground="#2f80ff",
            font=("Arial", 11),
        )

# Application entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = AICreatorApp(root)
    root.mainloop()
