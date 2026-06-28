import os
import fine_tuner
import shutil
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext

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

try:
    from autolearn import AutoLearn
except ImportError:
    AutoLearn = None


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "trained_model"
GGUF_DIR = APP_DIR / "gguf_models"
FINE_TUNED_DIR = APP_DIR / "fine_tuned_model"

MODEL_PRESETS = {
    "Auto Learn": {
        "task": "causal",
        "model": "sshleifer/tiny-gpt2",
        "description": "Automatically chooses a small local-friendly language model.",
    },
    "LLM": {
        "task": "causal",
        "model": "distilgpt2",
        "description": "Causal language model for prompt-to-answer generation.",
    },
    "MLM": {
        "task": "masked",
        "model": "distilroberta-base",
        "description": "Masked language model for fill-mask style learning.",
    },
    "SLM": {
        "task": "causal",
        "model": "sshleifer/tiny-gpt2",
        "description": "Small language model for faster training on normal PCs.",
    },
}

QUANTIZATION_PRESETS = {
    "4q": "q4_0",
    "8q": "q8_0",
    "16q": "f16",
    "36q": "q6_k",
}


class AICreatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Creator")
        self.root.geometry("900x900")   # slightly taller to fit the new section
        self.root.configure(bg="#000000")

        # Standard training data
        self.data = []
        self.file_path = None
        self.model = None
        self.tokenizer = None
        self.generator = None
        self.fill_mask = None
        self.trained_model_dir = None

        # Fine‑tuning specific
        self.fine_tuned_model_path = None   # path to the merged HF model or GGUF
        self.fine_tuned_gguf_path = None    # if GGUF was exported

        self.model_type = tk.StringVar(value="Auto Learn")
        self.fine_tune_model_type = tk.StringVar(value="LLM")  # for role‑based fine‑tuning
        self.quantization = tk.StringVar(value="")
        self.status = tk.StringVar(value="Select a training data file to begin.")
        self.fine_tune_quantize = tk.BooleanVar(value=False)
        self.fine_tune_export_gguf = tk.BooleanVar(value=False)

        self.build_ui()

    # ------------------------------------------------------------------
    # UI builders
    # ------------------------------------------------------------------
    def build_ui(self):
        # Title
        self.title_label = self.label("AI Creator", size=20, bold=True)
        self.title_label.pack(pady=(16, 8))

        # ---- Standard Training Section ----
        self.select_file_btn = self.button("Select Data File", self.select_file)
        self.select_file_btn.pack(pady=8)

        self.file_label = self.label("No file selected", size=10)
        self.file_label.pack(pady=(0, 12))

        self.section("Model Type").pack(pady=(6, 4))
        model_frame = self.frame()
        model_frame.pack(pady=(0, 8))
        for name in MODEL_PRESETS:
            self.radio(model_frame, name, self.model_type, name).pack(side=tk.LEFT, padx=6)

        self.status_label = self.label(textvariable=self.status, size=10)
        self.status_label.pack(pady=(0, 10))

        self.train_btn = self.button("Train AI", self.train_ai, state=tk.DISABLED)
        self.train_btn.pack(pady=8)

        # ---- Role‑based Fine‑tuning Section ----
        self.section("Fine‑tune with Roles (No Code)").pack(pady=(18, 6))
        self.label("Paste training examples (SYSTEM/USER/ASSISTANT):", size=11).pack()

        self.role_text = scrolledtext.ScrolledText(
            self.root,
            font=("Arial", 11),
            height=6,
            width=80,
            bg="#050505",
            fg="#ffffff",
            insertbackground="#2f80ff",
            relief=tk.SOLID,
            highlightthickness=1,
            highlightbackground="#2f80ff",
            highlightcolor="#2f80ff",
        )
        self.role_text.pack(pady=6)

        fine_model_frame = self.frame()
        fine_model_frame.pack(pady=(0, 6))
        self.label("Base model:", size=11).pack(side=tk.LEFT, padx=(0, 6))
        for name in ["LLM", "SLM"]:   # reasonable choices for role‑based
            self.radio(fine_model_frame, name, self.fine_tune_model_type, name).pack(
                side=tk.LEFT, padx=4
            )

        options_frame = self.frame()
        options_frame.pack(pady=(0, 8))
        self.checkbox(options_frame, "Quantize after training", self.fine_tune_quantize).pack(
            side=tk.LEFT, padx=8
        )
        self.checkbox(options_frame, "Export as GGUF", self.fine_tune_export_gguf).pack(
            side=tk.LEFT, padx=8
        )

        self.fine_tune_btn = self.button("Fine‑tune with Roles", self.fine_tune_with_roles)
        self.fine_tune_btn.pack(pady=6)

        # ---- Test Your AI Section ----
        self.section("Test Your AI").pack(pady=(18, 6))
        self.label("Input:", size=11).pack()
        self.input_text = tk.Entry(
            self.root,
            font=("Arial", 11),
            width=70,
            bg="#050505",
            fg="#ffffff",
            insertbackground="#2f80ff",
            relief=tk.SOLID,
            highlightthickness=1,
            highlightbackground="#2f80ff",
            highlightcolor="#2f80ff",
        )
        self.input_text.pack(pady=6)

        self.predict_btn = self.button("Predict", self.predict, state=tk.DISABLED)
        self.predict_btn.pack(pady=8)

        self.label("Output:", size=11).pack()
        self.output_text = scrolledtext.ScrolledText(
            self.root,
            font=("Arial", 11),
            height=8,
            width=72,
            bg="#050505",
            fg="#ffffff",
            insertbackground="#2f80ff",
            relief=tk.SOLID,
            highlightthickness=1,
            highlightbackground="#2f80ff",
            highlightcolor="#2f80ff",
        )
        self.output_text.pack(pady=6)
        self.output_text.config(state=tk.DISABLED)

        # ---- GGUF Download Section ----
        self.section("GGUF Download").pack(pady=(18, 6))
        quant_frame = self.frame()
        quant_frame.pack(pady=(0, 8))
        for label in QUANTIZATION_PRESETS:
            self.radio(quant_frame, label, self.quantization, label).pack(side=tk.LEFT, padx=6)

        self.export_btn = self.button("Download .gguf", self.export_gguf, state=tk.DISABLED)
        self.export_btn.pack(pady=8)

    # ---- Helper UI methods ----
    def frame(self):
        return tk.Frame(self.root, bg="#000000")

    def label(self, text=None, size=12, bold=False, textvariable=None):
        font = ("Arial", size, "bold" if bold else "normal")
        return tk.Label(
            self.root,
            text=text,
            textvariable=textvariable,
            font=font,
            bg="#000000",
            fg="#ffffff",
        )

    def section(self, text):
        return self.label(text, size=13, bold=True)

    def button(self, text, command, state=tk.NORMAL):
        return tk.Button(
            self.root,
            text=text,
            command=command,
            font=("Arial", 12),
            bg="#000000",
            fg="#ffffff",
            activebackground="#071a33",
            activeforeground="#ffffff",
            disabledforeground="#777777",
            relief=tk.SOLID,
            bd=1,
            highlightthickness=2,
            highlightbackground="#2f80ff",
            highlightcolor="#2f80ff",
            state=state,
            padx=14,
            pady=5,
        )

    def radio(self, parent, text, variable, value):
        return tk.Radiobutton(
            parent,
            text=text,
            variable=variable,
            value=value,
            bg="#000000",
            fg="#ffffff",
            selectcolor="#050505",
            activebackground="#000000",
            activeforeground="#2f80ff",
            highlightthickness=1,
            highlightbackground="#2f80ff",
            font=("Arial", 11),
        )

    def checkbox(self, parent, text, variable):
        return tk.Checkbutton(
            parent,
            text=text,
            variable=variable,
            bg="#000000",
            fg="#ffffff",
            selectcolor="#050505",
            activebackground="#000000",
            activeforeground="#2f80ff",
            highlightthickness=1,
            highlightbackground="#2f80ff",
            font=("Arial", 11),
        )

    # ------------------------------------------------------------------
    # Standard training logic (unchanged)
    # ------------------------------------------------------------------
    def select_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                lines = file.readlines()

            data = []
            for line in lines:
                if "=" not in line:
                    continue
                input_value, output_value = line.split("=", 1)
                input_value = input_value.strip()
                output_value = output_value.strip()
                if input_value and output_value:
                    data.append({"input": input_value, "output": output_value})

            if not data:
                messagebox.showerror(
                    "Error", "No valid data found. Format should be: input = output"
                )
                return

            self.file_path = file_path
            self.data = data
            self.file_label.config(text=f"Loaded {len(self.data)} training examples")
            self.status.set("Ready to train.")
            self.train_btn.config(state=tk.NORMAL)
        except Exception as error:
            messagebox.showerror("Error", f"Failed to load file: {error}")

    def train_ai(self):
        if not self.dependencies_available():
            return

        self.train_btn.config(state=tk.DISABLED)
        self.predict_btn.config(state=tk.DISABLED)
        self.export_btn.config(state=tk.DISABLED)
        self.status.set("Training started. This can take time on CPU.")

        thread = threading.Thread(target=self.train_ai_worker, daemon=True)
        thread.start()

    def train_ai_worker(self):
        try:
            preset = self.resolve_model_preset()
            task = preset["task"]
            model_name = preset["model"]

            self.set_status(f"Loading {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.mask_token

            dataset = Dataset.from_list(self.build_training_records(task))
            tokenized_dataset = dataset.map(
                lambda batch: self.tokenizer(
                    batch["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=128,
                ),
                batched=True,
                remove_columns=["text"],
            )

            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=(task == "masked"),
                mlm_probability=0.15,
            )

            if task == "masked":
                self.model = AutoModelForMaskedLM.from_pretrained(model_name)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                self.model.config.pad_token_id = self.tokenizer.pad_token_id

            training_args = TrainingArguments(
                output_dir=str(MODEL_DIR),
                overwrite_output_dir=True,
                num_train_epochs=3,
                per_device_train_batch_size=2,
                save_strategy="no",
                logging_steps=5,
                report_to=[],
                learning_rate=5e-5,
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
            )

            self.set_status("Fine-tuning model...")
            trainer.train()

            MODEL_DIR.mkdir(exist_ok=True)
            self.model.save_pretrained(MODEL_DIR)
            self.tokenizer.save_pretrained(MODEL_DIR)
            self.trained_model_dir = MODEL_DIR

            if task == "masked":
                self.fill_mask = pipeline(
                    "fill-mask", model=self.model, tokenizer=self.tokenizer
                )
                self.generator = None
            else:
                self.generator = pipeline(
                    "text-generation", model=self.model, tokenizer=self.tokenizer
                )
                self.fill_mask = None

            self.root.after(0, self.training_finished)
        except Exception as error:
            self.root.after(0, lambda: self.training_failed(error))

    def dependencies_available(self):
        if AutoTokenizer is None:
            messagebox.showerror(
                "Missing packages",
                "Install the ML packages first:\n\npip install -r requirements.txt",
            )
            return False
        return True

    def resolve_model_preset(self):
        selected = self.model_type.get()
        if selected != "Auto Learn":
            return MODEL_PRESETS[selected]

        if AutoLearn is not None:
            try:
                learner = AutoLearn()
                choice = getattr(learner, "choose_model", lambda *_: "SLM")(self.data)
                if choice in MODEL_PRESETS and choice != "Auto Learn":
                    return MODEL_PRESETS[choice]
            except Exception:
                pass

        if len(self.data) < 100:
            return MODEL_PRESETS["SLM"]
        return MODEL_PRESETS["LLM"]

    def build_training_records(self, task):
        records = []
        for row in self.data:
            if task == "masked":
                text = f"{row['input']} {self.tokenizer.mask_token} {row['output']}"
            else:
                text = f"Input: {row['input']}\nOutput: {row['output']}"
            records.append({"text": text})
        return records

    def training_finished(self):
        self.status.set(f"Training complete. Model saved to {MODEL_DIR.name}.")
        self.train_btn.config(state=tk.NORMAL)
        self.predict_btn.config(state=tk.NORMAL)
        self.export_btn.config(state=tk.NORMAL)
        messagebox.showinfo("Success", "AI trained successfully with Transformers.")

    def training_failed(self, error):
        self.status.set("Training failed.")
        self.train_btn.config(state=tk.NORMAL)
        messagebox.showerror("Error", str(error))

    # ------------------------------------------------------------------
    # New: Role‑based Fine‑tuning
    # ------------------------------------------------------------------
    def fine_tune_with_roles(self):
        examples = self.role_text.get("1.0", tk.END).strip()
        if not examples:
            messagebox.showerror("Error", "Paste training examples in the equals format.")
            return

        # Disable buttons during training
        self.fine_tune_btn.config(state=tk.DISABLED)
        self.train_btn.config(state=tk.DISABLED)
        self.export_btn.config(state=tk.DISABLED)
        self.status.set("Role‑based fine‑tuning started...")

        thread = threading.Thread(target=self._fine_tune_thread, args=(examples,), daemon=True)
        thread.start()

    def _fine_tune_thread(self, examples_str):
        try:
            # Get the base model ID from the fine‑tuning model preset
            preset_name = self.fine_tune_model_type.get()
            if preset_name not in MODEL_PRESETS or preset_name == "Auto Learn":
                raise ValueError("Invalid base model for role‑based fine‑tuning.")
            base_model_id = MODEL_PRESETS[preset_name]["model"]

            quantize = self.fine_tune_quantize.get()
            export_gguf = self.fine_tune_export_gguf.get()

            output_dir = str(FINE_TUNED_DIR)

            self.set_status("Fine‑tuning with QLoRA... (this can take several minutes)")

            # Call the imported pipeline
            result_path = fine_tuner.run_finetuning_pipeline(
                manual_examples_str=examples_str,
                base_model_id=base_model_id,
                output_dir=output_dir,
                quantize=quantize,
                export_gguf=export_gguf,
                auto_scan_gguf=False,   # we're using a HuggingFace model directly
            )

            # Store the result path
            if export_gguf:
                self.fine_tuned_gguf_path = result_path
                self.fine_tuned_model_path = None
            else:
                self.fine_tuned_model_path = result_path
                self.fine_tuned_gguf_path = None

            self.root.after(0, self._fine_tune_finished)
        except Exception as error:
            self.root.after(0, lambda: self._fine_tune_failed(error))

    def _fine_tune_finished(self):
        self.status.set("Role‑based fine‑tuning complete!")
        self.fine_tune_btn.config(state=tk.NORMAL)
        self.train_btn.config(state=tk.NORMAL)   # re‑enable standard training
        self.export_btn.config(state=tk.NORMAL)   # allow GGUF export from the fine‑tuned model
        messagebox.showinfo("Success", "Role‑based model has been fine‑tuned and saved.")

    def _fine_tune_failed(self, error):
        self.status.set("Fine‑tuning failed.")
        self.fine_tune_btn.config(state=tk.NORMAL)
        self.train_btn.config(state=tk.NORMAL)
        self.export_btn.config(state=tk.NORMAL)
        messagebox.showerror("Fine‑tuning Error", str(error))

    # ------------------------------------------------------------------
    # Prediction (unchanged)
    # ------------------------------------------------------------------
    def predict(self):
        if self.model is None or self.tokenizer is None:
            messagebox.showerror("Error", "Please train the AI first")
            return

        user_input = self.input_text.get().strip()
        if not user_input:
            messagebox.showerror("Error", "Please enter some input")
            return

        try:
            if self.fill_mask is not None:
                prompt = f"{user_input} {self.tokenizer.mask_token}"
                result = self.fill_mask(prompt, top_k=1)[0]["sequence"]
                output = result
            else:
                prompt = f"Input: {user_input}\nOutput:"
                result = self.generator(
                    prompt,
                    max_new_tokens=60,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id,
                )[0]["generated_text"]
                output = result.split("Output:", 1)[-1].strip()

            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, f"{user_input} = {output}")
            self.output_text.config(state=tk.DISABLED)
        except Exception as error:
            messagebox.showerror("Error", str(error))

    # ------------------------------------------------------------------
    # GGUF Export (updated to handle both standard and fine‑tuned models)
    # ------------------------------------------------------------------
    def export_gguf(self):
        selected_quantization = self.quantization.get()
        if not selected_quantization:
            messagebox.showerror(
                "Quantization required",
                "Select a quantization version first: 4q, 8q, 16q, or 36q.",
            )
            return

        # Determine which model to export:
        model_dir = None
        # If we have a fine‑tuned model (HuggingFace), prefer it over the standard model
        if self.fine_tuned_model_path and os.path.isdir(self.fine_tuned_model_path):
            model_dir = Path(self.fine_tuned_model_path)
        elif self.trained_model_dir and self.trained_model_dir.exists():
            model_dir = self.trained_model_dir
        else:
            messagebox.showerror("Error", "No trained model available. Train or fine‑tune first.")
            return

        quant_type = QUANTIZATION_PRESETS[selected_quantization]
        output_path = filedialog.asksaveasfilename(
            defaultextension=".gguf",
            initialfile=f"ai_creator_{selected_quantization}.gguf",
            filetypes=[("GGUF model", "*.gguf")],
        )
        if not output_path:
            return

        try:
            self.convert_to_gguf(model_dir, Path(output_path), quant_type)
            messagebox.showinfo("Success", f"GGUF exported:\n{output_path}")
        except Exception as error:
            messagebox.showerror("GGUF export failed", str(error))

    def convert_to_gguf(self, model_dir, output_path, quant_type):
        converter = self.find_llama_cpp_file("convert_hf_to_gguf.py")
        quantizer = self.find_llama_cpp_file("llama-quantize.exe") or self.find_llama_cpp_file(
            "quantize.exe"
        )

        if converter is None:
            raise RuntimeError(
                "GGUF export needs llama.cpp. Set LLAMA_CPP_DIR to your llama.cpp folder "
                "containing convert_hf_to_gguf.py."
            )

        GGUF_DIR.mkdir(exist_ok=True)
        f16_path = GGUF_DIR / "ai_creator_f16.gguf"

        subprocess.run(
            ["python", str(converter), str(model_dir), "--outfile", str(f16_path)],
            check=True,
        )

        if quant_type == "f16":
            shutil.copyfile(f16_path, output_path)
            return

        if quantizer is None:
            raise RuntimeError(
                "Quantized GGUF export needs llama-quantize.exe from llama.cpp. "
                "Build llama.cpp and set LLAMA_CPP_DIR to that folder."
            )

        subprocess.run(
            [str(quantizer), str(f16_path), str(output_path), quant_type],
            check=True,
        )

    def find_llama_cpp_file(self, filename):
        search_roots = [
            APP_DIR / "llama.cpp",
            Path(os.environ.get("LLAMA_CPP_DIR", "")),
        ]
        for root in search_roots:
            if not root or not root.exists():
                continue
            matches = list(root.rglob(filename))
            if matches:
                return matches[0]
        return None

    # ------------------------------------------------------------------
    # Status helper
    # ------------------------------------------------------------------
    def set_status(self, text):
        self.root.after(0, lambda: self.status.set(text))


if __name__ == "__main__":
    root = tk.Tk()
    app = AICreatorApp(root)
    root.mainloop()
