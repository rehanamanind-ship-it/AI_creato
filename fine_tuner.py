"""
finetuner.py – No‑code fine‑tuning, quantization, and GGUF in/out.
Fully self‑contained.

Usage from main app:
    from finetuner import run_finetuning_pipeline

    result = run_finetuning_pipeline(
        manual_examples_str = "=SYSTEM=...=USER=...=ASSISTANT=...===",
        base_model_id = "microsoft/phi-2",      # used if no GGUF found
        output_dir = "./my_finetuned_model",
        export_gguf = True,
        quantize = True,
        auto_scan_gguf = True  # automatically finds .gguf in current folder
    )
    print(f"Final model: {result}")
"""

import os
import re
import json
import logging
import shutil
import subprocess
from typing import Optional, List, Dict

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    TaskType,
)
from transformers.utils import logging as transformers_logging

# ----------------------------------------------------------------------
try:
    import gguf
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False

try:
    from trl import SFTTrainer
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False

try:
    from awq import AutoAWQForCausalLM
    AWQ_AVAILABLE = True
except ImportError:
    AWQ_AVAILABLE = False

try:
    import llama_cpp
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
transformers_logging.set_verbosity_info()


# ======================================================================
# 1. Parse structured "equals" input
# ======================================================================
def parse_manual_examples(text: str) -> List[Dict[str, str]]:
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
                logger.warning(f"Could not parse segment: {seg[:50]}...")
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


def create_dataset_from_manual(examples_str: str) -> Dataset:
    examples = parse_manual_examples(examples_str)
    if not examples:
        raise ValueError("No valid training examples found.")
    conversations = []
    for ex in examples:
        messages = []
        if ex["system"]:
            messages.append({"role": "system", "content": ex["system"]})
        messages.append({"role": "user", "content": ex["user"]})
        messages.append({"role": "assistant", "content": ex["assistant"]})
        conversations.append({"messages": messages})
    return Dataset.from_list(conversations)


# ======================================================================
# 2. Project‑folder .gguf scanner
# ======================================================================
def find_gguf_files(folder: Optional[str] = None, max_results: int = 1) -> List[str]:
    """Return up to *max_results* .gguf files found in *folder* (cwd default)."""
    folder = folder or os.getcwd()
    gguf_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(".gguf"):
                gguf_files.append(os.path.join(root, file))
    gguf_files.sort()
    return gguf_files[:max_results]


# ======================================================================
# 3. GGUF ↔ HuggingFace conversion
# ======================================================================
def convert_gguf_to_hf(gguf_path: str, base_model_id: str, output_dir: str) -> str:
    """Convert a GGUF file to a HuggingFace‑style directory."""
    if not GGUF_AVAILABLE:
        raise ImportError("The 'gguf' package is required. Install: pip install gguf")
    logger.info(f"Converting {gguf_path} to HF format using config from {base_model_id}...")

    os.makedirs(output_dir, exist_ok=True)
    config = AutoConfig.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    reader = gguf.GGUFReader(gguf_path)
    tensors = {}
    for tensor in reader.tensors:
        tensors[tensor.name] = torch.from_numpy(tensor.data)

    hf_weights = {}
    for gguf_name, weight in tensors.items():
        new_name = gguf_name
        m = re.match(r'blk\.(\d+)\.(.+)', gguf_name)
        if m:
            layer_idx = m.group(1)
            param = m.group(2)
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
                logger.warning(f"Unmapped GGUF parameter: {gguf_name}, skipping.")
                continue
        elif gguf_name == "token_embd.weight":
            new_name = "model.embed_tokens.weight"
        elif gguf_name == "output_norm.weight":
            new_name = "model.norm.weight"
        elif gguf_name == "output.weight":
            new_name = "lm_head.weight"
        elif gguf_name == "norm.weight":
            new_name = "model.final_layernorm.weight" if "final_layernorm" in [
                n for n, _ in hf_weights.keys()] else "model.norm.weight"
        # else keep original name

        hf_weights[new_name] = weight

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model.load_state_dict(hf_weights, strict=False, assign=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    logger.info(f"Converted GGUF model saved to {output_dir}")
    return output_dir


def convert_hf_to_gguf(
    hf_model_dir: str,
    output_gguf_path: str,
    quantization_type: Optional[str] = None,
) -> str:
    """Convert a HuggingFace model to a .gguf file."""
    if LLAMA_CPP_AVAILABLE:
        try:
            from llama_cpp.llama_cpp import llama_convert_hf_to_gguf
            params = {
                "model": hf_model_dir,
                "output": output_gguf_path,
                "outtype": quantization_type or "q8_0",
            }
            llama_convert_hf_to_gguf(**params)
            return output_gguf_path
        except ImportError:
            pass

    convert_script = shutil.which("convert-hf-to-gguf.py") or "convert-hf-to-gguf.py"
    cmd = [convert_script, hf_model_dir, "--outtype", quantization_type or "q8_0", "--outfile", output_gguf_path]
    subprocess.run(cmd, check=True)
    return output_gguf_path


# ======================================================================
# 4. Fine‑tuning with LoRA
# ======================================================================
def format_chat_template(example, tokenizer):
    messages = example["messages"]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False)
    # fallback
    text_parts = [f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages]
    return "\n".join(text_parts)


def fine_tune_with_lora(
    base_model_dir: str,
    dataset: Dataset,
    output_dir: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    bf16: bool = True,
    use_4bit: bool = True,
    logging_steps: int = 10,
    save_steps: int = 200,
) -> str:
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules="all-linear",
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    def tokenize_function(examples):
        texts = [format_chat_template(ex, tokenizer) for ex in examples]
        return tokenizer(texts, truncation=True, max_length=2048, padding="max_length")

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "lora_checkpoints"),
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        bf16=bf16,
        logging_steps=logging_steps,
        save_steps=save_steps,
        optim="paged_adamw_8bit",
        report_to="none",
    )

    if TRL_AVAILABLE:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=tokenized_dataset,
            args=training_args,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=lambda data: {
                "input_ids": torch.stack([d["input_ids"] for d in data]),
                "attention_mask": torch.stack([d["attention_mask"] for d in data]),
                "labels": torch.stack([d["input_ids"] for d in data]),
            },
        )

    trainer.train()

    model.save_pretrained(os.path.join(output_dir, "lora_adapter"))
    tokenizer.save_pretrained(os.path.join(output_dir, "lora_adapter"))

    merged_model = model.merge_and_unload()
    merged_path = os.path.join(output_dir, "final_model")
    merged_model.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)
    logger.info(f"Merged model saved to {merged_path}")

    del model, merged_model
    torch.cuda.empty_cache()
    return merged_path


# ======================================================================
# 5. Quantization of the merged HF model
# ======================================================================
def quantize_model(model_path: str, output_quant_dir: str, method: str = "autoawq") -> str:
    if method == "autoawq" and AWQ_AVAILABLE:
        model = AutoAWQForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.quantize(tokenizer, quant_config={"zero_point": True, "q_group_size": 128})
        model.save_quantized(output_quant_dir)
        tokenizer.save_pretrained(output_quant_dir)
        return output_quant_dir
    elif method == "autoawq":
        logger.warning("AutoAWQ not installed, falling back to bitsandbytes (disk size unchanged).")
    # bitsandbytes fallback
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
    model.save_pretrained(output_quant_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_quant_dir)
    return output_quant_dir


# ======================================================================
# 6. Main pipeline – with auto .gguf scanning
# ======================================================================
def run_finetuning_pipeline(
    manual_examples_str: str,
    base_model_id: Optional[str] = None,
    uploaded_gguf_path: Optional[str] = None,
    output_dir: str = "./fine_tuned_model",
    export_gguf: bool = False,
    quantize: bool = False,
    quant_method: str = "autoawq",
    gguf_quant_type: str = "q8_0",
    auto_scan_gguf: bool = True,
    gguf_scan_folder: Optional[str] = None,
) -> str:
    """Full pipeline."""
    # Resolve GGUF path
    if uploaded_gguf_path is None and auto_scan_gguf:
        scanned = find_gguf_files(gguf_scan_folder)
        if scanned:
            uploaded_gguf_path = scanned[0]
            logger.info(f"Auto‑detected GGUF file: {uploaded_gguf_path}")
        else:
            logger.info("No .gguf file found; will use base_model_id if provided.")

    # Prepare base model directory
    if uploaded_gguf_path:
        if not base_model_id:
            raise ValueError("When using a GGUF file, base_model_id must be specified.")
        hf_base_dir = os.path.join(output_dir, "gguf_converted_base")
        convert_gguf_to_hf(uploaded_gguf_path, base_model_id, hf_base_dir)
    else:
        if not base_model_id:
            raise ValueError("Provide base_model_id, upload a GGUF, or place a .gguf file in the project folder.")
        hf_base_dir = base_model_id

    # Build dataset
    dataset = create_dataset_from_manual(manual_examples_str)

    # Fine‑tune
    merged_path = fine_tune_with_lora(
        base_model_dir=hf_base_dir,
        dataset=dataset,
        output_dir=output_dir,
    )

    # Optional quantisation
    if quantize:
        quant_dir = os.path.join(output_dir, "quantized_model")
        merged_path = quantize_model(merged_path, quant_dir, method=quant_method)

    # Optional GGUF export
    if export_gguf:
        gguf_path = os.path.join(output_dir, "fine_tuned.gguf")
        convert_hf_to_gguf(merged_path, gguf_path, quantization_type=gguf_quant_type)
        logger.info(f"GGUF file created: {gguf_path}")
        return gguf_path

    return merged_path


# ======================================================================
# Quick demo
# ======================================================================
if __name__ == "__main__":
    sample_input = (
        "=SYSTEM=\n"
        "You are a cheerful coding tutor.\n"
        "=USER=\n"
        "How do I reverse a string in Python?\n"
        "=ASSISTANT=\n"
        "Use slicing: my_string[::-1]\n"
        "===\n"
        "=USER=\n"
        "What about JavaScript?\n"
        "=ASSISTANT=\n"
        "str.split('').reverse().join('');\n"
        "==="
    )
    final = run_finetuning_pipeline(
        manual_examples_str=sample_input,
        base_model_id="microsoft/phi-2",
        output_dir="./demo_finetuned",
        export_gguf=False,
        quantize=False,
        auto_scan_gguf=True,
    )
    print(f"Done! Model saved at: {final}")
