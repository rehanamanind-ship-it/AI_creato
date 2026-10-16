import os
from pathlib import Path

from .config import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Dataset,
    LoraConfig,
    TaskType,
    PEFT_AVAILABLE,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_peft_model,
    prepare_model_for_kbit_training,
    torch,
)
from .data import build_dataset_from_roles, parse_equals_input
from .ml import _prepare_model_and_tokenizer, _require_finetuning_dependencies, _require_dataset, gguf_to_hf, hf_to_gguf
from .export import ensure_model_directory, resolve_model_reference


def run_lora_finetuning(base_model_dir, dataset, output_dir, epochs=3, batch_size=4, lr=2e-4, use_4bit=True):
    """Fine-tune with LoRA and return the merged model directory."""
    _require_finetuning_dependencies()
    cuda_available = torch.cuda.is_available()
    use_4bit = use_4bit and cuda_available and hasattr(torch, "cuda")
    supports_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)
    use_bf16 = cuda_available and supports_bf16()
    use_fp16 = cuda_available and not use_bf16
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(base_model_dir):
        raise FileNotFoundError(f"Base model directory does not exist: {base_model_dir}")

    print(f"[terminal] Loading base model from: {base_model_dir}")
    print(f"[terminal] Training dataset rows: {len(dataset)}")
    print(f"[terminal] Fine-tuning config: epochs={epochs}, batch_size={batch_size}, lr={lr}")

    tokenizer, model = _prepare_model_and_tokenizer(base_model_dir, use_4bit, use_bf16)
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

    def tokenize(batch):
        texts = []
        for messages in batch["messages"]:
            msg_text = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)
            texts.append(msg_text)
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
        logging_steps=1,
        save_steps=200,
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    print("[terminal] Starting fine-tuning loop...")
    trainer.train()
    losses = [entry.get("loss") for entry in trainer.state.log_history if "loss" in entry]
    if losses:
        final_loss = losses[-1]
        avg_loss = sum(losses) / len(losses)
        print(f"[terminal] Fine-tuning summary: final_loss={final_loss:.4f}, avg_loss={avg_loss:.4f}")
    model.save_pretrained(os.path.join(output_dir, "adapter"))
    tokenizer.save_pretrained(os.path.join(output_dir, "adapter"))
    merged = model.merge_and_unload()
    merged_path = os.path.join(output_dir, "merged_model")
    merged.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)
    print(f"[terminal] Fine-tuned model saved to: {merged_path}")
    return merged_path


def apply_quantization(model_path: str, output_dir: str, method: str = "autoawq") -> str:
    """Quantize a model directory."""
    model_path = str(ensure_model_directory(model_path, operation="Model to quantize"))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if method == "autoawq":
        try:
            from awq import AutoAWQForCausalLM

            model = AutoAWQForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model.quantize(tokenizer, quant_config={"zero_point": True, "q_group_size": 128})
            model.save_quantized(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"[terminal] Quantized model saved to: {output_dir}")
            return output_dir
        except ImportError:
            pass

    if not (torch.cuda.is_available() and hasattr(torch, "cuda")):
        raise RuntimeError("4-bit quantization requires a CUDA GPU and the 'bitsandbytes' package.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print(f"[terminal] Quantized model saved to: {output_dir}")
    return output_dir


def quantize_model(model_dir: str, output_dir: str, method: str = "autoawq") -> str:
    """Public wrapper for quantizing a model directory."""
    return apply_quantization(model_dir, output_dir, method=method)


def run_finetuning_pipeline(
    training_text,
    base_model_id,
    output_dir="./fine_tuned",
    quantize=False,
    export_gguf=False,
    gguf_quant_type="q8_0",
    uploaded_gguf_path=None,
):
    """Run the full fine-tuning pipeline from equals-formatted training data."""
    _require_finetuning_dependencies()
    parsed = parse_equals_input(training_text)
    if not parsed:
        raise ValueError("No valid training examples found.")

    dataset = build_dataset_from_roles(parsed)

    base_model_id = resolve_model_reference(base_model_id, operation="base model")
    if uploaded_gguf_path:
        uploaded_gguf_path = resolve_model_reference(uploaded_gguf_path, operation="uploaded GGUF")
        if Path(uploaded_gguf_path).suffix.lower() != ".gguf":
            raise ValueError(f"Uploaded GGUF path must end with .gguf: {uploaded_gguf_path}")
        print(f"[terminal] Converting custom GGUF model: {uploaded_gguf_path}")
        base_model_dir = gguf_to_hf(uploaded_gguf_path, base_model_id, os.path.join(output_dir, "gguf_converted"))
    elif Path(base_model_id).exists():
        base_model_dir = str(Path(base_model_id).resolve())
        print(f"[terminal] Using local model directory: {base_model_dir}")
    elif str(base_model_id).lower().endswith(".gguf"):
        if not Path(base_model_id).exists():
            raise FileNotFoundError(f"GGUF model file not found: {base_model_id}")
        print(f"[terminal] Converting GGUF model: {base_model_id}")
        base_model_dir = gguf_to_hf(base_model_id, "distilgpt2", os.path.join(output_dir, "gguf_converted"))
    else:
        base_model_dir = base_model_id
        print(f"[terminal] Using Hugging Face base model: {base_model_id}")

    merged = run_lora_finetuning(base_model_dir, dataset, output_dir)
    if quantize:
        print("[terminal] Quantizing fine-tuned model...")
        merged = apply_quantization(merged, os.path.join(output_dir, "quantized"))
    if export_gguf:
        print("[terminal] Exporting merged model to GGUF...")
        merged = hf_to_gguf(merged, os.path.join(output_dir, "fine_tuned.gguf"), gguf_quant_type)
    return merged


__all__ = ["run_lora_finetuning", "apply_quantization", "quantize_model", "run_finetuning_pipeline"]
