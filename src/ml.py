import os
import re
import subprocess
import sys
from pathlib import Path

from .config import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Dataset,
    ML_AVAILABLE,
    PEFT_AVAILABLE,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    logger,
    torch,
)


def _require_dataset() -> None:
    if Dataset is None:
        raise ImportError("The 'datasets' library is required. Install with: pip install datasets")


def _require_finetuning_dependencies() -> None:
    _require_dataset()
    if not PEFT_AVAILABLE:
        raise ImportError("Fine-tuning requires 'peft'. Install with: pip install peft")


def _find_llama_cpp_file(name: str):
    search_dirs = [Path(__file__).resolve().parents[2] / "llama.cpp"]
    configured_dir = os.environ.get("LLAMA_CPP_DIR")
    if configured_dir:
        search_dirs.append(Path(configured_dir))

    for root in search_dirs:
        if root.exists():
            hits = list(root.rglob(name))
            if hits:
                return hits[0]
    return None


def _map_gguf_to_hf(tensors, config):
    hf_weights = {}
    for gguf_name, weight in tensors.items():
        new_name = gguf_name
        match = re.match(r"blk\.(\d+)\.(.+)", gguf_name)
        if match:
            layer_idx, param = match.group(1), match.group(2)
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
        hf_weights[new_name] = weight
    return hf_weights


def gguf_to_hf(gguf_path: str, base_model_id: str, output_dir: str) -> str:
    """Convert a GGUF file to a Hugging Face model directory."""
    try:
        import gguf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("gguf package required. Install with: pip install gguf") from exc

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


def hf_to_gguf(hf_dir: str, output_gguf: str, quant_type: str = "q8_0") -> str:
    """Export a Hugging Face model directory to GGUF."""
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


def _prepare_model_and_tokenizer(model_id_or_dir: str, use_4bit: bool = True, bf16: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    cuda_available = torch.cuda.is_available()
    if use_4bit and not (cuda_available and True):
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


def _format_chat(example, tokenizer):
    msgs = example["messages"]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(msgs, tokenize=False)
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in msgs)


def generate_text(model_dir: str, prompt: str, *, max_new_tokens: int = 60, temperature: float = 0.7, top_p: float = 0.95) -> str:
    """Generate text from a trained model directory using a command-line prompt."""
    if not prompt:
        raise ValueError("A prompt is required to generate text.")

    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    metadata_path = model_path / "metadata.json"
    if metadata_path.exists():
        import json
        import pickle

        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("route") == "sklearn":
            with open(model_path / "model.pkl", "rb") as handle:
                model = pickle.load(handle)
            result = str(model.predict([prompt])[0])
            print(f"[terminal] Generated output from sklearn model: {result}")
            return result

    if not ML_AVAILABLE:
        raise RuntimeError("Machine learning dependencies are not installed. Install the required packages first.")

    print(f"[terminal] Generating output using model directory: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(str(model_path))
    model.eval()
    encoded = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated


__all__ = [
    "_require_dataset",
    "_require_finetuning_dependencies",
    "_find_llama_cpp_file",
    "_map_gguf_to_hf",
    "gguf_to_hf",
    "hf_to_gguf",
    "_prepare_model_and_tokenizer",
    "_format_chat",
    "generate_text",
]
