import logging
from pathlib import Path

try:
    import torch
    from datasets import Dataset
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
        pipeline,
    )
    ML_AVAILABLE = True
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None
    Dataset = None
    AutoConfig = None
    AutoModelForCausalLM = None
    AutoModelForMaskedLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None
    DataCollatorForLanguageModeling = None
    Trainer = None
    TrainingArguments = None
    pipeline = None
    ML_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - optional runtime dependency
    CountVectorizer = None
    LogisticRegression = None
    Pipeline = None
    SKLEARN_AVAILABLE = False

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType,
    )
    PEFT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional runtime dependency
    PEFT_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = APP_DIR / "trained_model"
GGUF_DIR = APP_DIR / "gguf_models"
FINE_TUNED_DIR = APP_DIR / "fine_tuned_model"

PRESETS = {
    "auto": {"name": "Auto Learn", "task": "causal", "model": "sshleifer/tiny-gpt2"},
    "llm": {"name": "LLM", "task": "causal", "model": "distilgpt2"},
    "mlm": {"name": "MLM", "task": "masked", "model": "distilroberta-base"},
    "slm": {"name": "SLM", "task": "causal", "model": "sshleifer/tiny-gpt2"},
}

QUANT_OPTIONS = {
    "1": ("4q", "q4_0"),
    "2": ("8q", "q8_0"),
    "3": ("16q", "f16"),
    "4": ("36q", "q6_k"),
}

__all__ = [
    "APP_DIR",
    "GGUF_DIR",
    "FINE_TUNED_DIR",
    "MODEL_DIR",
    "PRESETS",
    "QUANT_OPTIONS",
    "ML_AVAILABLE",
    "SKLEARN_AVAILABLE",
    "PEFT_AVAILABLE",
    "logger",
    "torch",
    "Dataset",
    "AutoConfig",
    "AutoModelForCausalLM",
    "AutoModelForMaskedLM",
    "AutoTokenizer",
    "BitsAndBytesConfig",
    "DataCollatorForLanguageModeling",
    "Trainer",
    "TrainingArguments",
    "pipeline",
    "CountVectorizer",
    "LogisticRegression",
    "Pipeline",
    "LoraConfig",
    "get_peft_model",
    "prepare_model_for_kbit_training",
    "TaskType",
]
