import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List

from .autolearn import AutoLearn
from .config import (
    APP_DIR,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    CountVectorizer,
    DataCollatorForLanguageModeling,
    Dataset,
    LogisticRegression,
    ML_AVAILABLE,
    Pipeline,
    PRESETS,
    SKLEARN_AVAILABLE,
    TrainingArguments,
    Trainer,
    logger,
)
from .data import parse_training_file


def choose_training_route(rows: Iterable[Dict[str, str]], route: str = "auto") -> str:
    """Choose the best route for training based on dataset size and available libraries."""
    if route != "auto":
        return route

    rows_list = list(rows)
    if not rows_list:
        return "sklearn" if SKLEARN_AVAILABLE else "transformers"

    if SKLEARN_AVAILABLE and len(rows_list) <= 250:
        average_words = sum(len(row.get("input", "").split()) for row in rows_list) / len(rows_list)
        if average_words <= 20:
            return "sklearn"
    return "transformers"


def resolve_preset(rows: Iterable[Dict[str, str]], preset_name: str = "auto"):
    """Resolve a preset from AutoLearn or a user-specified choice."""
    if preset_name and preset_name != "auto":
        return PRESETS[preset_name]

    candidate_rows = list(rows)
    try:
        learner = AutoLearn()
        choice = learner.choose_model(candidate_rows)
        for key, value in PRESETS.items():
            if value["name"].lower() == choice.lower():
                return value
    except Exception:
        logger.debug("AutoLearn could not pick a preset; falling back to default.", exc_info=True)

    return PRESETS["slm"] if len(candidate_rows) < 100 else PRESETS["llm"]


def train_sklearn_model(
    input_file: str,
    *,
    output_dir: str,
    model_name: str = "logistic-regression",
) -> str:
    """Train a small scikit-learn classifier from input=output pairs."""
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is not installed. Install with: pip install scikit-learn")

    rows = parse_training_file(input_file)
    if not rows:
        raise ValueError(f"No valid training examples were found in {input_file!r}.")

    prompts = [row["input"] for row in rows]
    labels = [row["output"] for row in rows]

    if model_name == "logistic-regression":
        model = Pipeline([
            ("vectorizer", CountVectorizer()),
            ("classifier", LogisticRegression(max_iter=2000, multi_class="auto")),
        ])
    else:
        raise ValueError(f"Unsupported sklearn model: {model_name}")

    model.fit(prompts, labels)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_path = Path(output_dir) / "model.pkl"
    metadata_path = Path(output_dir) / "metadata.json"
    with open(model_path, "wb") as handle:
        pickle.dump(model, handle)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump({"route": "sklearn", "model": model_name, "labels": sorted(set(labels))}, handle)

    print("[terminal] Dataset loaded: {} rows".format(len(rows)))
    print("[terminal] Model trained using scikit-learn route: {}".format(model_name))
    return str(output_dir)


def train_model_from_file(
    input_file: str,
    *,
    preset: str = "auto",
    output_dir: str = str(APP_DIR / "trained_model"),
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
    route: str = "auto",
    model: str = "auto",
) -> str:
    """Train a simple input/output model from a file and return the model directory."""
    rows = parse_training_file(input_file)
    if not rows:
        raise ValueError(f"No valid training examples were found in {input_file!r}.")

    selected_route = choose_training_route(rows, route)
    print("[terminal] Dataset loaded: {} rows from {}".format(len(rows), input_file))
    print("[terminal] Selected route: {}".format(selected_route))

    if selected_route == "sklearn":
        sklearn_model = model if model != "auto" else "logistic-regression"
        return train_sklearn_model(input_file, output_dir=output_dir, model_name=sklearn_model)

    if not ML_AVAILABLE:
        raise RuntimeError("Machine learning dependencies are not installed. Install the required packages first.")

    selected_preset = resolve_preset(rows, preset)
    if model != "auto":
        selected_preset = {**selected_preset, "model": model}
    print("[terminal] Training with: {} ({})".format(selected_preset["name"], selected_preset["model"]))

    if Dataset is None:
        raise ImportError("The 'datasets' library is required. Install with: pip install datasets")

    tokenizer = AutoTokenizer.from_pretrained(selected_preset["model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.mask_token

    task = selected_preset["task"]
    records = []
    for row in rows:
        if task == "masked":
            records.append({"text": f"{row['input']} {tokenizer.mask_token} {row['output']}"})
        else:
            records.append({"text": f"Input: {row['input']}\nOutput: {row['output']}"})

    dataset = Dataset.from_list(records)
    tokenized = dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128),
        batched=True,
        remove_columns=["text"],
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=(task == "masked"),
        mlm_probability=0.15,
    )

    if task == "masked":
        model_obj = AutoModelForMaskedLM.from_pretrained(selected_preset["model"])
    else:
        model_obj = AutoModelForCausalLM.from_pretrained(selected_preset["model"])
        model_obj.config.pad_token_id = tokenizer.pad_token_id

    args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_strategy="no",
        logging_steps=5,
        report_to=[],
        learning_rate=learning_rate,
    )
    Trainer(model=model_obj, args=args, train_dataset=tokenized, data_collator=collator).train()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_obj.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[terminal] Model trained successfully.")
    return output_dir


__all__ = ["choose_training_route", "resolve_preset", "train_sklearn_model", "train_model_from_file"]
