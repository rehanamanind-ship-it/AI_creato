# AI_creato

<sub>Terminal-first AI builder for training, fine-tuning, export, quantization, and testing.</sub>

Creato AI or AI_creato is a lightweight Python framework for running AI workflows directly from the terminal. It is designed for developers who want to train models, fine-tune language models, export to GGUF, quantize models, and test outputs without a graphical UI.

## Features

- Train small dataset models from simple `input = output` files
- Fine-tune Hugging Face models from conversation-style examples
- Export trained models to GGUF format
- Quantize models for smaller and faster deployment
- Test saved models with direct terminal prompts
- Keep the whole workflow scriptable and automation-friendly

## Why use it?

Creato AI is aimed at:

- local AI experimentation from the shell
- easy dataset-driven model training
- quick testing of model behavior without a notebook UI
- simple deployment pipelines for terminal-based tools and scripts

## Installation

### Install from a wheel file

This is the recommended way if you already have the built wheel locally.

```bash
python -m pip install dist/creato_ai-0.1.0-py3-none-any.whl
```

After installation, the package dependencies are installed automatically by pip, and the CLI command becomes available:

```bash
creato-ai --help
creato-ai info
```

### Optional GPU support

```bash
python -m pip install -e .[gpu]
```

## Quick start

### 1) Show CLI help

```bash
creato-ai --help
```

### 2) Check environment status

```bash
creato-ai info
```

### 3) Train a model

```bash
creato-ai train examples/training.txt --preset auto --route auto --output-dir ./trained_model
```

### 4) Fine-tune a base model

```bash
creato-ai fine-tune examples/fine_tune.txt --base-model distilgpt2 --output-dir ./fine_tuned_model
```

### 5) Export to GGUF

```bash
creato-ai export ./trained_model --output ./model.gguf --quant-type q8_0
```

### 6) Quantize a model

```bash
creato-ai quantize ./trained_model --output-dir ./quantized_model --method bnb
```

### 7) Test a model from the terminal

```bash
creato-ai test ./trained_model --prompt "hello there"
```

## Data formats

### Training file format

```text
hello = hi there
how are you = I am doing well
what is your name = I am a helpful assistant
```

### Fine-tuning format

```text
=SYSTEM= You are a helpful assistant
=USER= What is the capital of France?
=ASSISTANT= The capital of France is Paris.
===
=USER= What is 2 + 2?
=ASSISTANT= 4
```

## Example workflow

```bash
creato-ai train examples/training.txt --output-dir ./models/my_model
creato-ai test ./models/my_model --prompt "How do I get started with Python?"
creato-ai export ./models/my_model --output ./models/my_model.gguf --quant-type q8_0
```

## Project structure

```text
creato.ai/
├── app.py
├── src/
│   └── creato_ai/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── data.py
│       ├── training.py
│       ├── finetune.py
│       ├── export.py
│       ├── ml.py
│       └── autolearn.py
├── tests/
|    app_test.py
├── requirements.txt
├── README.md
└── LICENSE
```

## Packaging and wheel usage

Build the wheel locally:

```bash
python -m build
```

Install the generated wheel:

```bash
python -m pip install dist/creato_ai-0.1.0-py3-none-any.whl
```

Then run the app normally:

```bash
creato-ai --help
creato-ai info
```

This is the intended workflow for end users: install the wheel, let pip install the dependencies, and use the CLI like a normal application.

## Notes

- This project is intended for terminal-first AI workflows rather than graphical UIs.
- For large model workloads, ensure the required ML libraries are installed in the target environment.
- Dependency installation is handled automatically through the package metadata when installing from PyPI.

## License

This project is available under the MIT license.
