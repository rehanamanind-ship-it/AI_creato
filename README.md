# AI Creator App

A desktop app for fine-tuning small Hugging Face language models from simple
`input = output` text files.

## Features

- Keeps the existing `input = output` training data format
- Trains with Hugging Face `transformers`
- Supports Auto Learn, LLM, MLM, and SLM training modes
- Tests the trained model inside the app
- Exports trained models to `.gguf` with 4q, 8q, 16q, or 36q selection
- Uses a black UI with blue outlined controls

## Installation

Install Python, then install the required packages:

```bash
pip install -r requirements.txt
```

The first training run downloads the selected Hugging Face base model.

## Data Format

Training data should be in a `.txt` file with each line in the format:

```text
input = output
```

Examples:

```text
hello = hi, how can I help?
what is ai = AI means artificial intelligence.
3bed 2bath 1500sqft = 300000
```

## GGUF Export

GGUF export requires local `llama.cpp` tools:

- `convert_hf_to_gguf.py`
- `llama-quantize.exe`

Put `llama.cpp` inside this project folder or set `LLAMA_CPP_DIR` to your
`llama.cpp` folder. The app will raise an error if you click "Download .gguf"
without selecting 4q, 8q, 16q, or 36q first.

## Running the App

```bash
python app.py
```
