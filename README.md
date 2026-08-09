# AI Creator

A single‑file, no‑code terminal application that creates brand‑new language models from `input = output` text files **and** fine‑tunes existing models using structured role‑based examples. Export your models as quantized `.gguf` files ready for llama.cpp, Ollama, or GPT4All – all without writing any code or JSONL.

## ✨ Features

- **Train from scratch** – provide a simple `input = output` text file and a base model type (Auto Learn, LLM, MLM, SLM).
- **Fine‑tune with roles** – paste SYSTEM/USER/ASSISTANT examples (separated by `===`) to adapt a model’s behavior.
- **Upload .gguf as base** – fine‑tune directly on a local GGUF file (requires the corresponding HuggingFace architecture ID).
- **Smart model selection** – optionally uses `autolearn` to pick the best base model size.
- **Quantized GGUF export** – choose from 4‑bit, 8‑bit, f16, or other quantization levels.
- **Interactive testing** – test your trained models with custom prompts right in the terminal.
- **No external LLM** – all training data comes from your input, no API keys needed.

## 📦 Installation

1. **Clone the repository** (or download `main_cli.py`):
   
    Install Python dependencies:
    bash

    pip install torch transformers datasets peft accelerate bitsandbytes trl gguf

    (Optional) For better auto‑model selection:
    bash

    pip install autolearn

    (Optional) For higher‑quality 4‑bit quantization (Linux only):
    bash

    pip install autoawq

    Set up llama.cpp (required for GGUF export):

        Clone and build llama.cpp.

        Set the environment variable pointing to its directory:
        bash

        export LLAMA_CPP_DIR=/path/to/llama.cpp

🚀 Usage

Run the application:
bash

python main_cli.py

You will see an interactive menu:
text

========================================
AI CREATOR – Terminal Edition
========================================
1. Train new model (input = output text file)
2. Fine‑tune with roles (equals‑format)
3. Export trained model to GGUF
4. Test prediction
5. Quit

1. Train a new model

    Choose option 1.

    Enter the path to a training file where each line is input = output.
    Example training.txt:
    text

    Hello = Hi there!
    How are you? = I'm fine, thank you.
    What is Python? = Python is a programming language.

    Pick a base model type (e.g., 2 for LLM, 4 for SLM, or 1 for Auto Learn).

    Training will start and the model will be saved in the trained_model/ folder.

2. Fine‑tune with roles

    Choose option 2.

    Paste your role‑based examples or provide a file path. Use the format:
    text

    =SYSTEM=
    You are a cheerful coding tutor.
    =USER=
    How do I reverse a string in Python?
    =ASSISTANT=
    Use slicing: my_string[::-1]
    ===
    =USER=
    What about JavaScript?
    =ASSISTANT=
    str.split('').reverse().join('');
    ===

    Type END on a new line to finish pasting.

    Choose a base model (or let Auto Learn decide).

    Optionally, upload a .gguf file as the base – you’ll be asked for its HuggingFace architecture ID.

    Decide whether to quantize or directly export as GGUF after training.

    The fine‑tuned model is saved in fine_tuned_model/.

3. Export to GGUF

    Choose option 3 after training or fine‑tuning.

    Select a quantization level (e.g., 3 for f16, 1 for 4‑bit).

    Provide an output file name (e.g., my_model.gguf).

    The app uses llama.cpp tools to produce the GGUF file.

4. Test prediction

    Choose option 4 to test the latest standard‑trained model.

    Enter any prompt; type exit to return to the menu.

📁 File Structure
text

ai-creato/
├── app.py         # The complete application (training + fine‑tuning + export)
├── requirements.txt # (optional) list of dependencies
|---- autolearn.py
└── README.md

(No separate fine_tuner.py is needed – everything is integrated in app.py.)
🧠 How It Works

    Standard training uses HuggingFace Trainer to fine‑tune a small base model (like DistilGPT2) on your input‑output pairs.

    Fine‑tuning with roles uses QLoRA (4‑bit quantization + LoRA) to efficiently adapt any causal language model to follow the SYSTEM/USER/ASSISTANT format you provide.

    GGUF conversion calls convert_hf_to_gguf.py from llama.cpp and then applies quantization with llama-quantize.

⚠️ Requirements & Notes

    GPU recommended for faster training; CPU works but is slow.

    llama.cpp must be installed and compiled to use GGUF export. The app searches for it via the LLAMA_CPP_DIR environment variable or a local llama.cpp folder.

    Tkinter is not required – this is a pure terminal application.

    The autolearn package is optional; if missing, the app uses a simple size heuristic.

    For fine‑tuning with a GGUF base, the gguf Python package is required (pip install gguf).

👤 Author & Contributions

Rehan Aman – creator and maintainer of AI Creator.

Contributions, issues, and feature requests are welcome!
Feel free to open an issue or submit a pull request on the project repository.
📄 License

This project is open‑source under the MIT License.
