#!/usr/bin/env bash
# setup.sh – Environment setup & dependency installer for AI Creator + Fine‑Tuner
set -euo pipefail

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
VENV_DIR=".venv"
REQUIREMENTS_FILE="requirements_ai_creator.txt"
PYTHON_MIN_VERSION="3.8"

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
print_error() { echo -e "\033[0;31m[ERROR]\033[0m $*"; }
print_success() { echo -e "\033[0;32m[OK]\033[0m $*"; }
print_info() { echo -e "\033[0;34m[INFO]\033[0m $*"; }
print_warning() { echo -e "\033[0;33m[WARNING]\033[0m $*"; }

check_command() {
    command -v "$1" >/dev/null 2>&1 || {
        print_error "$1 is not installed."
        return 1
    }
}

# ----------------------------------------------------------------------
# 1. Check Python version
# ----------------------------------------------------------------------
check_python() {
    print_info "Checking Python version..."
    if ! check_command python3; then
        print_error "Python3 is required. Please install Python $PYTHON_MIN_VERSION or later."
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if [[ $(echo "$PYTHON_VERSION < $PYTHON_MIN_VERSION" | bc -l 2>/dev/null || true) -eq 1 ]]; then
        print_error "Python $PYTHON_MIN_VERSION or higher is required. Found $PYTHON_VERSION."
        exit 1
    fi
    print_success "Python $PYTHON_VERSION detected."
}

# ----------------------------------------------------------------------
# 2. Create & activate virtual environment
# ----------------------------------------------------------------------
setup_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        print_info "Creating virtual environment in $VENV_DIR..."
        python3 -m venv "$VENV_DIR"
    else
        print_info "Virtual environment already exists."
    fi
    # Activate
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
    print_success "Virtual environment activated."
}

# ----------------------------------------------------------------------
# 3. Install system dependencies (Linux only)
# ----------------------------------------------------------------------
install_system_deps() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        print_info "Checking/installing system build tools (required for llama-cpp, bitsandbytes)..."
        if check_command apt-get; then
            sudo apt-get update -qq
            sudo apt-get install -y -qq build-essential cmake python3-dev || {
                print_warning "Could not install some build dependencies. Compilation may fail."
            }
        elif check_command dnf; then
            sudo dnf install -y gcc-c++ cmake python3-devel || true
        elif check_command pacman; then
            sudo pacman -S --noconfirm base-devel cmake || true
        else
            print_warning "Unknown package manager. Please install build-essential, cmake, and python3-dev manually."
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        print_info "macOS detected: Xcode command line tools should be installed."
        xcode-select --install 2>/dev/null || true
    fi
}

# ----------------------------------------------------------------------
# 4. Write requirements file (including fine_tuner dependencies)
# ----------------------------------------------------------------------
write_requirements() {
    print_info "Generating $REQUIREMENTS_FILE..."
    cat > "$REQUIREMENTS_FILE" << 'EOF'
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
peft>=0.4.0
accelerate>=0.20.0
bitsandbytes>=0.40.0
trl>=0.7.0
gguf>=0.6.0
autoawq>=0.1.0; platform_system == 'Linux'
scipy
sentencepiece
protobuf
tiktoken
huggingface_hub
tkinter
EOF

    # llama-cpp-python requires special handling (GPU/CPU)
    # We'll install it separately after PyTorch is installed.
    print_success "Requirements file created."
}

# ----------------------------------------------------------------------
# 5. Install PyTorch (detect CUDA)
# ----------------------------------------------------------------------
install_pytorch() {
    print_info "Installing PyTorch..."
    if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        print_success "CUDA already available."
    else
        # Try to install CUDA-enabled PyTorch if nvidia-smi exists
        if command -v nvidia-smi &> /dev/null; then
            print_info "NVIDIA GPU detected. Installing PyTorch with CUDA 11.8..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        else
            print_info "No GPU found. Installing CPU-only PyTorch."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        fi
    fi
    print_success "PyTorch installed."
}

# ----------------------------------------------------------------------
# 6. Install all Python packages
# ----------------------------------------------------------------------
install_python_deps() {
    print_info "Installing Python dependencies from $REQUIREMENTS_FILE..."
    pip install --upgrade pip
    pip install -r "$REQUIREMENTS_FILE"
    print_success "Core packages installed."
}

# ----------------------------------------------------------------------
# 7. Install llama-cpp-python with appropriate backend
# ----------------------------------------------------------------------
install_llama_cpp() {
    print_info "Installing llama-cpp-python (GGUF support)..."
    # Try with CUDA if available, else CPU
    if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir || {
            print_warning "CUDA build failed, falling back to CPU."
            pip install llama-cpp-python --force-reinstall --no-cache-dir
        }
    else
        pip install llama-cpp-python --force-reinstall --no-cache-dir
    fi
    print_success "llama-cpp-python installed."
}

# ----------------------------------------------------------------------
# 8. Final checks and fixes
# ----------------------------------------------------------------------
resolve_common_issues() {
    print_info "Running post-install checks..."
    # Ensure bitsandbytes works (may need CUDA)
    python3 -c "import bitsandbytes" 2>/dev/null || {
        print_warning "bitsandbytes failed to import. If you have an NVIDIA GPU, ensure CUDA is properly installed."
        print_warning "For CPU-only, bitsandbytes may not be needed for inference but is used for QLoRA. Consider using a GPU."
    }
    # Check tkinter (usually built-in)
    python3 -c "import tkinter" 2>/dev/null || {
        print_error "tkinter is missing. On Ubuntu/Debian, install python3-tk: sudo apt-get install python3-tk"
        print_error "On macOS with Homebrew: brew install python-tk"
        exit 1
    }
    # Check that fine_tuner module can be imported
    python3 -c "import fine_tuner" 2>/dev/null || {
        print_warning "fine_tuner module not found in current directory. Make sure fine_tuner.py is present."
    }
    print_success "All critical modules can be imported."
}

# ----------------------------------------------------------------------
# 9. (Optional) Run the app
# ----------------------------------------------------------------------
run_app() {
    print_info "Starting AI Creator GUI..."
    python3 main.py
}

# ----------------------------------------------------------------------
# Main execution
# ----------------------------------------------------------------------
main() {
    print_info "=== AI Creator Setup ==="
    check_python
    setup_venv
    install_system_deps
    write_requirements
    install_pytorch
    install_python_deps
    install_llama_cpp
    resolve_common_issues

    print_success "Setup complete!"
    echo ""
    echo "To run the app later, activate the environment and run main.py:"
    echo "  source $VENV_DIR/bin/activate"
    echo "  python main.py"
    echo ""

    # Ask if user wants to launch now
    read -rp "Do you want to start the AI Creator now? [y/N] " answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        run_app
    fi
}

main "$@"