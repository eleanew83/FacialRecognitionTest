# Environment Setup Guide

This guide provides detailed instructions for setting up the development environment on different operating systems. The recommended setup for NVIDIA GPU users on Windows is using WSL2.

## 1. Prerequisites

- **Git**: To clone the repository.
- **Python**: Version 3.10 or newer.
- **NVIDIA Driver** (for GPU users): Ensure you have the latest NVIDIA drivers installed on your host machine.

## 2. Clone the Repository

First, clone the project repository to your local machine:

```bash
git clone <your-repository-url>
cd animal-face-id
```

## 3. Platform-Specific Setup

Choose the instructions corresponding to your operating system.

---

### 🐧 Windows with WSL2 + Ubuntu (Recommended for NVIDIA GPU)

Using WSL2 (Windows Subsystem for Linux) provides the best performance and compatibility for PyTorch with NVIDIA GPUs on a Windows machine.

#### 3.1. Initial WSL2 and GPU Setup

1.  **Install WSL2**: If you haven't already, open PowerShell as an Administrator and run:
    ```powershell
    wsl --install
    ```
    Restart your computer after the installation is complete.

2.  **Verify NVIDIA Driver**: In a Windows PowerShell terminal, check that your driver is recognized:
    ```powershell
    nvidia-smi
    ```
    You should see your GPU details (e.g., NVIDIA GeForce RTX 5080).

3.  **Verify GPU Access in WSL2**: Open your WSL2 Ubuntu terminal and run the same command:
    ```bash
    nvidia-smi
    ```
    If you see the same GPU table, WSL2 is correctly configured to access your GPU.

#### 3.2. Python Environment Setup in WSL2

1.  **Navigate to Project Directory**:
    Your Windows file system is mounted under `/mnt/`. For example:
    ```bash
    # If your repo is at C:\Users\jones\Downloads\animal-face-id
    cd /mnt/c/Users/jones/Downloads/animal-face-id
    ```

2.  **Install Python and venv**:
    ```bash
    sudo apt update
    sudo apt install -y python3 python3-venv python3-pip
    python3 --version # Verify it's 3.10+
    ```

3.  **Create and Activate Virtual Environment**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    Your terminal prompt should now be prefixed with `(.venv)`.

4.  **Install Dependencies**:
    ```bash
    # Upgrade pip
    pip install --upgrade pip

    # Install PyTorch with CUDA support (check pytorch.org for the latest command)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    # Install project-specific packages
    pip install -r requirements.txt
    ```

---

### 🪟 Native Windows (Without WSL)

1.  **Create and Activate Virtual Environment**:
    ```powershell
    # In PowerShell or CMD
    python -m venv .venv

    # Activate (PowerShell)
    .\.venv\Scripts\Activate.ps1
    # Or if you are using CMD
    # .\.venv\Scripts\activate.bat
    ```

2.  **Install Dependencies**:
    ```bash
    # Install PyTorch with CUDA support
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    # Install project-specific packages
    pip install -r requirements.txt
    ```

---

### 🍎 macOS (Apple Silicon or Intel)

1.  **Create and Activate Virtual Environment**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install Dependencies**:
    ```bash
    # For Apple Silicon (M1/M2/M3), this will use MPS acceleration
    # For Intel Macs, this will be CPU-only
    pip install torch torchvision torchaudio

    # Install project-specific packages
    pip install -r requirements.txt
    ```

## 4. Verify Installation

After completing the setup, run the following Python snippet to verify that PyTorch and your hardware acceleration are working correctly.

```bash
python -c "
import torch
import platform

print(f'Python Version: {platform.python_version()}')
print(f'PyTorch Version: {torch.__version__}')

if torch.cuda.is_available():
    print(f'CUDA is available.')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
elif torch.backends.mps.is_available():
    print('Apple MPS is available.')
else:
    print('No hardware acceleration found. Using CPU.')
"
```

#### Expected Output (for NVIDIA GPU user):
```
Python Version: 3.11.5
PyTorch Version: 2.1.0+cu121
CUDA is available.
GPU: NVIDIA GeForce RTX 5080
```

#### Expected Output (for Apple Silicon user):
```
Python Version: 3.11.5
PyTorch Version: 2.1.0
Apple MPS is available.
```

Your environment is now ready for data preparation and model training.
