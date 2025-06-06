# QDax BinPack Heuristics

Evolving heuristic policies for Jumanji BinPack-v2 using QDax.

## Setup

This project uses `uv` for environment and package management. A setup script is provided to automate the process.

**Prerequisites:**
*   **`uv`:** Ensure `uv` is installed. If not, see [uv installation guide](https://github.com/astral-sh/uv) (e.g., `curl -LsSf https://astral.sh/uv/install.sh | sh`).
*   **Python:** The setup script aims to create a **Python 3.11** virtual environment, as this version is generally most compatible with the project's dependencies (especially QDax and its constraints). `uv` will attempt to use an existing Python 3.11 or download it if necessary and possible.

**Instructions:**

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Run the setup script:**
    Navigate to the project root directory (where `setup_env.sh` is located) and execute:
    ```bash
    chmod +x setup_env.sh
    ./setup_env.sh
    ```
    This script will:
    *   Create a local Python 3.11 virtual environment named `.venv`.
    *   Install `qdax` using specific version constraints.
    *   Install other dependencies from `requirements.txt`.

3.  **Activate the virtual environment:**
    After the script completes, activate the environment in your terminal:
    ```bash
    source .venv/bin/activate
    ```
    Your prompt should now indicate you are in the `(.venv)` environment.

## Running Experiments

With the virtual environment activated:

*   **Evolve Heuristic Policies (Multi-Device):**
    ```bash
    python heuristic_multi_device.py 
    ```
    (Configure `descriptor_type` inside this script to switch between genome-based or activity-based descriptors).

*   **Evolve Neural Network Policies (Multi-Device):**
    ```bash
    XLA_FLAGS="--xla_force_host_platform_device_count=8" python3 run_qdax_multi_device.py
    ```

Refer to the scripts themselves for hyperparameter configurations.