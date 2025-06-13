#!/bin/bash

# Script to set up the Python virtual environment and install dependencies
# for the project using uv, specifically targeting QDax compatibility.

PYTHON_UV_SPECIFIER="3.11" # Use this to tell uv which Python to use for the venv
                           # From your `uv python list`, this should resolve to the managed 3.11.11
VENV_NAME=".venv"          # Name of the virtual environment directory

echo "--- QDax BinPack Heuristics Environment Setup using uv ---"

# 1. Check for uv
echo -e "\n[Step 1/6] Checking for uv..." # Update total steps
if ! command -v uv &> /dev/null; then
    echo "ERROR: 'uv' command not found."
    echo "Please install uv: https://github.com/astral-sh/uv (e.g., curl -LsSf https://astral.sh/uv/install.sh | sh)"
    exit 1
fi
echo "uv found: $(uv --version)"
echo "uv known Python versions (output of 'uv python list'):"
uv python list # Show the user what uv sees

# 2. Create/Recreate virtual environment using uv with the Python specifier
echo -e "\n[Step 2/6] Creating virtual environment '${VENV_NAME}' using Python specifier '${PYTHON_UV_SPECIFIER}' via uv..."
if [ -d "${VENV_NAME}" ]; then
    echo "Removing existing virtual environment: ${VENV_NAME}"
    rm -rf "${VENV_NAME}"
fi

# Tell uv to use the Python version specifier.
uv venv "${VENV_NAME}" -p "${PYTHON_UV_SPECIFIER}" --seed
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment with uv using Python specifier '${PYTHON_UV_SPECIFIER}'."
    echo "Please check 'uv python list' to ensure a version matching '${PYTHON_UV_SPECIFIER}' is available or downloadable by uv."
    exit 1
fi

# Verify the Python version in the created venv
VENV_PYTHON_EXE="${PWD}/${VENV_NAME}/bin/python"
if [ ! -f "${VENV_PYTHON_EXE}" ]; then
    echo "ERROR: Virtual environment Python executable not found at ${VENV_PYTHON_EXE}"
    exit 1
fi
ACTUAL_VENV_PYTHON_VERSION=$(${VENV_PYTHON_EXE} --version 2>&1)
echo "Virtual environment '${VENV_NAME}' created successfully."
echo "Python version in venv: ${ACTUAL_VENV_PYTHON_VERSION}"

if [[ "${ACTUAL_VENV_PYTHON_VERSION}" != *"Python ${PYTHON_UV_SPECIFIER}"* ]]; then
    echo "WARNING: Created venv Python version (${ACTUAL_VENV_PYTHON_VERSION}) does not precisely match target specifier ${PYTHON_UV_SPECIFIER}."
    echo "However, if it's a 3.11.x version, it should be okay. Continuing..."
fi


# 3. Install QDax with constraint (within the created venv)
QDaxExtras=""
echo -e "\n[Step 3/6] Installing qdax${QDaxExtras} with constraints into '${VENV_NAME}'..."
uv pip install --python "${VENV_PYTHON_EXE}" "qdax${QDaxExtras}" --constraint https://raw.githubusercontent.com/adaptive-intelligent-robotics/QDax/main/requirements.txt

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install qdax with constraints using ${ACTUAL_VENV_PYTHON_VERSION}."
    exit 1
fi
echo "qdax installed successfully."

# 4. Install other dependencies from requirements.txt
echo -e "\n[Step 4/6] Installing other dependencies from requirements.txt into '${VENV_NAME}'..."
if [ -f "requirements.txt" ]; then
    grep -vE '^(# )?[qQ][dD][aA][xX]' requirements.txt > temp_requirements.txt
    if [ -s temp_requirements.txt ]; then
        echo "Installing from filtered requirements: $(cat temp_requirements.txt | tr '\n' ' ')"
        uv pip install --python "${VENV_PYTHON_EXE}" -r temp_requirements.txt
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to install dependencies from temp_requirements.txt."
            rm temp_requirements.txt
            exit 1
        fi
    else
        echo "No other dependencies found in requirements.txt after filtering qdax."
    fi
    rm temp_requirements.txt
else
    echo "requirements.txt not found, skipping."
fi
echo "Other dependencies installed."

# 5. NEW: Replace Jumanji with custom version from Git
echo -e "\n[Step 5/6] Replacing Jumanji with custom version from Git..."
echo "First, uninstalling any existing jumanji package..."
uv pip uninstall --python "${VENV_PYTHON_EXE}" jumanji
# We don't exit on failure, as it might not have been installed, which is fine.
# The important part is that no standard version remains.

echo "Now, installing custom jumanji from git+https://github.com/Hata41/jumanji_value_based.git..."
uv pip install --python "${VENV_PYTHON_EXE}" "git+https://github.com/Hata41/jumanji_value_based.git"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install custom Jumanji from the Git repository."
    exit 1
fi
echo "Custom Jumanji installed successfully."


# 6. Install the local project package in editable mode (renumbered)
echo -e "\n[Step 6/6] Installing the 'qdax_binpack' project in editable mode..."
uv pip install --python "${VENV_PYTHON_EXE}" -e .
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install the project package in editable mode."
    echo "Make sure you have a pyproject.toml file in the root directory."
    exit 1
fi
echo "Project package installed."


echo -e "\n--- Setup Complete ---"
echo "Virtual environment '${VENV_NAME}' is ready with ${ACTUAL_VENV_PYTHON_VERSION}."
echo "To activate it in your current shell session, run:"
echo "  source ${VENV_NAME}/bin/activate"
echo "Then you can run your Python scripts."