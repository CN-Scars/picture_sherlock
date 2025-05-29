#!/bin/bash

echo "Starting Picture Sherlock..."
echo "========================================"

# Read Python path from .env file
if [ -f .env ]; then
    source .env
fi

# Check if Python path is set
if [ -z "$PYTHON_PATH" ] || [ ! -f "$PYTHON_PATH" ]; then
    echo "No valid Python path found. Please set the correct Python interpreter path."
    echo "You can set the Python path by:"
    echo "1. Editing the .env file in the project root directory, setting the PYTHON_PATH variable"
    echo "2. Or by entering the full path to your Python interpreter now"
    echo ""
    read -p "Enter the full path to your Python interpreter (press Enter to skip and use system default): " user_python_path
    
    if [ ! -z "$user_python_path" ]; then
        if [ -f "$user_python_path" ]; then
            # Update .env file
            if grep -q "^PYTHON_PATH=" .env; then
                sed -i "s|^PYTHON_PATH=.*|PYTHON_PATH=$user_python_path|" .env
            else
                echo "PYTHON_PATH=$user_python_path" >> .env
            fi
            PYTHON_PATH=$user_python_path
            echo "Python path updated to: $PYTHON_PATH"
        else
            echo "Error: The specified path '$user_python_path' does not exist or is not a valid file."
            echo "Will use system default Python instead."
            PYTHON_PATH=$(which python3 2>/dev/null || which python)
        fi
    else
        # Use system default Python
        PYTHON_PATH=$(which python3 2>/dev/null || which python)
        echo "Will use system default Python: $PYTHON_PATH"
    fi
fi

# Ensure we have a valid Python path
if [ -z "$PYTHON_PATH" ] || [ ! -f "$PYTHON_PATH" ]; then
    if command -v python3 &>/dev/null; then
        PYTHON_PATH=$(which python3)
    elif command -v python &>/dev/null; then
        PYTHON_PATH=$(which python)
    else
        echo "Error: Python interpreter not found. Please install Python 3.7+ and try again."
        exit 1
    fi
fi

echo "Using Python path: $PYTHON_PATH"

# Check Python version
python_version=$("$PYTHON_PATH" -c "import sys; print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))")
echo "Python version: $python_version"

# Ensure Python version >= 3.6
if [[ ! "$python_version" =~ ^3\.[6-9]|^3\.[1-9][0-9] ]]; then
    echo "Warning: Python 3.6 or higher is recommended. Current version: $python_version"
    read -p "Continue anyway? (y/n): " continue_anyway
    if [[ ! "$continue_anyway" =~ ^[Yy] ]]; then
        echo "Cancelled. Please install Python 3.6+ and try again."
        exit 1
    fi
fi

# Check if virtual environment exists
venv_path=""
if [ -f "venv/bin/activate" ]; then
    echo "Virtual environment detected."
    venv_path="venv/bin/activate"
elif [ -f ".venv/bin/activate" ]; then
    echo "Virtual environment detected."
    venv_path=".venv/bin/activate"
fi

if [ ! -z "$venv_path" ]; then
    echo "Activating virtual environment..."
    source "$venv_path"
fi

# Check dependencies
echo "Checking dependencies..."
if ! "$PYTHON_PATH" -c "import streamlit" &>/dev/null; then
    echo "Required dependencies not installed. Installing now..."
    
    # Check if requirements.txt exists
    if [ -f "requirements.txt" ]; then
        "$PYTHON_PATH" -m pip install -r requirements.txt
        if [ $? -ne 0 ]; then
            echo "Failed to install dependencies. Check the error messages and install manually."
            exit 1
        fi
    else
        echo "Warning: requirements.txt file not found. Installing core dependency..."
        "$PYTHON_PATH" -m pip install streamlit
        if [ $? -ne 0 ]; then
            echo "Failed to install dependencies. Check the error messages and install manually."
            exit 1
        fi
    fi
    
    echo "Dependencies installation complete."
else
    echo "All required dependencies are installed."
fi

# Ensure script has execute permissions
chmod +x "$0"

# Run the app
echo "Running Picture Sherlock..."
"$PYTHON_PATH" -m streamlit run picture_sherlock/app.py -- --lang en "$@" 