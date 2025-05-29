# Set encoding to UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "Starting Picture Sherlock..."
Write-Host "========================================"

# Read Python path from .env file
$PYTHON_PATH = $null
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^PYTHON_PATH=(.*)$") {
            $PYTHON_PATH = $matches[1].Trim('"')
            # Handle environment variables
            $PYTHON_PATH = $PYTHON_PATH -replace '%USERPROFILE%', $env:USERPROFILE
            $PYTHON_PATH = $PYTHON_PATH -replace '\$HOME', $HOME
        }
    }
}

# Check if Python path is set
if ([string]::IsNullOrEmpty($PYTHON_PATH) -or -not (Test-Path $PYTHON_PATH)) {
    Write-Host "No valid Python path found. Please set the correct Python interpreter path."
    Write-Host "You can set the Python path by:"
    Write-Host "1. Editing the .env file in the project root directory, setting the PYTHON_PATH variable"
    Write-Host "2. Or by entering the full path to your Python interpreter now"
    Write-Host ""
    $user_python_path = Read-Host "Enter the full path to your Python interpreter (press Enter to skip and use system default)"
    
    if (-not [string]::IsNullOrEmpty($user_python_path)) {
        if (Test-Path $user_python_path) {
            # Update .env file
            $env_content = Get-Content ".env"
            $python_path_updated = $false
            
            for ($i = 0; $i -lt $env_content.Count; $i++) {
                if ($env_content[$i] -match "^PYTHON_PATH=") {
                    $env_content[$i] = "PYTHON_PATH=`"$user_python_path`""
                    $python_path_updated = $true
                    break
                }
            }
            
            if (-not $python_path_updated) {
                $env_content += "PYTHON_PATH=`"$user_python_path`""
            }
            
            $env_content | Set-Content ".env"
            $PYTHON_PATH = $user_python_path
            Write-Host "Python path updated to: $PYTHON_PATH"
        } else {
            Write-Host "Error: The specified path '$user_python_path' does not exist or is not a valid file."
            Write-Host "Will use system default Python instead."
            $PYTHON_PATH = $null
        }
    } else {
        # Use system default Python
        $PYTHON_PATH = $null
    }
}

# Ensure we have a valid Python path
if ([string]::IsNullOrEmpty($PYTHON_PATH) -or -not (Test-Path $PYTHON_PATH)) {
    # Find Python executable
    $python_cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $python_cmd) {
        $python_cmd = Get-Command python3 -ErrorAction SilentlyContinue
    }
    if ($null -eq $python_cmd) {
        $python_cmd = Get-Command py -ErrorAction SilentlyContinue
    }
    
    if ($null -ne $python_cmd) {
        $PYTHON_PATH = $python_cmd.Path
        Write-Host "Will use system default Python: $PYTHON_PATH"
    } else {
        Write-Host "Error: Python interpreter not found. Please install Python 3.7+ and try again."
        exit 1
    }
}

Write-Host "Using Python path: $PYTHON_PATH"

# Check Python version
$python_version = & $PYTHON_PATH -c "import sys; print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))"
Write-Host "Python version: $python_version"

# Ensure Python version >= 3.6
if ($python_version -notmatch "^3\.[6-9]|^3\.[1-9][0-9]") {
    Write-Host "Warning: Python 3.6 or higher is recommended. Current version: $python_version"
    $continue_anyway = Read-Host "Continue anyway? (y/n)"
    if ($continue_anyway -notmatch "^[Yy]") {
        Write-Host "Cancelled. Please install Python 3.6+ and try again."
        exit 1
    }
}

# Check if virtual environment exists
$venv_path = $null
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "Virtual environment detected."
    $venv_path = "venv\Scripts\Activate.ps1"
} elseif (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "Virtual environment detected."
    $venv_path = ".venv\Scripts\Activate.ps1"
}

if (-not [string]::IsNullOrEmpty($venv_path)) {
    Write-Host "Activating virtual environment..."
    . $venv_path
}

# Check dependencies
Write-Host "Checking dependencies..."
$has_streamlit = $false
try {
    & $PYTHON_PATH -c "import streamlit" 2>$null
    $has_streamlit = $LASTEXITCODE -eq 0
} catch {
    $has_streamlit = $false
}

if (-not $has_streamlit) {
    Write-Host "Required dependencies not installed. Installing now..."
    
    # Check if requirements.txt exists
    if (Test-Path "requirements.txt") {
        & $PYTHON_PATH -m pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Failed to install dependencies. Check the error messages and install manually."
            exit 1
        }
    } else {
        Write-Host "Warning: requirements.txt file not found. Installing core dependency..."
        & $PYTHON_PATH -m pip install streamlit
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Failed to install dependencies. Check the error messages and install manually."
            exit 1
        }
    }
    
    Write-Host "Dependencies installation complete."
} else {
    Write-Host "All required dependencies are installed."
}

# Run the app
Write-Host "Running Picture Sherlock..."
& $PYTHON_PATH -m streamlit run picture_sherlock/app.py -- --lang en $args 