# Pokerdice trainer - creates the Python environment .venv (Python 3.11) and installs everything.
# Run in the repository folder:
#   powershell -ExecutionPolicy Bypass -File setup.ps1
# Running it again only updates the packages; an existing .venv is kept.
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        Write-Host "The Python launcher 'py' was not found. Install Python 3.11 from https://www.python.org/downloads/" -ForegroundColor Red
        exit 1
    }
    & py -3.11 --version 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Python 3.11 is not installed (needed for esp-ppq). Install it from https://www.python.org/downloads/ -" -ForegroundColor Red
        Write-Host "it can live next to other versions. Installed versions: py -0" -ForegroundColor Red
        exit 1
    }
    Write-Host "Creating .venv with Python 3.11 ..."
    & py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { exit 1 }
} else {
    Write-Host ".venv exists already - updating the packages."
}

& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { exit 1 }
Write-Host "Installing torch (CPU version) ..."
& $venvPython -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { exit 1 }
Write-Host "Installing the trainer packages ..."
& $venvPython -m pip install -r trainer\requirements.txt
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host ""
Write-Host "Done. Start the trainer with start.bat (or: .venv\Scripts\python.exe trainer\assistant\app.py)" -ForegroundColor Green
