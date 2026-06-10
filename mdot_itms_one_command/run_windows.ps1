$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
if (!(Test-Path ".venv")) {
    py -3 -m venv .venv
}
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m playwright install chromium
python run.py @args
