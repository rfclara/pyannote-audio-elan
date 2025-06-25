# Set locale and encoding
$env:LC_ALL = "en_US.UTF-8"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTORCH_ENABLE_MPS_FALLBACK = "1"

. .\venv-pyannote-audio-elan\Scripts\Activate.ps1
Set-Location -Path $PSScriptRoot

python .\pyannote-audio-elan.py @args > .\elan_wrapper_debug.log 2>&1