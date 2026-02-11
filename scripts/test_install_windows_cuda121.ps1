param(
    [string]$EnvName = "multicellpose-pypi-cu121",
    [string]$PythonVersion = "3.10",
    [string]$PackageSpec = "."
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message"
}

Write-Step "Checking conda"
conda --version | Out-Null

Write-Step "Accepting conda ToS channels (safe to re-run)"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main | Out-Null
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r | Out-Null
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2 | Out-Null

Write-Step "Creating fresh env: $EnvName (Python $PythonVersion)"
conda env remove -n $EnvName -y 2>$null | Out-Null
conda create -n $EnvName python=$PythonVersion -y | Out-Null

Write-Step "Upgrading pip tooling"
conda run -n $EnvName python -m pip install --upgrade pip setuptools wheel | Out-Null

Write-Step "Installing PyTorch CUDA 12.1 wheels"
conda run -n $EnvName python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 | Out-Null

Write-Step "Installing MultiCellPose package spec: $PackageSpec"
if ($PackageSpec -eq ".") {
    conda run -n $EnvName python -m pip install ".[all]"
} else {
    conda run -n $EnvName python -m pip install $PackageSpec
}

Write-Step "Verifying CUDA visibility in torch"
conda run -n $EnvName python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_version', torch.version.cuda)"

Write-Step "Checking entry points"
conda run -n $EnvName python -m train_headless --help | Out-Null
conda run -n $EnvName python -c "import run_server, guv_app.main; print('imports-ok')"

Write-Step "Done"
Write-Host "Environment ready: $EnvName"
Write-Host "Launch GUI with: conda run -n $EnvName python -m guv_app.main"
