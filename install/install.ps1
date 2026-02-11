param(
    [string]$EnvName = "cpsam_gui",
    [string]$PythonVersion = "3.11",
    [ValidateSet("auto", "cpu", "gpu")] [string]$Gpu = "auto"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Test-NvidiaGpu {
    try {
        $controllers = Get-CimInstance -ClassName Win32_VideoController
        foreach ($controller in $controllers) {
            if ($controller.Caption -like "*NVIDIA*") {
                return $true
            }
        }
    }
    catch {
        Write-Warning "Could not query for video controllers: $_"
    }
    return $false
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..")).Path

$minicondaRoot = Join-Path $env:LOCALAPPDATA "Miniconda3"
$condaExe = Join-Path $minicondaRoot "Scripts\\conda.exe"
$minicondaInstaller = Join-Path $env:TEMP "Miniconda3-latest-Windows-x86_64.exe"

Write-Host "GUVpose installer (Windows)"
Write-Host "Repo: $repoRoot"
Write-Host "Env: $EnvName"
Write-Host ""

if (-not (Test-Path $condaExe)) {
    Write-Host "Miniconda not found. Downloading..."
    Invoke-WebRequest -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile $minicondaInstaller
    Write-Host "Installing Miniconda to $minicondaRoot..."
    & $minicondaInstaller /S /D=$minicondaRoot | Out-Null
}

if (-not (Test-Path $condaExe)) {
    throw "conda.exe not found after Miniconda install."
}

Write-Host "Accepting Anaconda Terms of Service for default channels..."
try {
    & $condaExe tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main | Out-Null
    & $condaExe tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r | Out-Null
    & $condaExe tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2 | Out-Null
} catch {
    Write-Warning "Failed to accept conda Terms of Service automatically. Please run 'conda tos accept' manually."
}

Write-Host "Ensuring conda environment $EnvName..."
$envList = & $condaExe env list
if ($envList -match "^\s*$EnvName\s") {
    Write-Host "Environment already exists. Skipping create."
} else {
    & $condaExe create -y -n $EnvName python=$PythonVersion
}

$envPath = Join-Path $minicondaRoot "envs\\$EnvName"
if (-not (Test-Path $envPath)) {
    throw "Environment $EnvName was not created. Please re-run installer after accepting conda TOS."
}

$useGpu = $false
if ($Gpu -eq "gpu") {
    $useGpu = $true
} elseif ($Gpu -eq "auto") {
    $useGpu = Test-NvidiaGpu
}

if ($useGpu) {
    Write-Host "NVIDIA GPU selected. Installing PyTorch CUDA build..."
    $pytorchIndex = "https://download.pytorch.org/whl/cu129"
    & $condaExe run -n $EnvName pip install torch==2.8.0+cu129 torchvision==0.23.0+cu129 --index-url $pytorchIndex
} else {
    Write-Host "Installing PyTorch CPU-only build..."
    & $condaExe run -n $EnvName pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
}

Write-Host "Pinning numpy to match reference environment..."
& $condaExe run -n $EnvName pip install numpy==2.3.4

Write-Host "Installing GUI dependencies..."
& $condaExe run -n $EnvName pip install numpy scipy scikit-image pyqt5 pyqtgraph superqt natsort tifffile fastremap tqdm matplotlib

Write-Host "Installing gRPC and optional image readers..."
& $condaExe run -n $EnvName pip install grpcio grpcio-tools protobuf nd2 readlif

Write-Host "Installing GUVpose project..."
& $condaExe run -n $EnvName pip install -e $repoRoot

Write-Host "Re-pinning numpy after project install..."
& $condaExe run -n $EnvName pip install numpy==2.3.4

Write-Host ""
Write-Host "Done."
Write-Host "Launch: conda run -n $EnvName python -m cellpose"
