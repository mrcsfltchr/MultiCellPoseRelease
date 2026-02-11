param(
    [string]$EnvName = "cpsam_gui"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..")).Path

$minicondaRoot = Join-Path $env:LOCALAPPDATA "Miniconda3"
$condaExe = Join-Path $minicondaRoot "Scripts\\conda.exe"

Write-Host "GUVpose repair (Windows)"
Write-Host "Repo: $repoRoot"
Write-Host "Env: $EnvName"
Write-Host ""

if (-not (Test-Path $condaExe)) {
    throw "conda.exe not found. Install Miniconda first."
}

$envPath = Join-Path $minicondaRoot "envs\\$EnvName"
if (-not (Test-Path $envPath)) {
    throw "Environment $EnvName not found. Run install.ps1 first."
}

Write-Host "Re-installing dependencies..."
& $condaExe run -n $EnvName pip install numpy==2.3.4
& $condaExe run -n $EnvName pip install numpy scipy scikit-image pyqt5 pyqtgraph superqt natsort tifffile fastremap tqdm matplotlib
& $condaExe run -n $EnvName pip install grpcio grpcio-tools protobuf nd2 readlif

Write-Host "Re-installing GUVpose project..."
& $condaExe run -n $EnvName pip install -e $repoRoot

Write-Host "Done."
