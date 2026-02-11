param(
  [Parameter(Mandatory = $true)][string]$InstallDir,
  [Parameter(Mandatory = $true)][string]$ServerAddress,
  [Parameter(Mandatory = $true)][string]$ServerPort,
  [Parameter(Mandatory = $true)][string]$CondaPath,
  [Parameter(Mandatory = $true)][string]$InstallMode,
  [Parameter(Mandatory = $true)][string]$LogPath,
  [Parameter(Mandatory = $true)][string]$StatusPath
)

$ErrorActionPreference = "Stop"

function Write-Log {
  param([string]$Message)
  $line = "[SETUP] $Message"
  $line | Tee-Object -FilePath $LogPath -Append | Out-Host
}

function Run-Conda {
  param([string[]]$CondaArgs)
  $cmd = "$CondaExe $($CondaArgs -join ' ')"
  Write-Log "RUN: $cmd"
  $outputText = ""
  & $CondaExe @CondaArgs 2>&1 | ForEach-Object {
    $line = $_
    $line | Tee-Object -FilePath $LogPath -Append | Out-Host
    $outputText += "$line`n"
  }
  if ($LASTEXITCODE -ne 0) {
    Write-Log "ERROR: conda exit code $LASTEXITCODE"
    if (-not $outputText.Trim()) {
      Write-Log "ERROR: conda produced no output."
    }
    if ($outputText -match "Failed to remove contents in a temporary directory") {
      Write-Log "WARN: pip temp cleanup failed; continuing."
      return
    }
    throw "Command failed with exit code $LASTEXITCODE"
  }
}

function Get-DetectedCudaVersion {
  try {
    $nvidiaSmi = $null
    $command = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($command) {
      $nvidiaSmi = $command.Source
    } else {
      $candidates = @(
        (Join-Path $env:SystemRoot "System32\nvidia-smi.exe"),
        (Join-Path $env:ProgramFiles "NVIDIA Corporation\NVSMI\nvidia-smi.exe"),
        (Join-Path $env:ProgramW6432 "NVIDIA Corporation\NVSMI\nvidia-smi.exe")
      )
      foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) {
          $nvidiaSmi = $candidate
          break
        }
      }
    }
    if (-not $nvidiaSmi) {
      return $null
    }
    $raw = (& $nvidiaSmi --query-gpu=cuda_version --format=csv,noheader 2>$null).Trim()
    if (-not $raw) {
      return $null
    }
    $parts = $raw.Split(".")
    if ($parts.Length -lt 2) {
      return $null
    }
    return "$($parts[0]).$($parts[1])"
  } catch {
    return $null
  }
}

function Get-DetectedGpuName {
  try {
    $gpu = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "NVIDIA" } | Select-Object -First 1 -ExpandProperty Name
    if ($gpu) {
      return $gpu
    }
    return $null
  } catch {
    return $null
  }
}

function Select-PytorchCudaTag {
  param([string]$CudaVersion)
  if (-not $CudaVersion) {
    return $null
  }
  $major = [int]($CudaVersion.Split(".")[0])
  $minor = [int]($CudaVersion.Split(".")[1])
  $value = ($major * 10) + $minor
  if ($value -ge 126) { return "cu126" }
  if ($value -ge 121) { return "cu121" }
  if ($value -ge 118) { return "cu118" }
  return $null
}

function Get-InstalledTorchVariant {
  param([string]$EnvName)
  try {
    $out = & $CondaExe run -n $EnvName python -c "import torch; print(torch.__version__)" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $out) {
      return $null
    }
    if ($out -match "\+cu") {
      return "cuda"
    }
    if ($out -match "\+cpu") {
      return "cpu"
    }
    return "unknown"
  } catch {
    return $null
  }
}

function Env-Exists {
  param([string]$EnvName)
  $envList = & $CondaExe env list 2>&1
  $envList | Tee-Object -FilePath $LogPath -Append | Out-Host
  return ($envList -match "^\s*$EnvName\s")
}

function Ensure-Shortcut {
  param([string]$TargetPath, [string]$WorkingDir)
  $startMenu = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\MultiCellPose.lnk"
  $shell = New-Object -ComObject WScript.Shell
  $shortcut = $shell.CreateShortcut($startMenu)
  $shortcut.TargetPath = "$env:ComSpec"
  $shortcut.Arguments = "/k `"$TargetPath`""
  $shortcut.WindowStyle = 1
  $shortcut.Description = "Launch the MultiCellPose application"
  $shortcut.WorkingDirectory = $WorkingDir
  $shortcut.Save()
}

try {
  Write-Log "setup_environment.ps1 starting."
  "RUNNING" | Out-File -FilePath $StatusPath -Encoding ASCII
  Write-Log "CondaPath=$CondaPath"
  Write-Log "InstallMode=$InstallMode"

  $CondaExe = $CondaPath
  if (-not (Test-Path $CondaExe)) {
    throw "conda.exe not found at $CondaExe"
  }

  $CondaBase = Split-Path (Split-Path $CondaExe -Parent) -Parent
  Write-Log "Conda base: $CondaBase"
  Write-Log "Conda exe: $CondaExe"

  Write-Log "Checking Conda ToS status..."
  $oldEA = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  $oldNativePref = $null
  if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -Scope Global -ErrorAction SilentlyContinue) {
    $oldNativePref = $global:PSNativeCommandUseErrorActionPreference
    $global:PSNativeCommandUseErrorActionPreference = $false
  }
  try {
    $tosOutput = & $CondaExe tos 2>&1
    $tosExit = $LASTEXITCODE
  } catch {
    $tosOutput = @($_.Exception.Message)
    $tosExit = 1
  }
  if ($oldNativePref -ne $null) {
    $global:PSNativeCommandUseErrorActionPreference = $oldNativePref
  }
  $ErrorActionPreference = $oldEA
  $tosOutput | Tee-Object -FilePath $LogPath -Append | Out-Host
  if ($tosExit -eq 0) {
    Write-Log "Attempting to accept Conda ToS."
    try {
      Run-Conda @("tos","accept","--override-channels","--channel","https://repo.anaconda.com/pkgs/main")
      Run-Conda @("tos","accept","--override-channels","--channel","https://repo.anaconda.com/pkgs/r")
      Run-Conda @("tos","accept","--override-channels","--channel","https://repo.anaconda.com/pkgs/msys2")
    } catch {
      Write-Log "WARN: Conda ToS acceptance failed. Continuing."
    }
  } else {
    Write-Log "Conda ToS command unavailable or failed. Continuing."
  }

  $envName = "multicellpose"
  $envPath = Join-Path (Join-Path $CondaBase "envs") $envName
  $pytorchIndex = "https://download.pytorch.org/whl/cpu"
  if ($InstallMode -eq "cpu") {
    Write-Log "CPU-only install selected."
  } elseif ($InstallMode -eq "gpu") {
    Write-Log "GPU install selected."
  } else {
    Write-Log "Auto install selected."
  }

  if ($InstallMode -ne "cpu") {
    $detectedCuda = Get-DetectedCudaVersion
    $detectedGpu = Get-DetectedGpuName
    if ($detectedCuda) {
      Write-Log "Detected CUDA capability from driver: $detectedCuda"
      $cudaTag = Select-PytorchCudaTag -CudaVersion $detectedCuda
      if ($cudaTag) {
        $pytorchIndex = "https://download.pytorch.org/whl/$cudaTag"
        Write-Log "Selected PyTorch CUDA build: $cudaTag"
      } else {
        Write-Log "WARN: Unsupported CUDA version for PyTorch mapping. Falling back to CPU."
        $pytorchIndex = "https://download.pytorch.org/whl/cpu"
      }
    } elseif ($detectedGpu) {
      Write-Log "Detected NVIDIA GPU without CUDA version. Using default CUDA build."
      $pytorchIndex = "https://download.pytorch.org/whl/cu121"
    } elseif ($InstallMode -eq "gpu") {
      Write-Log "WARN: GPU selected but no CUDA version detected. Falling back to CPU."
      $pytorchIndex = "https://download.pytorch.org/whl/cpu"
    }
  }

  Write-Log "Creating conda environment..."
  if (Env-Exists -EnvName $envName) {
    Write-Log "Environment already exists. Skipping create."
  } else {
    if (Test-Path $envPath) {
      Write-Log "Stale environment folder detected at $envPath. Removing."
      try {
        Remove-Item -Recurse -Force $envPath
      } catch {
        Write-Log "WARN: Failed to remove stale env folder: $($_.Exception.Message)"
      }
    }
    Run-Conda @("create","-n",$envName,"python=3.10","-y")
  }

  Write-Log "Installing PyTorch from index $pytorchIndex"
  if ($pytorchIndex) {
    if ($pytorchIndex -notmatch "/cpu$") {
      $torchVariant = Get-InstalledTorchVariant -EnvName $envName
      if ($torchVariant -eq "cpu") {
        Write-Log "CPU-only PyTorch detected. Removing before CUDA install."
        Run-Conda @("run","-n",$envName,"pip","uninstall","-y","torch","torchvision","torchaudio")
      }
    }
    Write-Log "Installing PyTorch from index $pytorchIndex"
    Run-Conda @("run","-n",$envName,"python","-m","pip","install","torch","torchvision","torchaudio","--index-url",$pytorchIndex)
  }

  Write-Log "Installing pip requirements..."
  Run-Conda @("run","-n",$envName,"python","-m","pip","install","-r", (Join-Path $InstallDir "requirements.txt"))

  Write-Log "Installing application package..."
  Run-Conda @("run","-n",$envName,"python","-m","pip","install","-U","pip","setuptools","wheel")
  try {
    Run-Conda @("run","-n",$envName,"python","-m","pip","install","-e","$InstallDir[all]")
  } catch {
    Write-Log "WARN: Editable install failed, retrying without build isolation."
    Run-Conda @("run","-n",$envName,"python","-m","pip","install","$InstallDir[all]","--no-build-isolation")
  }

  Write-Log "Creating remote_config.json..."
  $config = @{
    server_address = $ServerAddress
    server_port = [int]$ServerPort
  } | ConvertTo-Json -Depth 2
  $config | Out-File -FilePath (Join-Path $InstallDir "remote_config.json") -Encoding UTF8

  Write-Log "Creating start_multicellpose.bat..."
  $batchPath = Join-Path $InstallDir "start_multicellpose.bat"
  $batchContent = "@echo off`r`n" +
    "echo Starting MultiCellPose...`r`n" +
    "`"$CondaExe`" run -n $envName python -m guv_app.main`r`n" +
    "if %errorlevel% neq 0 (`r`n" +
    "  echo Failed to start MultiCellPose. Check your environment and try again.`r`n" +
    ")`r`n" +
    "pause`r`n"
  $batchContent | Out-File -FilePath $batchPath -Encoding ASCII

  Write-Log "Creating Start Menu shortcut..."
  Ensure-Shortcut -TargetPath $batchPath -WorkingDir $InstallDir

  Write-Log "setup_environment.ps1 complete."
  "OK" | Out-File -FilePath $StatusPath -Encoding ASCII
  exit 0
} catch {
  Write-Log "ERROR: $($_.Exception.Message)"
  Write-Log "Stack: $($_.ScriptStackTrace)"
  "ERROR" | Out-File -FilePath $StatusPath -Encoding ASCII
  exit 1
}
