#!/usr/bin/env python
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys


def _run(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except Exception as exc:
        return 1, "", str(exc)


def _which(cmd):
    return shutil.which(cmd) is not None


def detect_conda():
    conda_exe = os.environ.get("CONDA_EXE", "")
    if conda_exe and os.path.isfile(conda_exe):
        return True, conda_exe
    return _which("conda"), shutil.which("conda") or ""


def detect_nvidia():
    if _which("nvidia-smi"):
        code, out, _ = _run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
        )
        if code == 0 and out:
            first = out.splitlines()[0]
            parts = [p.strip() for p in first.split(",")]
            name = parts[0] if parts else "NVIDIA GPU"
            driver = parts[1] if len(parts) > 1 else ""
            return True, name, driver
    if sys.platform == "win32" and _which("wmic"):
        code, out, _ = _run(["wmic", "path", "win32_VideoController", "get", "name"])
        if code == 0 and out:
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            for line in lines[1:]:
                if "NVIDIA" in line.upper():
                    return True, line, ""
    return False, "", ""


def detect_disk():
    try:
        usage = shutil.disk_usage(os.getcwd())
        return {"total_gb": int(usage.total / 1e9), "free_gb": int(usage.free / 1e9)}
    except Exception:
        return {"total_gb": None, "free_gb": None}


def main():
    parser = argparse.ArgumentParser(description="GUVpose preflight check")
    parser.add_argument("--json", action="store_true", help="output JSON only")
    args = parser.parse_args()

    os_name = platform.system()
    os_release = platform.release()
    python_ver = sys.version.split()[0]

    conda_ok, conda_path = detect_conda()
    nvidia_ok, nvidia_name, nvidia_driver = detect_nvidia()
    disk = detect_disk()

    recommendation = "gpu" if nvidia_ok else "cpu"

    report = {
        "os": os_name,
        "os_release": os_release,
        "python": python_ver,
        "conda_found": conda_ok,
        "conda_path": conda_path,
        "nvidia_gpu": nvidia_ok,
        "nvidia_name": nvidia_name,
        "nvidia_driver": nvidia_driver,
        "disk": disk,
        "recommendation": recommendation,
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print("GUVpose Preflight Report")
    print(f"- OS: {os_name} {os_release}")
    print(f"- Python: {python_ver}")
    print(f"- Conda found: {conda_ok} {conda_path}")
    if nvidia_ok:
        print(f"- NVIDIA GPU: {nvidia_name} (driver {nvidia_driver})")
    else:
        print("- NVIDIA GPU: not detected")
    if disk["free_gb"] is not None:
        print(f"- Disk free: {disk['free_gb']} GB")
    print(f"- Recommendation: {recommendation.upper()} install")
    print("")
    if os_name == "Windows":
        print("Suggested command:")
        if recommendation == "gpu":
            print("  powershell -ExecutionPolicy Bypass -File install\\install.ps1 -Gpu gpu")
        else:
            print("  powershell -ExecutionPolicy Bypass -File install\\install.ps1 -Gpu cpu")
    else:
        print("Suggested command:")
        if recommendation == "gpu":
            print("  bash install/install.sh --gpu")
        else:
            print("  bash install/install.sh --cpu")


if __name__ == "__main__":
    main()
