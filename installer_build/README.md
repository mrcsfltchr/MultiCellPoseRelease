# Installer build (NSIS)

This folder contains the assets for the advanced NSIS-based installer that
drives a Python setup script.

## What to add before compiling

1) Download the 64-bit Miniconda installer and rename it to:
   - `installer_build/Miniconda-Installer.exe`

2) Confirm `installer_build/requirements.txt` matches your target dependencies.

3) If you want a custom install location, update it in `installer.nsi`.

## Build steps

1) Copy `installer_build/installer.nsi` to the repo root (optional).
2) In Explorer, right-click the `.nsi` file and select **Compile NSIS Script**.
3) The output is `MultiCellPose_Installer.exe`.

## Notes

- The NSIS script calls `setup_environment.py` during install to perform:
  conda setup, GPU detection, env creation, dependency install, and config file creation.
- If you update the install logic, modify `setup_environment.py`.
