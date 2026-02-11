# PyPI Release Checklist (MultiCellPose / multicellpose)

## 1) Pre-flight

- Ensure working tree is clean:
  - `git status`
- Confirm package metadata:
  - `pyproject.toml` project name/version metadata
  - entry points: `multicellpose-gui`, `multicellpose-server`, `multicellpose-train`
- Confirm no local secrets/config are tracked:
  - `remote_config.json` must remain ignored
  - no credentials in source/docs

## 2) Local build and validation

- Create/activate clean environment:
  - `conda create -n mcp_pkg_test python=3.10 -y`
  - `conda activate mcp_pkg_test`
- Install build tools:
  - `python -m pip install --upgrade pip build twine`
- Build package:
  - `python -m build`
- Validate package metadata:
  - `python -m twine check dist/*`

## 3) Wheel install smoke-test

- Create second clean environment:
  - `conda create -n mcp_wheel_test python=3.10 -y`
  - `conda activate mcp_wheel_test`
- Install CUDA 12.1 PyTorch wheels first:
  - `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
- Install built wheel:
  - `pip install dist/*.whl`
- Verify commands are installed:
  - `multicellpose-gui --help`
  - `multicellpose-server --help`
  - `multicellpose-train --help`

## 4) Trusted publishing setup

- Create project on PyPI (`multicellpose`).
- Add Trusted Publisher in PyPI project settings:
  - GitHub owner/repo
  - workflow: `.github/workflows/publish-pypi.yml`
  - environment: `pypi`
- Create GitHub environment `pypi`.

## 5) Publish

- Tag/release:
  - `git tag vX.Y.Z`
  - `git push origin vX.Y.Z`
- Create GitHub Release for the tag (or use workflow dispatch).
- Confirm successful upload on PyPI.

## 6) Post-release verification

- In a fresh environment:
  - install CUDA 12.1 torch
  - `pip install multicellpose`
  - run `multicellpose-gui --help`, `multicellpose-server --help`, `multicellpose-train --help`
- Update release notes with known platform notes.
