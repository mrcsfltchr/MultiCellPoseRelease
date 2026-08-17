"""Workers for previewing and applying class-agnostic class-lock tracks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import pyqtSignal

from guv_app.services.class_lock_tracking_service import (
    TrackFrame,
    apply_multi_class_override,
    save_and_verify_multi_class_overrides,
    track_many_from_anchors,
)
from guv_app.workers.base_worker import BaseWorker


def _load_frames(frame_specs):
    frames = []
    for frame_id, path_text in frame_specs:
        path = Path(path_text)
        if not path.exists():
            raise FileNotFoundError(f"Missing prediction for {frame_id or 'single frame'}: {path}")
        payload = np.load(path, allow_pickle=True).item()
        masks = payload.get("masks")
        classes = payload.get("classes")
        if masks is None or classes is None:
            raise ValueError(f"Prediction requires semantic masks and classes: {path}")
        frames.append(TrackFrame(frame_id, path, np.asarray(masks), np.asarray(classes), payload))
    return frames


class ClassLockPreviewWorker(BaseWorker):
    preview_ready = pyqtSignal(object)
    progress = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, frame_specs, anchor_frame_id, anchor_mask_ids, target_class, max_displacement):
        super().__init__()
        self.frame_specs = list(frame_specs)
        self.anchor_frame_id = anchor_frame_id
        self.anchor_mask_ids = [int(value) for value in anchor_mask_ids]
        self.target_class = int(target_class)
        self.max_displacement = float(max_displacement)

    def run(self):
        try:
            self.progress.emit("Loading series predictions for class-lock tracking…")
            print("Class lock: loading series predictions", flush=True)
            frames = _load_frames(self.frame_specs)
            anchor_index = next(
                index for index, frame in enumerate(frames) if frame.frame_id == self.anchor_frame_id
            )
            tracks, conflicts = track_many_from_anchors(
                frames, anchor_index, self.anchor_mask_ids, self.max_displacement
            )
            result = {
                "frames": frames,
                "tracks": tracks,
                "conflicts": conflicts,
                "anchor_index": anchor_index,
                "anchor_mask_ids": self.anchor_mask_ids,
                "target_class": self.target_class,
                "max_displacement": self.max_displacement,
            }
            print(
                "Class lock: counterfactual tracks found "
                + ", ".join(
                    f"mask {anchor}: {len(members)}/{len(frames)} frames"
                    for anchor, members in tracks.items()
                ),
                flush=True,
            )
            self.preview_ready.emit(result)
        except StopIteration:
            print("Class lock error: current frame was not found in the loaded series", flush=True)
            self.error.emit("Current frame was not found in the loaded series.")
        except Exception as exc:
            print(f"Class lock preview error: {exc}", flush=True)
            self.error.emit(str(exc))
        finally:
            self.finished.emit()


class ClassLockApplyWorker(BaseWorker):
    applied = pyqtSignal(object)
    progress = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, preview, override_id):
        super().__init__()
        self.preview = preview
        self.override_id = str(override_id)

    def run(self):
        try:
            frames = self.preview["frames"]
            tracks = self.preview["tracks"]
            target_class = int(self.preview["target_class"])
            print("Class lock: apply worker started", flush=True)
            self.progress.emit("Saving and verifying class-lock overrides…")
            payloads = apply_multi_class_override(
                frames, tracks, target_class, self.override_id
            )
            verified = save_and_verify_multi_class_overrides(
                frames, payloads, tracks, target_class
            )
            result = dict(self.preview)
            result.update(override_id=self.override_id, verified_frames=verified)
            print(
                f"Class lock applied: class={target_class}, objects={verified}, "
                f"tracks={len(tracks)}, id={self.override_id}",
                flush=True,
            )
            self.applied.emit(result)
        except Exception as exc:
            print(f"Class lock apply error: {exc}", flush=True)
            self.error.emit(str(exc))
        finally:
            self.finished.emit()
