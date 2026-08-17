from pathlib import Path

import numpy as np
import pytest
import guv_app.services.class_lock_tracking_service as class_lock_service

from guv_app.services.class_lock_tracking_service import (
    TrackFrame,
    apply_class_override,
    apply_multi_class_override,
    save_and_verify_class_overrides,
    save_and_verify_multi_class_overrides,
    track_many_from_anchors,
    track_from_anchor,
)


def _frame(index, objects, classes):
    masks = np.zeros((32, 32), dtype=np.int32)
    for mask_id, y, x in objects:
        masks[y : y + 3, x : x + 3] = mask_id
    class_array = np.zeros(max([0, *classes]) + 1, dtype=np.int16)
    for mask_id, class_id in classes.items():
        if mask_id >= len(class_array):
            class_array = np.pad(class_array, (0, mask_id - len(class_array) + 1))
        class_array[mask_id] = class_id
    payload = {"masks": masks, "classes": class_array}
    return TrackFrame(f"T{index}", Path(f"T{index}_pred.npy"), masks, class_array, payload)


def test_track_from_anchor_ignores_alternating_classes_and_runs_both_directions():
    frames = [
        _frame(0, [(4, 8, 8), (7, 22, 22)], {4: 1, 7: 2}),
        _frame(1, [(2, 9, 10), (8, 21, 22)], {2: 2, 8: 2}),
        _frame(2, [(9, 10, 12), (3, 20, 22)], {9: 1, 3: 2}),
    ]

    members = track_from_anchor(frames, anchor_index=1, anchor_mask_id=2, max_displacement=8)

    assert [(member.frame_id, member.mask_id) for member in members] == [
        ("T0", 4),
        ("T1", 2),
        ("T2", 9),
    ]


def test_track_stops_when_no_geometrically_plausible_continuation_exists():
    frames = [
        _frame(0, [(1, 5, 5)], {1: 1}),
        _frame(1, [(1, 6, 6)], {1: 2}),
        _frame(2, [(1, 25, 25)], {1: 1}),
    ]

    members = track_from_anchor(frames, anchor_index=0, anchor_mask_id=1, max_displacement=5)

    assert [member.frame_id for member in members] == ["T0", "T1"]


def test_apply_class_override_updates_classes_and_preserves_audit_record():
    frames = [
        _frame(0, [(1, 5, 5)], {1: 1}),
        _frame(1, [(3, 6, 6)], {3: 2}),
    ]
    members = track_from_anchor(frames, anchor_index=0, anchor_mask_id=1, max_displacement=5)

    payloads = apply_class_override(frames, members, target_class=2, override_id="test-lock")

    assert int(payloads[0]["classes"][1]) == 2
    assert int(payloads[1]["classes"][3]) == 2
    assert payloads[0]["class_track_overrides"][0]["previous_class"] == 1
    assert payloads[0]["class_track_overrides"][0]["class_agnostic_tracking"] is True


def test_apply_class_override_updates_active_channel_class_vector():
    frame = _frame(0, [(2, 5, 5)], {2: 1})
    state_classes = np.array([0, 0, 1], dtype=np.int16)
    payload = dict(frame.payload)
    payload["active_channel_index"] = 0
    payload["channel_segmentations"] = {
        "0": {"masks": frame.masks[None, ...], "mask_classes": state_classes}
    }
    frame = TrackFrame(frame.frame_id, frame.path, frame.masks, frame.classes, payload)
    members = track_from_anchor([frame], 0, 2, max_displacement=5)

    updated = apply_class_override([frame], members, 2, "test-lock")[0]

    assert int(updated["channel_segmentations"]["0"]["mask_classes"][2]) == 2


def test_save_and_verify_persists_corrected_classes(tmp_path):
    frames = [
        _frame(0, [(1, 5, 5)], {1: 1}),
        _frame(1, [(3, 6, 6)], {3: 2}),
    ]
    disk_frames = []
    for index, frame in enumerate(frames):
        path = tmp_path / f"T{index}_pred.npy"
        with path.open("wb") as handle:
            np.save(handle, frame.payload, allow_pickle=True)
        disk_frames.append(TrackFrame(frame.frame_id, path, frame.masks, frame.classes, frame.payload))
    members = track_from_anchor(disk_frames, 0, 1, max_displacement=5)
    payloads = apply_class_override(disk_frames, members, 2, "persisted-lock")

    verified = save_and_verify_class_overrides(disk_frames, payloads, members, 2)

    assert verified == 2
    for frame, member in zip(disk_frames, members):
        saved = np.load(frame.path, allow_pickle=True).item()
        assert int(saved["classes"][member.mask_id]) == 2
        assert saved["class_track_overrides"][-1]["override_id"] == "persisted-lock"


def test_multiple_tracks_end_independently_and_report_shared_mask_conflicts():
    frames = [
        _frame(0, [(1, 4, 4), (2, 20, 20)], {1: 1, 2: 2}),
        _frame(1, [(3, 5, 5), (4, 19, 19)], {3: 2, 4: 1}),
        _frame(2, [(5, 6, 6)], {5: 2}),
    ]
    tracks, conflicts = track_many_from_anchors(frames, 0, [1, 2], 5)

    assert [m.mask_id for m in tracks[1]] == [1, 3, 5]
    assert [m.mask_id for m in tracks[2]] == [2, 4]
    assert conflicts == []

    converged = [
        _frame(0, [(1, 4, 4), (2, 8, 8)], {1: 1, 2: 2}),
        _frame(1, [(3, 6, 6)], {3: 1}),
    ]
    _tracks, conflicts = track_many_from_anchors(converged, 0, [1, 2], 8)
    assert conflicts[0]["mask_id"] == 3
    assert conflicts[0]["anchor_mask_ids"] == [1, 2]


def test_multi_track_apply_saves_all_objects_in_same_frame_once(tmp_path):
    frames = [
        _frame(0, [(1, 4, 4), (2, 20, 20)], {1: 2, 2: 2}),
        _frame(1, [(3, 5, 5), (4, 19, 19)], {3: 2, 4: 2}),
    ]
    disk_frames = []
    for index, frame in enumerate(frames):
        path = tmp_path / f"multi_T{index}_pred.npy"
        with path.open("wb") as handle:
            np.save(handle, frame.payload, allow_pickle=True)
        disk_frames.append(TrackFrame(frame.frame_id, path, frame.masks, frame.classes, frame.payload))
    tracks, _conflicts = track_many_from_anchors(disk_frames, 0, [1, 2], 5)
    payloads = apply_multi_class_override(disk_frames, tracks, 1, "multi-lock")

    verified = save_and_verify_multi_class_overrides(disk_frames, payloads, tracks, 1)

    assert verified == 4
    for frame in disk_frames:
        saved = np.load(frame.path, allow_pickle=True).item()
        assert set(np.unique(saved["classes"])) <= {0, 1}
        assert len(saved["class_track_overrides"]) == 2


def test_multi_track_save_rolls_back_if_replacement_fails(tmp_path, monkeypatch):
    frames = [
        _frame(0, [(1, 4, 4)], {1: 2}),
        _frame(1, [(2, 5, 5)], {2: 2}),
    ]
    disk_frames = []
    for index, frame in enumerate(frames):
        path = tmp_path / f"rollback_T{index}_pred.npy"
        with path.open("wb") as handle:
            np.save(handle, frame.payload, allow_pickle=True)
        disk_frames.append(TrackFrame(frame.frame_id, path, frame.masks, frame.classes, frame.payload))
    tracks, _conflicts = track_many_from_anchors(disk_frames, 0, [1], 5)
    payloads = apply_multi_class_override(disk_frames, tracks, 1, "rollback-lock")
    real_replace = class_lock_service.os.replace
    calls = {"writes": 0}

    def fail_second_normal_replace(source, destination):
        if str(source).endswith(".classlock.tmp"):
            calls["writes"] += 1
            if calls["writes"] == 2:
                raise OSError("simulated replacement failure")
        return real_replace(source, destination)

    monkeypatch.setattr(class_lock_service.os, "replace", fail_second_normal_replace)
    with pytest.raises(OSError, match="simulated replacement failure"):
        save_and_verify_multi_class_overrides(disk_frames, payloads, tracks, 1)

    for frame in disk_frames:
        saved = np.load(frame.path, allow_pickle=True).item()
        assert np.array_equal(saved["classes"], frame.classes)
