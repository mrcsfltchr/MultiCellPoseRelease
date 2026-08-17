"""Class-agnostic seeded tracking used to propagate a manual class correction."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class TrackFrame:
    frame_id: str | None
    path: Path
    masks: np.ndarray
    classes: np.ndarray
    payload: dict


@dataclass(frozen=True)
class TrackMember:
    frame_index: int
    frame_id: str | None
    mask_id: int
    score: float


def _objects(masks: np.ndarray) -> dict[int, tuple[np.ndarray, float]]:
    arr = np.squeeze(np.asarray(masks))
    result = {}
    for mask_id in np.unique(arr[arr > 0]).astype(np.int64):
        yy, xx = np.nonzero(arr == mask_id)
        if yy.size:
            result[int(mask_id)] = (np.array([yy.mean(), xx.mean()]), float(yy.size))
    return result


def _best_geometry_match(
    source: tuple[np.ndarray, float],
    candidates: dict[int, tuple[np.ndarray, float]],
    max_displacement: float,
) -> tuple[int, float] | None:
    source_centroid, source_area = source
    plausible = []
    for mask_id, (centroid, area) in candidates.items():
        distance = float(np.linalg.norm(centroid - source_centroid))
        if distance > max_displacement or source_area <= 0 or area <= 0:
            continue
        area_ratio = area / source_area
        if not 0.25 <= area_ratio <= 4.0:
            continue
        cost = distance / max(max_displacement, 1.0) + 0.35 * abs(math.log(area_ratio))
        if cost > 1.0:
            continue
        score = max(0.0, 1.0 - cost)
        plausible.append((mask_id, score, cost))
    if not plausible:
        return None
    plausible.sort(key=lambda item: item[2])
    mask_id, score, best_cost = plausible[0]
    if len(plausible) > 1:
        # A close alternative is an ambiguity signal even if the best geometry
        # is individually plausible. Surface it through a lower match score.
        margin = plausible[1][2] - best_cost
        score *= min(1.0, max(0.0, margin / 0.25))
    return int(mask_id), float(score)


def track_from_anchor(
    frames: Sequence[TrackFrame],
    anchor_index: int,
    anchor_mask_id: int,
    max_displacement: float,
) -> list[TrackMember]:
    """Track one mask bidirectionally using geometry only; classes are ignored."""
    if not 0 <= anchor_index < len(frames):
        raise IndexError("anchor_index is outside the frame sequence")
    objects = [_objects(frame.masks) for frame in frames]
    if int(anchor_mask_id) not in objects[anchor_index]:
        raise ValueError(f"anchor mask {anchor_mask_id} is absent from the anchor frame")

    members = {
        anchor_index: TrackMember(anchor_index, frames[anchor_index].frame_id, int(anchor_mask_id), 1.0)
    }
    for direction in (-1, 1):
        current_index = anchor_index
        current_id = int(anchor_mask_id)
        while 0 <= current_index + direction < len(frames):
            next_index = current_index + direction
            match = _best_geometry_match(
                objects[current_index][current_id], objects[next_index], float(max_displacement)
            )
            if match is None:
                break
            current_id, score = match
            members[next_index] = TrackMember(
                next_index, frames[next_index].frame_id, current_id, score
            )
            current_index = next_index
    return [members[index] for index in sorted(members)]


def track_many_from_anchors(
    frames: Sequence[TrackFrame],
    anchor_index: int,
    anchor_mask_ids: Sequence[int],
    max_displacement: float,
) -> tuple[dict[int, list[TrackMember]], list[dict]]:
    """Track anchors independently and report masks claimed by multiple tracks."""
    tracks = {
        int(anchor_id): track_from_anchor(
            frames, anchor_index, int(anchor_id), max_displacement
        )
        for anchor_id in dict.fromkeys(int(value) for value in anchor_mask_ids)
    }
    claims: dict[tuple[int, int], list[int]] = {}
    for anchor_id, members in tracks.items():
        for member in members:
            claims.setdefault((member.frame_index, member.mask_id), []).append(anchor_id)
    conflicts = [
        {
            "frame_index": frame_index,
            "frame_id": frames[frame_index].frame_id,
            "mask_id": mask_id,
            "anchor_mask_ids": anchor_ids,
        }
        for (frame_index, mask_id), anchor_ids in sorted(claims.items())
        if len(anchor_ids) > 1
    ]
    return tracks, conflicts


def unique_track_members(tracks: dict[int, Sequence[TrackMember]]) -> list[TrackMember]:
    """Flatten tracks while writing a shared frame/mask claim only once."""
    unique = {}
    for members in tracks.values():
        for member in members:
            key = (member.frame_index, member.mask_id)
            previous = unique.get(key)
            if previous is None or member.score > previous.score:
                unique[key] = member
    return [unique[key] for key in sorted(unique)]


def apply_class_override(
    frames: Sequence[TrackFrame],
    members: Sequence[TrackMember],
    target_class: int,
    override_id: str,
) -> list[dict]:
    """Return updated payload copies with an auditable class-track override."""
    if target_class <= 0:
        raise ValueError("target_class must be a positive semantic class")
    updated = []
    member_by_index = {member.frame_index: member for member in members}
    for index, frame in enumerate(frames):
        payload = dict(frame.payload)
        member = member_by_index.get(index)
        if member is None:
            updated.append(payload)
            continue
        classes = np.array(payload.get("classes", frame.classes), dtype=np.int16, copy=True)
        if member.mask_id >= len(classes):
            classes = np.pad(classes, (0, member.mask_id - len(classes) + 1))
        previous_class = int(classes[member.mask_id])
        classes[member.mask_id] = int(target_class)
        payload["classes"] = classes
        channel_segmentations = payload.get("channel_segmentations")
        if isinstance(channel_segmentations, dict):
            copied_channels = dict(channel_segmentations)
            active = int(payload.get("active_channel_index", 0))
            state_key = active if active in copied_channels else str(active)
            state = copied_channels.get(state_key)
            if isinstance(state, dict):
                state = dict(state)
                mask_classes = np.array(state.get("mask_classes", classes), dtype=np.int16, copy=True)
                if member.mask_id >= len(mask_classes):
                    mask_classes = np.pad(mask_classes, (0, member.mask_id - len(mask_classes) + 1))
                mask_classes[member.mask_id] = int(target_class)
                state["mask_classes"] = mask_classes
                copied_channels[state_key] = state
                payload["channel_segmentations"] = copied_channels
        overrides = list(payload.get("class_track_overrides") or [])
        overrides.append(
            {
                "override_id": override_id,
                "frame_id": frame.frame_id,
                "mask_id": int(member.mask_id),
                "previous_class": previous_class,
                "target_class": int(target_class),
                "match_score": float(member.score),
                "class_agnostic_tracking": True,
            }
        )
        payload["class_track_overrides"] = overrides
        updated.append(payload)
    return updated


def apply_multi_class_override(
    frames: Sequence[TrackFrame],
    tracks: dict[int, Sequence[TrackMember]],
    target_class: int,
    override_id: str,
) -> list[dict]:
    """Apply all unique track claims and retain their originating anchors in audit data."""
    if target_class <= 0:
        raise ValueError("target_class must be a positive semantic class")
    claims: dict[int, dict[int, dict]] = {}
    for anchor_id, members in tracks.items():
        for member in members:
            claim = claims.setdefault(member.frame_index, {}).setdefault(
                member.mask_id, {"score": member.score, "anchors": []}
            )
            claim["score"] = max(float(claim["score"]), float(member.score))
            claim["anchors"].append(int(anchor_id))

    updated = []
    for frame_index, frame in enumerate(frames):
        payload = dict(frame.payload)
        frame_claims = claims.get(frame_index, {})
        if not frame_claims:
            updated.append(payload)
            continue
        classes = np.array(payload.get("classes", frame.classes), dtype=np.int16, copy=True)
        max_id = max(frame_claims)
        if max_id >= len(classes):
            classes = np.pad(classes, (0, max_id - len(classes) + 1))
        previous = {mask_id: int(classes[mask_id]) for mask_id in frame_claims}
        for mask_id in frame_claims:
            classes[mask_id] = int(target_class)
        payload["classes"] = classes

        channel_segmentations = payload.get("channel_segmentations")
        if isinstance(channel_segmentations, dict):
            copied_channels = dict(channel_segmentations)
            active = int(payload.get("active_channel_index", 0))
            state_key = active if active in copied_channels else str(active)
            state = copied_channels.get(state_key)
            if isinstance(state, dict):
                state = dict(state)
                mask_classes = np.array(state.get("mask_classes", classes), dtype=np.int16, copy=True)
                if max_id >= len(mask_classes):
                    mask_classes = np.pad(mask_classes, (0, max_id - len(mask_classes) + 1))
                for mask_id in frame_claims:
                    mask_classes[mask_id] = int(target_class)
                state["mask_classes"] = mask_classes
                copied_channels[state_key] = state
                payload["channel_segmentations"] = copied_channels

        overrides = list(payload.get("class_track_overrides") or [])
        for mask_id, claim in frame_claims.items():
            overrides.append(
                {
                    "override_id": override_id,
                    "frame_id": frame.frame_id,
                    "mask_id": int(mask_id),
                    "anchor_mask_ids": sorted(set(claim["anchors"])),
                    "previous_class": previous[mask_id],
                    "target_class": int(target_class),
                    "match_score": float(claim["score"]),
                    "class_agnostic_tracking": True,
                }
            )
        payload["class_track_overrides"] = overrides
        updated.append(payload)
    return updated


def save_and_verify_class_overrides(
    frames: Sequence[TrackFrame],
    payloads: Sequence[dict],
    members: Sequence[TrackMember],
    target_class: int,
) -> int:
    """Atomically replace each frame payload, verify it, and roll back on error."""
    temporary = []
    replaced = []
    member_by_index = {member.frame_index: member for member in members}

    def _write(path, payload):
        with path.open("wb") as handle:
            np.save(handle, payload, allow_pickle=True)

    try:
        for frame, payload in zip(frames, payloads):
            temp_path = frame.path.with_name(frame.path.name + ".classlock.tmp")
            _write(temp_path, payload)
            temporary.append((temp_path, frame.path))
        for index, (temp_path, destination) in enumerate(temporary):
            os.replace(temp_path, destination)
            replaced.append((destination, frames[index].payload))
        verified = 0
        for index, frame in enumerate(frames):
            member = member_by_index.get(index)
            if member is None:
                continue
            saved = np.load(frame.path, allow_pickle=True).item()
            classes = np.asarray(saved.get("classes"))
            if member.mask_id >= len(classes) or int(classes[member.mask_id]) != int(target_class):
                raise RuntimeError(f"Class override verification failed for {frame.path.name}")
            verified += 1
        return verified
    except Exception:
        for destination, original_payload in replaced:
            rollback_path = destination.with_name(destination.name + ".classlock.rollback")
            _write(rollback_path, original_payload)
            os.replace(rollback_path, destination)
        raise
    finally:
        for temp_path, _destination in temporary:
            if temp_path.exists():
                temp_path.unlink()


def save_and_verify_multi_class_overrides(
    frames: Sequence[TrackFrame],
    payloads: Sequence[dict],
    tracks: dict[int, Sequence[TrackMember]],
    target_class: int,
) -> int:
    """Transactionally save and verify every unique frame/mask assignment."""
    members = unique_track_members(tracks)
    temporary = []
    replaced = []

    def _write(path, payload):
        with path.open("wb") as handle:
            np.save(handle, payload, allow_pickle=True)

    try:
        for frame, payload in zip(frames, payloads):
            temp_path = frame.path.with_name(frame.path.name + ".classlock.tmp")
            _write(temp_path, payload)
            temporary.append((temp_path, frame.path))
        for index, (temp_path, destination) in enumerate(temporary):
            os.replace(temp_path, destination)
            replaced.append((destination, frames[index].payload))
        verified = 0
        for member in members:
            saved = np.load(frames[member.frame_index].path, allow_pickle=True).item()
            classes = np.asarray(saved.get("classes"))
            if member.mask_id >= len(classes) or int(classes[member.mask_id]) != int(target_class):
                raise RuntimeError(
                    f"Class override verification failed for {frames[member.frame_index].path.name}"
                )
            verified += 1
        return verified
    except Exception:
        for destination, original_payload in replaced:
            rollback_path = destination.with_name(destination.name + ".classlock.rollback")
            _write(rollback_path, original_payload)
            os.replace(rollback_path, destination)
        raise
    finally:
        for temp_path, _destination in temporary:
            if temp_path.exists():
                temp_path.unlink()
