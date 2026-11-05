"""Build the full-match label array from the tactical bundle."""

import os

import numpy as np

from mtah.params import (
    CONTROL_GAP_FRAMES, CONTROL_RADIUS_M, MAX_CONTROL_SPEED_MPS,
    MIN_CONTROL_FRAMES, OUT_OF_PITCH_MARGIN_M,
)
from mtah.regions import make_region_masks
from mtah.signals import read_signals
from mtah.state import absorb_touches, decide
from mtah.tags import masks_from_events


def build_labels(data_dir=None, game="BAR-ATH"):
    """H/A/D/X for every tactical frame. Requires the ATD and Smart Tagging."""
    from games import GAMES, require, source_frames
    from starter.baseline import team_of_each_track
    from starter.load_smart_tagging import load

    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    signals = read_signals(
        raw_path,
        team_of_each_track(meta_path),
        n_frames,
        GAMES[game]["pitch"],
        OUT_OF_PITCH_MARGIN_M,
    )
    set_piece, restart = masks_from_events(load(game, data_dir), n_frames)
    ungraded, pre_kickoff = make_region_masks(n_frames, game)
    labels = decide(
        signals.ball_present,
        signals.ball_out,
        signals.nearest_dist,
        signals.nearest_team,
        set_piece,
        restart,
        ungraded,
        pre_kickoff,
        CONTROL_RADIUS_M,
        MIN_CONTROL_FRAMES,
        CONTROL_GAP_FRAMES,
        signals.ball_speed,
        MAX_CONTROL_SPEED_MPS,
    )
    return absorb_touches(labels, MIN_CONTROL_FRAMES)


def write_submission(path, labels):
    """frame,label CSV. One row per video frame, frames starting at 0."""
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))
    return path


def summarise(labels):
    """Label counts only. No coordinates, no frame dumps."""
    lines = ["rows {}".format(len(labels))]
    total = float(len(labels)) if len(labels) else 1.0
    for label in ("H", "A", "D", "X"):
        count = int(np.sum(labels == label))
        lines.append("{} {:>7} {:5.1f}%".format(label, count, 100.0 * count / total))
    return "\n".join(lines)
