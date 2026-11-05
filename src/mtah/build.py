"""Build the full-match label array from the tactical bundle."""

import os

import numpy as np

from mtah.params import (
    CONTROL_GAP_FRAMES, OUT_OF_PITCH_MARGIN_M, RECEPTION_FRAMES, RECEPTION_RADIUS_M,
    SUPPORT_RADIUS_M, TOUCH_FRAMES,
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
        SUPPORT_RADIUS_M,
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
        RECEPTION_RADIUS_M,
        RECEPTION_FRAMES,
        CONTROL_GAP_FRAMES,
        signals.ball_speed,
        None,
        signals.nearest_speed,
        signals.second_dist,
        signals.second_team,
        signals.home_near,
        signals.away_near,
    )
    return absorb_touches(labels, TOUCH_FRAMES)


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
    """Label counts and in-play time. No coordinates, no frame dumps.

    The home and away split is what this file claims. It is not the private
    ground truth and not an official match report.
    """
    from games import GAME, GAMES

    fps = float(GAMES[GAME]["fps"])
    lines = ["rows {}".format(len(labels))]
    total = float(len(labels)) if len(labels) else 1.0
    counts = {}
    for label in ("H", "A", "D", "X"):
        count = int(np.sum(labels == label))
        counts[label] = count
        lines.append("{} {:>7} {:5.1f}%".format(label, count, 100.0 * count / total))
    alive = counts["H"] + counts["A"]
    lines.append("in-play possession over the whole match, from this file")
    lines.append("not a published share and not the private ground truth")
    if alive:
        lines.append("H {:>7.1f}s  {:5.1f}% of in-play   ({:.1f} min)".format(
            counts["H"] / fps, 100.0 * counts["H"] / alive, counts["H"] / fps / 60.0))
        lines.append("A {:>7.1f}s  {:5.1f}% of in-play   ({:.1f} min)".format(
            counts["A"] / fps, 100.0 * counts["A"] / alive, counts["A"] / fps / 60.0))
    else:
        lines.append("no in-play frames")
    return "\n".join(lines)
