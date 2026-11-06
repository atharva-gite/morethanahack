"""Aggregates only. No coordinates and no sample rows.

    python scripts/diagnose.py --data-dir /content/data

Read the build, in this order, and then stop:
possession count and mean duration, lat_f, home and away in-play share.
Public spellF1 is only a check that the file is not near zero.
Do not keep a file because that number rose.
Discard a file that collapses to a handful of minute-long spells.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from games import GAME, GAMES, source_frames  # noqa: E402
from starter.baseline import team_of_each_track  # noqa: E402
from starter.load_atd import atd_paths  # noqa: E402
from mtah.params import OUT_OF_PITCH_MARGIN_M  # noqa: E402
from mtah.regions import make_region_masks  # noqa: E402
from mtah.signals import read_signals  # noqa: E402
from mtah.state import _motion_speed, _under_control  # noqa: E402


def _run_lengths(mask):
    lengths = []
    count = 0
    for flag in mask:
        if flag:
            count += 1
        elif count:
            lengths.append(count)
            count = 0
    if count:
        lengths.append(count)
    return np.array(lengths, dtype=float) if lengths else np.zeros(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    meta_path, raw_path = atd_paths(GAME, args.data_dir)
    n_frames = source_frames("tactical", GAME)
    signals = read_signals(
        raw_path, team_of_each_track(meta_path), n_frames,
        GAMES[GAME]["pitch"], OUT_OF_PITCH_MARGIN_M,
    )
    ungraded, _pre = make_region_masks(n_frames, GAME)
    graded = ~ungraded
    missing = graded & ~signals.ball_present
    lengths = _run_lengths(missing)
    print("missing-ball runs on graded frames")
    print("  n {:>6}  median {:>6.0f} frames  p90 {:>6.0f} frames".format(
        len(lengths), np.median(lengths), np.percentile(lengths, 90)))

    speed = _motion_speed(
        signals.ball_present, signals.ball_x, signals.ball_y, signals.ball_speed,
    )
    visible = speed[signals.ball_present]
    visible = visible[visible == visible]
    if len(visible):
        p10, p50, p90 = np.percentile(visible, [10, 50, 90])
        print("ball speed m/s on visible frames  p10 {:.1f}  p50 {:.1f}  p90 {:.1f}".format(
            p10, p50, p90))
    else:
        print("ball speed m/s on visible frames  none")

    home_only = away_only = both = 0
    visible_n = 0
    for index in np.flatnonzero(signals.ball_present):
        visible_n += 1
        spd = speed[index]
        home = _under_control(
            signals.home_dist[index], 0, spd, signals.home_speed[index],
            signals.away_dist[index], 1,
        )
        away = _under_control(
            signals.away_dist[index], 1, spd, signals.away_speed[index],
            signals.home_dist[index], 0,
        )
        if home and away:
            both += 1
        elif home:
            home_only += 1
        elif away:
            away_only += 1
    denom = float(visible_n) if visible_n else 1.0
    print("unique player at the ball, share of visible frames")
    print("  home {:>5.1f}%  away {:>5.1f}%  both {:>5.1f}%".format(
        100.0 * home_only / denom, 100.0 * away_only / denom, 100.0 * both / denom))
    print("frames {}  visible {}".format(n_frames, visible_n))
    print("Freeze: possession count, mean duration, lat_f, home/away share.")
    print("Public spellF1 only confirms the file is not near zero.")
    print("Do not keep a file because spellF1 rose.")


if __name__ == "__main__":
    main()
