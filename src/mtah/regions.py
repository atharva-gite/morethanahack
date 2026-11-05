"""Graded, ungraded, and second-half pre-kickoff masks.

Frame bounds come from games.half_video_blocks. The kickoff instant is the
published frame in FAQ.md, not a number read off the video.
"""

import numpy as np

from mtah.params import SECOND_HALF_KICKOFF


def make_region_masks(n_frames, game="BAR-ATH"):
    """Return (ungraded, pre_kickoff), each a bool array of length n_frames.

    pre_kickoff is the graded dead-ball stretch from the start of the second
    half up to, but not including, the published kickoff.
    """
    from games import half_video_blocks

    ungraded = np.ones(n_frames, dtype=bool)
    pre_kickoff = np.zeros(n_frames, dtype=bool)
    for half, first, last in half_video_blocks(game):
        first = max(0, int(first))
        last = min(n_frames - 1, int(last))
        if first > last:
            continue
        ungraded[first:last + 1] = False
        if half == 2:
            dead_last = min(last, SECOND_HALF_KICKOFF - 1)
            if first <= dead_last:
                pre_kickoff[first:dead_last + 1] = True
    return ungraded, pre_kickoff
