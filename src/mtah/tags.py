"""Smart Tagging as evidence of stoppages and restarts, not as possessions.

SET PIECES intervals become dead ball. A named team on a restart is applied
at the first frame after the interval. Team N/A, which is every throw-in,
leaves the possessor as the last side that had control.
"""

import numpy as np

from mtah.params import FIRST_HALF_KICKOFF, KICKOFF_TYPES, SECOND_HALF_KICKOFF, SET_PIECE_CODE

_KICKOFF_FRAMES = (FIRST_HALF_KICKOFF, SECOND_HALF_KICKOFF)


def side_of_team_tag(value):
    """Map a Team tag to 0 (home) or 1 (away). None when it is missing or N/A.

    The file spells the away club 'Atletic Club'.
    """
    if value is None:
        return None
    key = value.strip().lower()
    if key in ("", "n/a", "na"):
        return None
    if "barc" in key:
        return 0
    if "atlet" in key or "athlet" in key:
        return 1
    return None


def _kickoff_inside(start, end):
    """Published kickoff contained by this window, using the official check.

    load() documents end_frame as exclusive, but check_time_base treats it as
    inclusive. Matching that check is what keeps a kickoff on the boundary.
    """
    for frame in _KICKOFF_FRAMES:
        if start <= frame <= end:
            return frame
    return None


def masks_from_events(events, n_frames):
    """Return (set_piece, restart_team) arrays of length n_frames.

    set_piece is true on dead-ball frames. restart_team is -1, or 0/1 on the
    frame a named restart should hand possession to that side.
    """
    set_piece = np.zeros(n_frames, dtype=bool)
    restart = np.full(n_frames, -1, dtype=np.int8)

    for event in events:
        if (event.get("code") or "").strip().lower() != SET_PIECE_CODE:
            continue
        start = int(event["start_frame"])
        end = int(event["end_frame"])
        if end < start:
            continue
        typ = (event.get("tags", {}).get("Type") or "").strip().lower()
        side = side_of_team_tag(event.get("tags", {}).get("Team"))
        kickoff = _kickoff_inside(start, end) if typ in KICKOFF_TYPES else None

        if kickoff is None:
            dead_lo = max(0, start)
            dead_hi = min(n_frames, end)
            if dead_lo < dead_hi:
                set_piece[dead_lo:dead_hi] = True
            if side is not None and 0 <= end < n_frames:
                restart[end] = side
            continue

        dead_lo = max(0, start)
        dead_hi = min(n_frames, kickoff)
        if dead_lo < dead_hi:
            set_piece[dead_lo:dead_hi] = True
        if side is not None and 0 <= kickoff < n_frames:
            restart[kickoff] = side

    return set_piece, restart
