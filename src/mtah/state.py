"""One pass from signals to H/A/D/X, then a cleanup of touches.

Dead ball wins over control. A missing ball, or a ball nobody is close to,
keeps the last team that had completed a dwell. A new team takes the ball
only after it stays inside the control radius for dwell_frames, and that
change is written from the frame the dwell started.

A spell shorter than min_control_frames is not control. It is given back to
the team on either side when that side already owns the ball, and it never
crosses a dead-ball frame.
"""

import numpy as np

_ALIVE = ("H", "A")


def _letter(possessor):
    if possessor == 0:
        return "H"
    if possessor == 1:
        return "A"
    return "X"


def decide(ball_present, ball_out, nearest_dist, nearest_team,
           set_piece, restart_team, ungraded, pre_kickoff,
           control_radius_m, dwell_frames):
    """Label every frame. Arrays are the same length. Possessor codes are 0/1."""
    n = len(ball_present)
    labels = np.empty(n, dtype="<U1")
    possessor = -1
    dwell_team = -1
    dwell_count = 0
    dwell_frames = max(1, int(dwell_frames))

    for i in range(n):
        if ungraded[i]:
            labels[i] = "X"
            possessor = -1
            dwell_team = -1
            dwell_count = 0
            continue
        if pre_kickoff[i]:
            labels[i] = "D"
            possessor = -1
            dwell_team = -1
            dwell_count = 0
            continue

        restart = int(restart_team[i])
        if restart >= 0:
            possessor = restart
            dwell_team = -1
            dwell_count = 0

        if set_piece[i] or ball_out[i]:
            labels[i] = "D"
            dwell_team = -1
            dwell_count = 0
            continue

        if not ball_present[i]:
            labels[i] = _letter(possessor)
            dwell_team = -1
            dwell_count = 0
            continue

        team = int(nearest_team[i])
        dist = nearest_dist[i]
        close = team >= 0 and dist == dist and dist <= control_radius_m
        if close:
            if team == dwell_team:
                dwell_count += 1
            else:
                dwell_team = team
                dwell_count = 1
            if dwell_count >= dwell_frames and possessor != team:
                possessor = team
                # The reception started when the dwell started, not when the
                # counter finished. Leave dead-ball frames alone.
                letter = _letter(team)
                for frame in range(max(0, i - dwell_frames + 1), i + 1):
                    if labels[frame] in ("H", "A", "X"):
                        labels[frame] = letter
        else:
            dwell_team = -1
            dwell_count = 0
        labels[i] = _letter(possessor)

    return labels


def _runs(labels):
    runs = []
    start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start]:
            runs.append((start, index, str(labels[start])))
            start = index
    return runs


def absorb_touches(labels, min_frames):
    """Drop touches that never become control. D and X are walls."""
    labels = np.array(labels, dtype="<U1", copy=True)
    min_frames = max(1, int(min_frames))
    for _ in range(8):
        runs = _runs(labels)
        changed = False
        for index, (start, end, label) in enumerate(runs):
            if label not in _ALIVE or end - start >= min_frames:
                continue
            left = runs[index - 1] if index else None
            right = runs[index + 1] if index + 1 < len(runs) else None
            left_team = left[2] if left and left[2] in _ALIVE else None
            right_team = right[2] if right and right[2] in _ALIVE else None
            right_long = (
                right is not None
                and right[2] in _ALIVE
                and right[1] - right[0] >= min_frames
            )
            if left_team and right_team and left_team == right_team and left_team != label:
                replacement = left_team
            elif right_long and right_team != label:
                replacement = right_team
            elif left_team and left_team != label and not right_long:
                replacement = left_team
            else:
                continue
            labels[start:end] = replacement
            changed = True
        if not changed:
            break
    return labels
