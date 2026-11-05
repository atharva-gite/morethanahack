"""Control from the ball and the players around it, then a cleanup of touches.

Dead ball wins. A loose ball or a duel stays with the last team that had
control. A new team takes the ball when it is at their feet, or when a slow
ball reaches them and the other team is not also there. The label changes
on the frame control sticks. The frames before that stay with the previous team.

A spell shorter than a second is a touch when the same team is on both
sides, or when it dies straight into a stoppage. It never crosses a dead ball.
"""

import numpy as np

from mtah.params import (
    CONTEST_RADIUS_M, FEET_RADIUS_M, ISOLATION_GAP_M, RECEPTION_RADIUS_M,
    SLOW_BALL_MPS, SPEED_MATCH_MPS,
)

_ALIVE = ("H", "A")


def _under_control(dist, team, ball_speed, player_speed, other_dist, other_team):
    """True when one player has the ball and the other team is not on it too."""
    if team < 0 or dist != dist:
        return False
    contested = (
        other_team >= 0 and other_team != team and other_dist == other_dist
        and other_dist <= CONTEST_RADIUS_M and dist <= CONTEST_RADIUS_M
    )
    if contested:
        return False
    if (
        other_team >= 0 and other_team != team and other_dist == other_dist
        and other_dist < dist + ISOLATION_GAP_M
    ):
        return False
    speeds_known = (
        ball_speed == ball_speed and player_speed == player_speed
    )
    if dist <= FEET_RADIUS_M:
        if not speeds_known:
            return True
        return abs(ball_speed - player_speed) <= SPEED_MATCH_MPS or ball_speed <= SLOW_BALL_MPS
    return (
        dist <= RECEPTION_RADIUS_M and speeds_known and ball_speed <= SLOW_BALL_MPS
    )


def _letter(possessor):
    if possessor == 0:
        return "H"
    if possessor == 1:
        return "A"
    return "X"


def decide(ball_present, ball_out, nearest_dist, nearest_team,
           set_piece, restart_team, ungraded, pre_kickoff,
           control_radius_m, dwell_frames, gap_frames=0, ball_speed=None,
           max_control_speed=None, nearest_speed=None, second_dist=None,
           second_team=None):
    """Label every frame. Arrays are the same length. Possessor codes are 0/1.

    With player speeds, control means the ball is at a player's feet or has
    slowed into a reception, and the other team is not also on it. Without
    those arrays, control falls back to distance inside control_radius_m.
    A challenge survives gap_frames of a loose or missing ball. The new team
    starts on the frame control sticks, not on the frames that led up to it.
    """
    n = len(ball_present)
    labels = np.empty(n, dtype="<U1")
    possessor = -1
    dwell_team = -1
    dwell_count = 0
    gap_left = 0
    dwell_frames = max(1, int(dwell_frames))
    gap_frames = max(0, int(gap_frames))

    for i in range(n):
        if ungraded[i]:
            labels[i] = "X"
            possessor = -1
            dwell_team = -1
            dwell_count = 0
            gap_left = 0
            continue
        if pre_kickoff[i]:
            labels[i] = "D"
            possessor = -1
            dwell_team = -1
            dwell_count = 0
            gap_left = 0
            continue

        restart = int(restart_team[i])
        if restart >= 0:
            possessor = restart
            dwell_team = -1
            dwell_count = 0
            gap_left = 0

        if set_piece[i] or ball_out[i]:
            labels[i] = "D"
            dwell_team = -1
            dwell_count = 0
            gap_left = 0
            continue

        team = int(nearest_team[i])
        dist = nearest_dist[i]
        speed = ball_speed[i] if ball_speed is not None else np.nan
        in_flight = (
            max_control_speed is not None and speed == speed and speed > max_control_speed
        )
        if nearest_speed is None:
            close = (
                ball_present[i] and team >= 0 and dist == dist
                and dist <= control_radius_m and not in_flight
            )
        elif ball_present[i] and not in_flight:
            other_dist = second_dist[i] if second_dist is not None else np.nan
            other_team = int(second_team[i]) if second_team is not None else -1
            close = _under_control(
                dist, team, speed, nearest_speed[i], other_dist, other_team,
            )
        else:
            close = False
        if close and team == possessor:
            dwell_team = -1
            dwell_count = 0
            gap_left = 0
        elif close:
            if team != dwell_team:
                dwell_team = team
                dwell_count = 1
            else:
                dwell_count += 1
            gap_left = gap_frames
            if dwell_count >= dwell_frames and possessor != team:
                possessor = team
        elif dwell_team >= 0 and gap_left > 0:
            gap_left -= 1
        else:
            dwell_team = -1
            dwell_count = 0
            gap_left = 0
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
