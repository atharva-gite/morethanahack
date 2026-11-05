"""Stream the tactical ATD the same way starter/baseline.py does.

One record per video frame. frame_id is 1-based, so video_frame = frame_id - 1.
Coordinates in the file are 0-1; distances below are metres on games.py's pitch.
"""

from collections import namedtuple

import numpy as np

FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING = "N"

Signals = namedtuple(
    "Signals",
    "ball_present ball_x ball_y ball_speed nearest_dist nearest_team nearest_speed "
    "second_dist second_team ball_out",
)


def _parts(group):
    """(x, y, speed) from one 'x,y,speed' group, or None when it is missing."""
    group = group.strip()
    if not group or group[0] == MISSING or group[0] == "n":
        return None
    bits = group.split(",")
    if len(bits) < 2:
        return None
    try:
        x = float(bits[0])
        y = float(bits[1])
    except ValueError:
        return None
    if np.isnan(x) or np.isnan(y):
        return None
    speed = np.nan
    if len(bits) >= 3:
        try:
            speed = float(bits[2])
        except ValueError:
            speed = np.nan
    return x, y, speed


def read_signals(raw_path, team_of_track, n_frames, pitch, margin_m):
    """Per-frame ball and nearest-player signals. Missing ball stays NaN.

    team_of_track is the slot array from starter.baseline.team_of_each_track:
    0 home, 1 away. nearest_team is -1 when no player is next to a visible ball.
    """
    length, width = pitch
    ball_present = np.zeros(n_frames, dtype=bool)
    ball_x = np.full(n_frames, np.nan)
    ball_y = np.full(n_frames, np.nan)
    ball_speed = np.full(n_frames, np.nan)
    nearest_dist = np.full(n_frames, np.nan)
    nearest_team = np.full(n_frames, -1, dtype=np.int8)
    nearest_speed = np.full(n_frames, np.nan)
    second_dist = np.full(n_frames, np.nan)
    second_team = np.full(n_frames, -1, dtype=np.int8)
    ball_out = np.zeros(n_frames, dtype=bool)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            try:
                video_frame = int(parts[0]) - 1
            except ValueError:
                continue
            if video_frame < 0 or video_frame >= n_frames:
                continue

            ball = _parts(parts[2])
            if ball is None:
                continue
            norm_x, norm_y, speed = ball
            x = norm_x * length
            y = norm_y * width
            ball_present[video_frame] = True
            ball_x[video_frame] = x
            ball_y[video_frame] = y
            ball_speed[video_frame] = speed
            ball_out[video_frame] = (
                x < -margin_m or x > length + margin_m
                or y < -margin_m or y > width + margin_m
            )

            best = None
            second = None
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if slot >= len(team_of_track) or not group:
                    continue
                player = _parts(group)
                if player is None:
                    continue
                dx = player[0] * length - x
                dy = player[1] * width - y
                distance = dx * dx + dy * dy
                team = int(team_of_track[slot])
                if best is None or distance < best[0]:
                    second = best
                    best = (distance, team, player[2])
                elif second is None or distance < second[0]:
                    second = (distance, team, player[2])
            if best is not None:
                nearest_dist[video_frame] = best[0] ** 0.5
                nearest_team[video_frame] = best[1]
                nearest_speed[video_frame] = best[2]
            if second is not None:
                second_dist[video_frame] = second[0] ** 0.5
                second_team[video_frame] = second[1]

    return Signals(
        ball_present, ball_x, ball_y, ball_speed, nearest_dist, nearest_team,
        nearest_speed, second_dist, second_team, ball_out,
    )
