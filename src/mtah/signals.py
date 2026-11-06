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
    "nearest_slot second_dist second_team home_near away_near ball_out "
    "home_dist away_dist home_speed away_speed home_x home_y away_x away_y "
    "ball_vx ball_vy",
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


def _fill_ball_velocity(ball_present, ball_x, ball_y, ball_vx, ball_vy, fps=25):
    """Metres per second from consecutive visible ball positions. Missing stays NaN."""
    for index in range(1, len(ball_present)):
        if not ball_present[index] or not ball_present[index - 1]:
            continue
        ball_vx[index] = (ball_x[index] - ball_x[index - 1]) * fps
        ball_vy[index] = (ball_y[index] - ball_y[index - 1]) * fps


def read_signals(raw_path, team_of_track, n_frames, pitch, margin_m, support_radius_m=12.0):
    """Per-frame ball and nearest-player signals. Missing ball stays NaN.

    team_of_track is the slot array from starter.baseline.team_of_each_track:
    0 home, 1 away. nearest_team is -1 when no player is next to a visible ball.
    Ball velocity is metres per second from consecutive visible positions.
    """
    length, width = pitch
    ball_present = np.zeros(n_frames, dtype=bool)
    ball_x = np.full(n_frames, np.nan)
    ball_y = np.full(n_frames, np.nan)
    ball_speed = np.full(n_frames, np.nan)
    nearest_dist = np.full(n_frames, np.nan)
    nearest_team = np.full(n_frames, -1, dtype=np.int8)
    nearest_speed = np.full(n_frames, np.nan)
    nearest_slot = np.full(n_frames, -1, dtype=np.int16)
    second_dist = np.full(n_frames, np.nan)
    second_team = np.full(n_frames, -1, dtype=np.int8)
    home_near = np.zeros(n_frames, dtype=np.int16)
    away_near = np.zeros(n_frames, dtype=np.int16)
    ball_out = np.zeros(n_frames, dtype=bool)
    home_dist = np.full(n_frames, np.nan)
    away_dist = np.full(n_frames, np.nan)
    home_speed = np.full(n_frames, np.nan)
    away_speed = np.full(n_frames, np.nan)
    home_x = np.full(n_frames, np.nan)
    home_y = np.full(n_frames, np.nan)
    away_x = np.full(n_frames, np.nan)
    away_y = np.full(n_frames, np.nan)
    ball_vx = np.full(n_frames, np.nan)
    ball_vy = np.full(n_frames, np.nan)
    support_2 = float(support_radius_m) * float(support_radius_m)

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
            home_best = None
            away_best = None
            home_near_n = 0
            away_near_n = 0
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
                if team == 0:
                    if home_best is None or distance < home_best[0]:
                        home_best = (distance, player[2], player[0] * length, player[1] * width)
                elif away_best is None or distance < away_best[0]:
                    away_best = (distance, player[2], player[0] * length, player[1] * width)
                if distance <= support_2:
                    if team == 0:
                        home_near_n += 1
                    else:
                        away_near_n += 1
                if best is None or distance < best[0]:
                    second = best
                    best = (distance, team, player[2], slot)
                elif second is None or distance < second[0]:
                    second = (distance, team, player[2], slot)
            if best is not None:
                nearest_dist[video_frame] = best[0] ** 0.5
                nearest_team[video_frame] = best[1]
                nearest_speed[video_frame] = best[2]
                nearest_slot[video_frame] = best[3]
            if second is not None:
                second_dist[video_frame] = second[0] ** 0.5
                second_team[video_frame] = second[1]
            home_near[video_frame] = home_near_n
            away_near[video_frame] = away_near_n
            if home_best is not None:
                home_dist[video_frame] = home_best[0] ** 0.5
                home_speed[video_frame] = home_best[1]
                home_x[video_frame] = home_best[2]
                home_y[video_frame] = home_best[3]
            if away_best is not None:
                away_dist[video_frame] = away_best[0] ** 0.5
                away_speed[video_frame] = away_best[1]
                away_x[video_frame] = away_best[2]
                away_y[video_frame] = away_best[3]

    _fill_ball_velocity(ball_present, ball_x, ball_y, ball_vx, ball_vy)

    return Signals(
        ball_present, ball_x, ball_y, ball_speed, nearest_dist, nearest_team,
        nearest_speed, nearest_slot, second_dist, second_team, home_near, away_near,
        ball_out, home_dist, away_dist, home_speed, away_speed,
        home_x, home_y, away_x, away_y, ball_vx, ball_vy,
    )
