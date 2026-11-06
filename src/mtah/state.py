"""Possession from arrivals.

Dead ball wins. The spell belongs to the team the ball last arrived to and
stayed with. A pass in flight, a loose ball, and an opponent who is only
nearby do not move it. The other team takes it when the ball arrives at one
of their players and stays there. Slots are not people: a jump in distance
is a different player.
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


def _has_received(ball_speed, player_speed):
    """The new team is dribbling, or the ball has stopped with them.

    A ball still rolling past a standing player is a pass, not a reception.
    Missing speeds do not block the change.
    """
    if not (ball_speed == ball_speed and player_speed == player_speed):
        return True
    if abs(ball_speed - player_speed) <= SPEED_MATCH_MPS:
        return True
    return ball_speed <= 4.0 and player_speed <= 4.0 and ball_speed <= player_speed + 1.5


def _letter(possessor):
    if possessor == 0:
        return "H"
    if possessor == 1:
        return "A"
    return "X"


def _motion_speed(ball_present, ball_x, ball_y, ball_speed):
    """Speed from positions when two frames exist, otherwise the ATD scalar."""
    n = len(ball_present)
    out = np.array(ball_speed, dtype=float, copy=True) if ball_speed is not None else np.full(n, np.nan)
    if ball_x is None or ball_y is None:
        return out
    for index in range(1, n):
        if not ball_present[index] or not ball_present[index - 1]:
            continue
        vx = (ball_x[index] - ball_x[index - 1]) * 25.0
        vy = (ball_y[index] - ball_y[index - 1]) * 25.0
        out[index] = (vx * vx + vy * vy) ** 0.5
    return out


def _carrier_place(team, index, home_x, home_y, away_x, away_y):
    if home_x is None:
        return None
    if team == 0:
        return float(home_x[index]), float(home_y[index])
    return float(away_x[index]), float(away_y[index])


def _moved_too_far(previous, current):
    if previous is None or current is None:
        return False
    if any(value != value for value in previous + current):
        return False
    dx = current[0] - previous[0]
    dy = current[1] - previous[1]
    from mtah.params import CARRIER_JUMP_M
    return (dx * dx + dy * dy) ** 0.5 > CARRIER_JUMP_M


def _touch_is_arrival(start, last, team, who, speed_used, home_speed, away_speed):
    """A change of team is a reception: the ball slowed into them, or they are dribbling."""
    from mtah.params import SLOWDOWN_LOOKBACK, SLOWDOWN_MPS

    player_speed = home_speed if team == 0 else away_speed
    for frame in range(start, last + 1):
        if who[frame] != team:
            continue
        speed = speed_used[frame]
        if _has_received(speed, player_speed[frame]):
            return True
        if frame < SLOWDOWN_LOOKBACK:
            continue
        earlier = speed_used[frame - SLOWDOWN_LOOKBACK]
        if speed == speed and earlier == earlier and speed <= earlier - SLOWDOWN_MPS:
            return True
    return False


def _mark_stopped_restarts(kind, who, ball_present, speed_used, home_dist, away_dist):
    """Visible, stopped, and far from both teams for a full second becomes dead ball."""
    from mtah.params import FAR_FROM_PLAYERS_M, STOPPED_BALL_MPS, STOPPED_DEAD_FRAMES

    run = 0
    start = 0
    for index in range(len(kind)):
        speed = speed_used[index]
        far = (
            home_dist[index] == home_dist[index] and away_dist[index] == away_dist[index]
            and home_dist[index] > FAR_FROM_PLAYERS_M and away_dist[index] > FAR_FROM_PLAYERS_M
        )
        stopped = speed == speed and speed <= STOPPED_BALL_MPS
        if kind[index] == "P" and ball_present[index] and far and stopped:
            if run == 0:
                start = index
            run += 1
            if run >= STOPPED_DEAD_FRAMES:
                kind[start:index + 1] = "D"
                who[start:index + 1] = -1
            continue
        run = 0


def _team_arriving(last, ball_present, home_dist, away_dist, home_speed, away_speed, speed_used):
    """Team the ball is moving toward just before it disappears. -1 if neither."""
    from mtah.params import APPROACH_FRAMES, FEET_RADIUS_M, ISOLATION_GAP_M, RECEPTION_RADIUS_M

    if last < 0 or not ball_present[last]:
        return -1
    earlier = last
    steps = 0
    while earlier > 0 and steps < APPROACH_FRAMES and ball_present[earlier - 1]:
        earlier -= 1
        steps += 1
    home_now = home_dist[last]
    away_now = away_dist[last]
    if home_now != home_now or away_now != away_now:
        return -1
    if min(home_now, away_now) > RECEPTION_RADIUS_M:
        return -1
    if home_now + ISOLATION_GAP_M <= away_now:
        team = 0
        dist_now = home_now
        dist_then = home_dist[earlier]
    elif away_now + ISOLATION_GAP_M <= home_now:
        team = 1
        dist_now = away_now
        dist_then = away_dist[earlier]
    else:
        return -1
    closing = dist_then == dist_then and dist_now < dist_then - 0.3
    at_feet = dist_now <= FEET_RADIUS_M
    if not closing and not at_feet:
        return -1
    player_speed = home_speed[last] if team == 0 else away_speed[last]
    if not _has_received(speed_used[last], player_speed):
        return -1
    return team


def _gap_receptions(ball_present, kind, home_dist, away_dist, home_speed, away_speed, speed_used):
    """(first missing frame, team) for a long hole the ball disappeared into."""
    from mtah.params import GAP_OWN_FRAMES

    found = []
    index = 0
    n = len(ball_present)
    while index < n:
        if ball_present[index] or kind[index] != "P":
            index += 1
            continue
        start = index
        while index < n and (not ball_present[index]) and kind[index] == "P":
            index += 1
        if index - start < GAP_OWN_FRAMES:
            continue
        team = _team_arriving(
            start - 1, ball_present, home_dist, away_dist, home_speed, away_speed, speed_used,
        )
        if team >= 0:
            found.append((start, team))
    return found


def possess_from_arrivals(ball_present, ball_out, ball_speed, home_dist, away_dist,
                          home_speed, away_speed, set_piece, restart_team, ungraded,
                          pre_kickoff, dwell_frames, gap_frames, ball_x=None, ball_y=None,
                          home_x=None, home_y=None, away_x=None, away_y=None):
    """Who the ball arrives to, then who keeps it.

    Each frame asks which team, if either, actually has the ball at a player's
    feet. A reception is that contact lasting dwell_frames without the closest
    player jumping to someone else. Until that reception sticks, and through
    the pass that follows, the spell stays with the previous team. A touch
    shorter than that is pressure. Dead ball and ungraded frames break the spell.
    """
    from mtah.params import (
        CARRIER_JUMP_M, FAR_FROM_PLAYERS_M, SLOWDOWN_LOOKBACK, SLOWDOWN_MPS,
        STOPPED_BALL_MPS, STOPPED_DEAD_FRAMES,
    )

    n = len(ball_present)
    kind = np.full(n, "P", dtype="<U1")
    who = np.full(n, -1, dtype=np.int8)
    how_far = np.full(n, np.nan)
    dwell_frames = max(1, int(dwell_frames))
    gap_frames = max(0, int(gap_frames))

    for index in range(n):
        if ungraded[index]:
            kind[index] = "X"
            continue
        if pre_kickoff[index] or set_piece[index] or ball_out[index]:
            kind[index] = "D"
            continue
        if not ball_present[index]:
            continue
        speed = ball_speed[index] if ball_speed is not None else np.nan
        home_ok = _under_control(
            home_dist[index], 0, speed, home_speed[index], away_dist[index], 1,
        )
        away_ok = _under_control(
            away_dist[index], 1, speed, away_speed[index], home_dist[index], 0,
        )
        if home_ok and not away_ok:
            who[index] = 0
            how_far[index] = home_dist[index]
        elif away_ok and not home_ok:
            who[index] = 1
            how_far[index] = away_dist[index]

    speed_used = _motion_speed(ball_present, ball_x, ball_y, ball_speed)
    _mark_stopped_restarts(
        kind, who, ball_present, speed_used, home_dist, away_dist,
    )

    touches = []
    index = 0
    while index < n:
        if kind[index] != "P" or who[index] < 0:
            index += 1
            continue
        team = int(who[index])
        start = index
        last = index
        count = 1
        hole = 0
        prev_place = _carrier_place(team, index, home_x, home_y, away_x, away_y)
        cursor = index + 1
        while cursor < n and kind[cursor] == "P":
            if who[cursor] == team:
                place = _carrier_place(team, cursor, home_x, home_y, away_x, away_y)
                if _moved_too_far(prev_place, place):
                    break
                count += 1
                last = cursor
                prev_place = place
                hole = 0
                cursor += 1
                continue
            if who[cursor] >= 0:
                break
            hole += 1
            if hole > gap_frames:
                break
            cursor += 1
        if count >= dwell_frames:
            if touches and int(touches[-1][1]) != team and not _touch_is_arrival(
                start, last, team, who, speed_used, home_speed, away_speed,
            ):
                index = last + 1
                continue
            seen = 0
            stick = start
            for frame in range(start, last + 1):
                if who[frame] == team:
                    seen += 1
                    if seen >= dwell_frames:
                        stick = frame
                        break
            touches.append((stick, team))
        index = last + 1

    touches.extend(_gap_receptions(
        ball_present, kind, home_dist, away_dist, home_speed, away_speed, speed_used,
    ))
    touches.sort(key=lambda item: item[0])

    labels = np.empty(n, dtype="<U1")
    possessor = -1
    touch_at = 0
    for index in range(n):
        if kind[index] == "X":
            labels[index] = "X"
            possessor = -1
            continue
        if kind[index] == "D":
            labels[index] = "D"
            restart = int(restart_team[index])
            possessor = restart if restart >= 0 else -1
            continue
        restart = int(restart_team[index])
        if restart >= 0:
            possessor = restart
        while touch_at < len(touches) and touches[touch_at][0] <= index:
            possessor = touches[touch_at][1]
            touch_at += 1
        labels[index] = _letter(possessor)
    return labels


def _accepted_touches(controlled_slot, controlled_team, kind, dwell_frames, gap_frames):
    """(stick_frame, team) for touches one slot actually keeps.

    A hole shorter than gap_frames stays inside the touch. A different slot,
    a dead ball, or an ungraded frame ends it. The stick frame is the moment
    the touch has lasted dwell_frames, not the first instant of contact.
    """
    n = len(controlled_slot)
    touches = []
    index = 0
    while index < n:
        if kind[index] != "P" or controlled_slot[index] < 0:
            index += 1
            continue
        slot = int(controlled_slot[index])
        team = int(controlled_team[index])
        start = index
        last = index
        count = 1
        hole = 0
        cursor = index + 1
        while cursor < n and kind[cursor] == "P":
            if controlled_slot[cursor] == slot:
                count += 1
                last = cursor
                hole = 0
                cursor += 1
                continue
            if controlled_slot[cursor] >= 0:
                break
            hole += 1
            if hole > gap_frames:
                break
            cursor += 1
        if count >= dwell_frames:
            seen = 0
            stick = start
            for frame in range(start, last + 1):
                if controlled_slot[frame] == slot:
                    seen += 1
                    if seen >= dwell_frames:
                        stick = frame
                        break
            touches.append((stick, team))
        index = last + 1
    return touches


def possess_from_touches(ball_present, ball_out, nearest_dist, nearest_team, nearest_slot,
                         nearest_speed, second_dist, second_team, ball_speed,
                         set_piece, restart_team, ungraded, pre_kickoff,
                         dwell_frames, gap_frames):
    """Possession is the chain of receptions, not the nearest body each frame.

    The ball stays with the team of the last player who kept it. A pass, a
    loose ball, and a touch too short to be a reception do not change that.
    The other team takes over on the frame their own reception sticks.
    """
    n = len(ball_present)
    kind = np.full(n, "P", dtype="<U1")
    controlled_team = np.full(n, -1, dtype=np.int8)
    controlled_slot = np.full(n, -1, dtype=np.int16)
    dwell_frames = max(1, int(dwell_frames))
    gap_frames = max(0, int(gap_frames))

    for index in range(n):
        if ungraded[index]:
            kind[index] = "X"
            continue
        if pre_kickoff[index] or set_piece[index] or ball_out[index]:
            kind[index] = "D"
            continue
        if not ball_present[index]:
            continue
        team = int(nearest_team[index])
        other_dist = second_dist[index] if second_dist is not None else np.nan
        other_team = int(second_team[index]) if second_team is not None else -1
        speed = ball_speed[index] if ball_speed is not None else np.nan
        player_speed = nearest_speed[index] if nearest_speed is not None else np.nan
        if _under_control(
            nearest_dist[index], team, speed, player_speed, other_dist, other_team,
        ):
            controlled_team[index] = team
            controlled_slot[index] = int(nearest_slot[index])

    touches = _accepted_touches(
        controlled_slot, controlled_team, kind, dwell_frames, gap_frames,
    )
    labels = np.empty(n, dtype="<U1")
    possessor = -1
    touch_at = 0
    for index in range(n):
        if kind[index] == "X":
            labels[index] = "X"
            possessor = -1
            continue
        if kind[index] == "D":
            labels[index] = "D"
            restart = int(restart_team[index])
            possessor = restart if restart >= 0 else -1
            continue
        restart = int(restart_team[index])
        if restart >= 0:
            possessor = restart
        while touch_at < len(touches) and touches[touch_at][0] <= index:
            possessor = touches[touch_at][1]
            touch_at += 1
        labels[index] = _letter(possessor)
    return labels


def decide(ball_present, ball_out, nearest_dist, nearest_team,
           set_piece, restart_team, ungraded, pre_kickoff,
           control_radius_m, dwell_frames, gap_frames=0, ball_speed=None,
           max_control_speed=None, nearest_speed=None, second_dist=None,
           second_team=None, nearest_slot=None, pass_frames=0):
    """Label every frame. Arrays are the same length. Possessor codes are 0/1.

    With a slot id, possession is the chain of receptions: one player keeping
    the ball, then whoever receives the pass. Without it, control falls back
    to distance inside control_radius_m. A challenge survives gap_frames of a
    loose or missing ball. The new team starts when the reception sticks.
    """
    if nearest_slot is not None:
        return possess_from_touches(
            ball_present, ball_out, nearest_dist, nearest_team, nearest_slot,
            nearest_speed, second_dist, second_team, ball_speed,
            set_piece, restart_team, ungraded, pre_kickoff,
            dwell_frames, gap_frames,
        )
    n = len(ball_present)
    labels = np.empty(n, dtype="<U1")
    possessor = -1
    dwell_team = -1
    dwell_count = 0
    gap_left = 0
    dwell_frames = max(1, int(dwell_frames))
    gap_frames = max(0, int(gap_frames))
    secure = np.full(n, -1, dtype=np.int8)

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
            if close:
                secure[i] = team
        elif ball_present[i] and not in_flight:
            other_dist = second_dist[i] if second_dist is not None else np.nan
            other_team = int(second_team[i]) if second_team is not None else -1
            controlled = _under_control(
                dist, team, speed, nearest_speed[i], other_dist, other_team,
            )
            if controlled:
                secure[i] = team
            close = controlled and (
                possessor < 0 or team == possessor
                or _has_received(speed, nearest_speed[i])
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

    return bridge_passes(labels, secure, pass_frames)


def bridge_passes(labels, secure, max_gap):
    """A pass between two controlled touches by one team stays with that team.

    The opponent must not have controlled the ball in between, and the gap
    must not contain a dead ball or an ungraded frame.
    """
    if max_gap <= 0:
        return labels
    labels = np.array(labels, dtype="<U1", copy=True)
    held = np.flatnonzero(np.asarray(secure) >= 0)
    for left, right in zip(held[:-1], held[1:]):
        if int(secure[left]) != int(secure[right]):
            continue
        if right - left - 1 > max_gap or right <= left + 1:
            continue
        segment = labels[left + 1:right]
        if np.any((segment == "D") | (segment == "X")):
            continue
        labels[left + 1:right] = _letter(int(secure[left]))
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
