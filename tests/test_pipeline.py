"""State machine, tagging, and archive selection. No match data required."""

import os
import sys
import tempfile
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from mtah.fetch import WANTED, choose_members, extract_archive  # noqa: E402
from mtah.regions import make_region_masks  # noqa: E402
from mtah.signals import read_signals  # noqa: E402
from mtah.state import absorb_touches, bridge_passes, decide, possess_from_arrivals  # noqa: E402
from mtah.tags import masks_from_events, side_of_team_tag  # noqa: E402


def _blank(n):
    return {
        "ball_present": np.zeros(n, dtype=bool),
        "ball_out": np.zeros(n, dtype=bool),
        "nearest_dist": np.full(n, np.nan),
        "nearest_team": np.full(n, -1, dtype=np.int8),
        "set_piece": np.zeros(n, dtype=bool),
        "restart": np.full(n, -1, dtype=np.int8),
        "ungraded": np.zeros(n, dtype=bool),
        "pre": np.zeros(n, dtype=bool),
    }


def _run(signals, dwell=3, radius=2.5, gap=0):
    return decide(
        signals["ball_present"], signals["ball_out"], signals["nearest_dist"],
        signals["nearest_team"], signals["set_piece"], signals["restart"],
        signals["ungraded"], signals["pre"], radius, dwell, gap,
    )


def test_regions_match_published_bounds():
    ungraded, pre = make_region_masks(199319)
    assert ungraded[0] and ungraded[11184]
    assert not ungraded[11185] and not ungraded[90651]
    assert ungraded[90652] and ungraded[110807]
    assert not ungraded[110808] and not ungraded[190197]
    assert ungraded[190198] and ungraded[199318]
    assert pre[110808] and pre[113741]
    assert not pre[113742] and not pre[11185]
    assert not ungraded[113742]


def test_missing_ball_holds_the_last_team():
    n = 8
    s = _blank(n)
    s["ball_present"][:] = True
    s["nearest_team"][:] = 0
    s["nearest_dist"][:] = 0.5
    s["ball_present"][5:] = False
    labels = _run(s, dwell=2)
    assert list(labels) == ["X"] + ["H"] * (n - 1)


def test_one_frame_touch_does_not_flip():
    n = 8
    s = _blank(n)
    s["ball_present"][:] = True
    s["nearest_team"][:] = 0
    s["nearest_dist"][:] = 1.0
    s["nearest_team"][5] = 1
    labels = _run(s, dwell=3)
    assert set(labels[3:]) == {"H"}


def test_dwell_hands_possession_over():
    n = 8
    s = _blank(n)
    s["ball_present"][:] = True
    s["nearest_team"][:4] = 0
    s["nearest_team"][4:] = 1
    s["nearest_dist"][:] = 1.0
    labels = _run(s, dwell=3)
    assert list(labels) == ["X", "X", "H", "H", "H", "H", "A", "A"]


def test_loose_ball_does_not_change_team():
    n = 6
    s = _blank(n)
    s["ball_present"][:] = True
    s["nearest_team"][:] = 0
    s["nearest_dist"][:3] = 1.0
    s["nearest_dist"][3:] = 9.0
    s["nearest_team"][3:] = 1
    labels = _run(s, dwell=2, radius=2.5)
    assert list(labels[2:]) == ["H", "H", "H", "H"]


def test_out_of_pitch_and_set_piece_are_dead():
    n = 5
    s = _blank(n)
    s["ball_present"][:] = True
    s["nearest_team"][:] = 0
    s["nearest_dist"][:] = 0.2
    s["ball_out"][2] = True
    s["set_piece"][3] = True
    labels = _run(s, dwell=1)
    assert list(labels) == ["H", "H", "D", "D", "H"]


def test_named_restart_overrides_last_touch_and_na_does_not():
    n = 6
    s = _blank(n)
    s["ball_present"][:2] = True
    s["nearest_team"][:2] = 0
    s["nearest_dist"][:2] = 0.2
    s["set_piece"][2:4] = True
    # Ball is missing after the stoppage, so only the restart can change team.
    s["restart"][4] = 1
    labels = _run(s, dwell=1)
    assert list(labels) == ["H", "H", "D", "D", "A", "A"]

    s["restart"][4] = -1
    labels = _run(s, dwell=1)
    assert list(labels) == ["H", "H", "D", "D", "H", "H"]


def test_a_pass_stays_with_the_team_until_someone_receives_it():
    n = 12
    present = np.ones(n, dtype=bool)
    out = np.zeros(n, dtype=bool)
    dist = np.ones(n)
    team = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.int8)
    slot = np.array([3, 3, 3, 3, 8, 8, 4, 4, 4, 4, 4, 4], dtype=np.int16)
    speed = np.full(n, 3.0)
    second_dist = np.full(n, 8.0)
    second_team = np.ones(n, dtype=np.int8)
    piece = np.zeros(n, dtype=bool)
    restart = np.full(n, -1, dtype=np.int8)
    ungraded = np.zeros(n, dtype=bool)
    pre = np.zeros(n, dtype=bool)
    labels = decide(
        present, out, dist, team, piece, restart, ungraded, pre,
        2.5, 3, 1, speed, None, speed, second_dist, second_team, slot,
    )
    # Home keeps it through frame 2. Away's two frames are not a reception.
    # The next home player receives on the third frame of that touch.
    assert list(labels[:6]) == ["X", "X", "H", "H", "H", "H"]
    assert list(labels[8:]) == ["H", "H", "H", "H"]


def _arrival(n, dwell=6):
    present = np.ones(n, dtype=bool)
    out = np.zeros(n, dtype=bool)
    ball_speed = np.full(n, 3.0)
    home_dist = np.full(n, 8.0)
    away_dist = np.full(n, 8.0)
    home_speed = np.full(n, 3.0)
    away_speed = np.full(n, 3.0)
    piece = np.zeros(n, dtype=bool)
    restart = np.full(n, -1, dtype=np.int8)
    ungraded = np.zeros(n, dtype=bool)
    pre = np.zeros(n, dtype=bool)
    return (present, out, ball_speed, home_dist, away_dist, home_speed, away_speed,
            piece, restart, ungraded, pre, dwell, 2)


def test_a_pass_between_teammates_is_one_spell():
    fields = list(_arrival(24))
    home_dist, away_dist = fields[3], fields[4]
    home_dist[:8] = 1.0
    home_dist[16:] = 1.0
    away_dist[8:12] = 1.2
    fields[5][8:12] = 0.0  # home player speed irrelevant
    fields[2][8:12] = 11.0  # ball still traveling
    fields[6][8:12] = 1.0  # standing opponent
    labels = possess_from_arrivals(*fields)
    assert list(labels[5:]) == ["H"] * 19


def test_a_brief_press_is_not_a_reception():
    fields = list(_arrival(16))
    fields[3][:8] = 1.0
    fields[4][8:11] = 1.0
    fields[6][8:11] = 3.0
    labels = possess_from_arrivals(*fields)
    assert list(labels[5:]) == ["H"] * 11


def test_a_kept_ball_changes_team_when_it_sticks():
    fields = list(_arrival(16, dwell=4))
    fields[3][:6] = 1.0
    fields[4][6:] = 1.0
    labels = possess_from_arrivals(*fields)
    assert list(labels[:3]) == ["X", "X", "X"]
    assert list(labels[3:9]) == ["H"] * 6
    assert list(labels[9:]) == ["A"] * 7


def test_a_fast_constant_pass_does_not_change_team():
    fields = list(_arrival(20, dwell=6))
    fields[3][:8] = 1.0
    fields[4][8:] = 1.0
    fields[2][8:] = 6.0
    fields[6][8:] = 0.0
    labels = possess_from_arrivals(*fields)
    assert "A" not in list(labels)


def test_a_slowing_ball_is_a_reception():
    fields = list(_arrival(24, dwell=6))
    fields[3][:8] = 1.0
    fields[2][:8] = 12.0
    fields[4][8:] = 1.0
    fields[2][8:] = 3.0
    fields[6][8:] = 3.0
    labels = possess_from_arrivals(*fields)
    assert "A" in list(labels[12:])


def test_a_stopped_ball_far_from_players_is_dead():
    fields = list(_arrival(40))
    fields[3][:] = 12.0
    fields[4][:] = 12.0
    fields[2][:] = 0.4
    labels = possess_from_arrivals(*fields)
    assert list(labels[:25]) == ["D"] * 25


def test_a_missing_ball_is_not_dead():
    fields = list(_arrival(40))
    fields[0][:] = False
    labels = possess_from_arrivals(*fields)
    assert "D" not in list(labels)


def test_a_pass_between_one_teams_touches_stays_with_them():
    labels = np.array(list("HHHAAAAHHH"))
    secure = np.full(10, -1, dtype=np.int8)
    secure[0:3] = 0
    secure[7:10] = 0
    assert "".join(bridge_passes(labels, secure, 10)) == "H" * 10


def test_a_real_reception_between_two_passes_is_kept():
    labels = np.array(list("HHHAAAAHHH"))
    secure = np.full(10, -1, dtype=np.int8)
    secure[0:3] = 0
    secure[3:7] = 1
    secure[7:10] = 0
    assert "".join(bridge_passes(labels, secure, 10)) == "HHHAAAAHHH"


def test_a_pass_does_not_cross_a_dead_ball():
    labels = np.array(list("HHHDDDHHH"))
    secure = np.full(9, -1, dtype=np.int8)
    secure[0:3] = 0
    secure[6:9] = 0
    assert "".join(bridge_passes(labels, secure, 10)) == "HHHDDDHHH"


def test_a_gap_longer_than_a_touch_is_not_a_pass():
    labels = np.array(list("HHH" + "A" * 40 + "HHH"))
    secure = np.full(len(labels), -1, dtype=np.int8)
    secure[:3] = 0
    secure[-3:] = 0
    assert "".join(bridge_passes(labels, secure, 25)) == "HHH" + "A" * 40 + "HHH"


def test_brief_touch_between_the_same_team_disappears():
    labels = np.array(list("HHHHHHHAAHHHHHHH"))
    assert "".join(absorb_touches(labels, min_frames=5)) == "H" * 16


def test_a_stoppage_stays_and_the_touch_after_it_does_not():
    labels = np.array(list("HHHHHDDDDAAHHHHH"))
    assert "".join(absorb_touches(labels, min_frames=5)) == "HHHHHDDDDHHHHHHH"


def test_a_short_lead_in_joins_the_spell_that_keeps_the_ball():
    labels = np.array(list("HHAAAAAAAAAA"))
    assert "".join(absorb_touches(labels, min_frames=5)) == "A" * 12


def test_a_touch_that_dies_at_a_stoppage_returns_to_the_holder():
    labels = np.array(list("HHHHHHHHAAADDDD"))
    assert "".join(absorb_touches(labels, min_frames=5)) == "HHHHHHHHHHHDDDD"


def test_a_short_gap_does_not_cancel_control():
    n = 9
    s = _blank(n)
    s["ball_present"][:] = True
    s["nearest_dist"][:] = 1.0
    s["nearest_team"][:4] = 0
    s["nearest_team"][4:] = 1
    s["ball_present"][6] = False
    held = _run(s, dwell=4, gap=2)
    assert list(held) == ["X", "X", "X", "H", "H", "H", "H", "H", "A"]
    reset = _run(s, dwell=4, gap=0)
    assert list(reset[4:]) == ["H"] * 5


def test_a_pass_in_flight_does_not_change_team():
    n = 6
    s = _blank(n)
    s["ball_present"][:] = True
    s["nearest_dist"][:] = 1.0
    s["nearest_team"][:] = 0
    s["nearest_team"][2:] = 1
    speed = np.zeros(n)
    speed[2:] = 25.0
    labels = decide(
        s["ball_present"], s["ball_out"], s["nearest_dist"], s["nearest_team"],
        s["set_piece"], s["restart"], s["ungraded"], s["pre"],
        2.5, 2, 0, speed, 15.0,
    )
    assert list(labels[1:]) == ["H"] * 5


def test_a_duel_does_not_hand_the_ball_over():
    n = 6
    s = _blank(n)
    s["ball_present"][:] = True
    s["nearest_dist"][:] = 1.0
    s["nearest_team"][:2] = 0
    s["nearest_team"][2:] = 1
    speed = np.ones(n)
    second_dist = np.full(n, 8.0)
    second_team = np.zeros(n, dtype=np.int8)
    second_dist[2:] = 1.2
    labels = decide(
        s["ball_present"], s["ball_out"], s["nearest_dist"], s["nearest_team"],
        s["set_piece"], s["restart"], s["ungraded"], s["pre"],
        2.5, 2, 0, speed, None, speed, second_dist, second_team,
    )
    assert list(labels[1:]) == ["H"] * 5


def test_a_slow_ball_at_the_feet_is_control():
    n = 4
    s = _blank(n)
    s["ball_present"][:] = True
    s["nearest_dist"][:] = 1.0
    s["nearest_team"][:] = 1
    ball_speed = np.full(n, 3.0)
    player_speed = np.full(n, 3.0)
    second_dist = np.full(n, 12.0)
    second_team = np.zeros(n, dtype=np.int8)
    labels = decide(
        s["ball_present"], s["ball_out"], s["nearest_dist"], s["nearest_team"],
        s["set_piece"], s["restart"], s["ungraded"], s["pre"],
        2.5, 2, 0, ball_speed, None, player_speed, second_dist, second_team,
    )
    assert list(labels) == ["X", "A", "A", "A"]


def test_a_rolling_ball_past_a_standing_player_is_not_a_reception():
    n = 6
    s = _blank(n)
    s["ball_present"][:] = True
    s["nearest_dist"][:] = 1.0
    s["nearest_team"][:2] = 0
    s["nearest_team"][2:] = 1
    ball_speed = np.full(n, 3.0)
    player_speed = np.full(n, 3.0)
    ball_speed[2:] = 6.0
    player_speed[2:] = 0.0
    second_dist = np.full(n, 8.0)
    second_team = np.ones(n, dtype=np.int8)
    labels = decide(
        s["ball_present"], s["ball_out"], s["nearest_dist"], s["nearest_team"],
        s["set_piece"], s["restart"], s["ungraded"], s["pre"],
        2.5, 2, 0, ball_speed, None, player_speed, second_dist, second_team,
    )
    assert list(labels[1:]) == ["H"] * 5


def test_a_nearby_opponent_does_not_take_a_pass():
    n = 6
    s = _blank(n)
    s["ball_present"][:] = True
    s["nearest_dist"][:] = 1.0
    s["nearest_team"][:2] = 0
    s["nearest_team"][2:] = 1
    speed = np.full(n, 3.0)
    second_dist = np.full(n, 6.0)
    second_team = np.ones(n, dtype=np.int8)
    # After the home player has it, an away player is nearest but a home
    # player is still within a metre of them. That is pressure, not a take.
    second_dist[2:] = 1.8
    second_team[2:] = 0
    labels = decide(
        s["ball_present"], s["ball_out"], s["nearest_dist"], s["nearest_team"],
        s["set_piece"], s["restart"], s["ungraded"], s["pre"],
        2.5, 2, 0, speed, None, speed, second_dist, second_team,
    )
    assert list(labels[1:]) == ["H"] * 5


def test_ungraded_and_pre_kickoff():
    n = 4
    s = _blank(n)
    s["ball_present"][:] = True
    s["nearest_team"][:] = 0
    s["nearest_dist"][:] = 0.0
    s["ungraded"][0] = True
    s["pre"][1] = True
    labels = _run(s, dwell=1)
    assert list(labels[:2]) == ["X", "D"]
    assert labels[2] == "H"


def test_team_tag_and_set_piece_masks():
    assert side_of_team_tag("Barcelona") == 0
    assert side_of_team_tag("Atletic Club") == 1
    assert side_of_team_tag("N/A") is None

    throw = masks_from_events([
        {"code": "SET PIECES", "start_frame": 20, "end_frame": 25,
         "tags": {"Type": "Throw-in", "Team": "N/A"}},
    ], 40)
    assert throw[0][20] and throw[0][24] and not throw[0][25]
    assert throw[1][25] == -1

    corner = masks_from_events([
        {"code": "SET PIECES", "start_frame": 10, "end_frame": 30,
         "tags": {"Type": "Corner Kick", "Team": "Barcelona"}},
    ], 40)
    assert corner[0][10] and corner[0][29] and not corner[0][30]
    assert corner[1][30] == 0

    phase = masks_from_events([
        {"code": "BUILD UP", "start_frame": 0, "end_frame": 50,
         "tags": {"Team": "Atletic Club"}},
    ], 60)
    assert not phase[0].any()
    assert (phase[1] == -1).all()

    kick = masks_from_events([
        {"code": "SET PIECES", "start_frame": 11180, "end_frame": 11190,
         "tags": {"Type": "Kick Off (Start)", "Team": "Barcelona", "Half": "1st Half"}},
    ], 20000)
    assert kick[0][11180] and kick[0][11184]
    assert not kick[0][11185]
    assert kick[1][11185] == 0


def test_read_signals_distance_and_out_of_pitch():
    # Two slots: home at the ball, away far. Then a missing ball. Then out of play.
    lines = [
        "1:0.50,0.50,1.0;0.50,0.80,1.0:0.50,0.50,2.5\n",
        "2:0.50,0.50,1.0;0.50,0.80,1.0:NaN,NaN,NaN\n",
        "3:0.50,0.50,1.0;0.50,0.80,1.0:1.20,0.50,8.0\n",
    ]
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "atd.txt")
        with open(path, "w") as handle:
            handle.writelines(lines)
        signals = read_signals(path, np.array([0, 1], dtype=np.int8), 3, (105.3, 68.0), 1.0)
    assert signals.ball_present.tolist() == [True, False, True]
    assert signals.nearest_team[0] == 0
    assert signals.nearest_dist[0] < 0.01
    assert signals.nearest_speed[0] == 1.0
    assert signals.second_team[0] == 1
    assert signals.second_dist[0] > 10.0
    assert signals.ball_speed[0] == 2.5
    assert not signals.ball_present[1]
    assert signals.ball_out[2]
    assert not signals.ball_out[0]


def test_extract_archive_skips_video():
    with tempfile.TemporaryDirectory() as folder:
        archive_path = os.path.join(folder, "bundle.zip")
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("tactical/BAR-ATH_tactical.mp4", b"video-bytes")
            archive.writestr("broadcast/BAR-ATH_broadcast_atd.txt", b"broadcast")
            for name in WANTED:
                archive.writestr("data/tactical/" + name, b"ok-" + name.encode())
        dest = os.path.join(folder, "out")
        written = extract_archive(archive_path, dest)
        names = sorted(os.listdir(dest))
        assert names == sorted(WANTED)
        assert all(size > 0 for _, size in written)
        assert not any(name.endswith(".mp4") for name in names)


def test_choose_members_prefers_tactical_path():
    names = [
        "broadcast/BAR-ATH_tactical_atd.txt",
        "data/tactical/BAR-ATH_tactical_atd.txt",
    ] + ["data/tactical/" + name for name in WANTED if name != "BAR-ATH_tactical_atd.txt"]
    chosen, missing = choose_members(names)
    assert missing == []
    assert chosen["BAR-ATH_tactical_atd.txt"] == "data/tactical/BAR-ATH_tactical_atd.txt"


def test_team_of_each_track_home_is_first_team():
    from starter.baseline import team_of_each_track
    xml = """<root>
      <Team id="10"><Name>Barcelona</Name></Team>
      <Team id="20"><Name>Atletic Club</Name></Team>
      <Player teamId="10"/><Player teamId="20"/><Player teamId="10"/>
    </root>"""
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "meta.xml")
        with open(path, "w") as handle:
            handle.write(xml)
        teams = team_of_each_track(path)
    assert teams.tolist() == [0, 1, 0]


def test_build_labels_on_a_tiny_bundle():
    """The full 199319-frame path, with a fake bundle that contains no real tracking."""
    from mtah.build import build_labels

    xml_meta = """<root>
      <Team id="10"><Name>Barcelona</Name></Team>
      <Team id="20"><Name>Atletic Club</Name></Team>
      <Player teamId="10"/><Player teamId="20"/>
    </root>"""
    tagging = """<file><instance>
      <code>SET PIECES</code>
      <start>447.0</start>
      <end>447.4</end>
      <label><group>Team</group><text>Barcelona</text></label>
      <label><group>Type</group><text>Kick Off (Start)</text></label>
      <label><group>Half</group><text>1st Half</text></label>
    </instance></file>"""
    with tempfile.TemporaryDirectory() as folder:
        tactical = os.path.join(folder, "tactical")
        os.makedirs(tactical)
        with open(os.path.join(tactical, "BAR-ATH_tactical_atd.xml"), "w") as handle:
            handle.write(xml_meta)
        with open(os.path.join(tactical, "BAR-ATH_tactical_atd.txt"), "w") as handle:
            handle.write("11186:0.50,0.50,1;0.80,0.50,1:0.50,0.50,1\n")
        with open(os.path.join(tactical, "BAR-ATH_tactical_smart_tagging.xml"), "w") as handle:
            handle.write(tagging)
        labels = build_labels(folder)

    assert len(labels) == 199319
    assert labels[0] == "X" and labels[11184] == "X"
    assert labels[11185] == "H"
    assert labels[90651] == "H"
    assert labels[90652] == "X"
    assert labels[110808] == "D" and labels[113741] == "D"
    assert labels[190198] == "X"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all tests passed")
