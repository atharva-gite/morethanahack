"""Football priors. These are not fitted on the public five minutes.

Changing them because the public spellF1 moved is the failure mode in
EVALUATION.md: baseline_smooth125 wins that window and loses the match.
"""

# A player has the ball at their feet. Nearest-body-on-the-pitch is not control.
CONTROL_RADIUS_M = 2.5

# A few tenths of a second. One frame is a deflection; this is a reception.
# 8 frames is 0.32 s at 25 fps.
DWELL_FRAMES = 8

# CHALLENGE.md: a controlled possession is a player keeping the ball for a few
# seconds, and a touch without that control is not a change. Two seconds at
# 25 fps. A shorter spell is given back to the team that already had it.
MIN_CONTROL_FRAMES = 50

# Tracking noise puts an in-play ball just past the line. A metre does not.
OUT_OF_PITCH_MARGIN_M = 1.0

# Published in FAQ.md and DATA.md. Boundaries, not labels learned from video.
FIRST_HALF_KICKOFF = 11185
SECOND_HALF_KICKOFF = 113742

SET_PIECE_CODE = "set pieces"
KICKOFF_TYPES = ("kick off (start)", "kick off (after goal)")
