"""Football priors. These are not fitted on the public five minutes.

Changing them because the public spellF1 moved is the failure mode in
EVALUATION.md: baseline_smooth125 wins that window and loses the match.
"""

# A player has the ball at their feet. Nearest-body-on-the-pitch is not control.
CONTROL_RADIUS_M = 2.5

# How long a new team must actually keep the ball before it is a change.
# CHALLENGE.md calls control "a few seconds". Three seconds at 25 fps, applied
# to every frame, not only to spells that already look short.
MIN_CONTROL_FRAMES = 75

# A one-second hole in the tracking, or a loose ball, does not cancel a
# challenge that is already underway and does not hand the ball over.
CONTROL_GAP_FRAMES = 25

# Passes in flight are not control, even if a player is the nearest body.
MAX_CONTROL_SPEED_MPS = 15.0

# Tracking noise puts an in-play ball just past the line. A metre does not.
OUT_OF_PITCH_MARGIN_M = 1.0

# Published in FAQ.md and DATA.md. Boundaries, not labels learned from video.
FIRST_HALF_KICKOFF = 11185
SECOND_HALF_KICKOFF = 113742

SET_PIECE_CODE = "set pieces"
KICKOFF_TYPES = ("kick off (start)", "kick off (after goal)")
