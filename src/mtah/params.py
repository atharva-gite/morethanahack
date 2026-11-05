"""Football priors. These are not fitted on the public five minutes.

Changing them because the public spellF1 moved is the failure mode in
EVALUATION.md: baseline_smooth125 wins that window and loses the match.
"""

# Ball at a player's feet. Nearer than "whoever is closest on the pitch".
FEET_RADIUS_M = 1.5

# A reception: the ball has slowed and a player is close, before it is at his feet.
RECEPTION_RADIUS_M = 2.5

# Both teams inside this radius means a duel. Possession stays where it was.
CONTEST_RADIUS_M = 2.0

# The nearest opponent must be at least this much farther than the player on
# the ball. A presser standing next to a pass does not take possession.
ISOLATION_GAP_M = 1.0

# Dribbling: ball speed and the controlling player's speed agree, in m/s.
SPEED_MATCH_MPS = 4.0

# A slow ball next to a player is a reception even if the speeds do not match yet.
SLOW_BALL_MPS = 7.0

# Frames of real control before a change is committed. 10 frames is 0.4 s.
RECEPTION_FRAMES = 10

# A brief loose or missing ball does not cancel a reception already underway.
CONTROL_GAP_FRAMES = 10

# A spell shorter than a second, with the same team on both sides or dying
# straight into a stoppage, is a touch. One second at 25 fps.
TOUCH_FRAMES = 25

# Tracking noise puts an in-play ball just past the line. A metre does not.
OUT_OF_PITCH_MARGIN_M = 1.0

# Published in FAQ.md and DATA.md. Boundaries, not labels learned from video.
FIRST_HALF_KICKOFF = 11185
SECOND_HALF_KICKOFF = 113742

SET_PIECE_CODE = "set pieces"
KICKOFF_TYPES = ("kick off (start)", "kick off (after goal)")
