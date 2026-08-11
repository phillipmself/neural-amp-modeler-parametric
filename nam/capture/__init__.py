"""
Guided parametric capture: plan knob settings, reamp through an audio interface, and
keep a resumable project plus a training-ready ``data.json`` up to date.

Submodules are imported explicitly (``nam.capture.params``, ``nam.capture.planner``,
...) rather than re-exported here: several of them pull in heavy dependencies (torch,
sounddevice, the training stack) that the desktop app defers until needed.
"""

# The capture app's own version, hand-maintained. Separate from ``nam.__version__``,
# which is generated from version-control tags and describes the whole package. This
# exists to date a project's behaviour: it is stamped into ``capture_project.json`` when
# a project is created and never changed afterwards, so a project always says which
# version's rules it started under.
#
# Bump this when the rules change, not merely when the code does -- 1.2.0 is where each
# capture began deriving its timebase from its own loopback blip peak instead of sharing
# one measured offset across the project.
#
# It dates a project; it does not describe its captures. A project part-captured across
# a rules change keeps the stamp it was created under while holding captures made under
# both, so this cannot say which timebase any given capture used and must not be used to
# decide. What can say is the project's ``alignment_reference``, a capture's
# ``qa.peak_delay``, and above all ``captures_raw/``, which can be re-measured and cannot
# go stale (see ``nam.capture.session.audit_captures``). Compare as parsed numbers if it
# is ever compared at all: as strings, "1.10.0" sorts below "1.2.0".
CAPTURE_APP_VERSION = "1.2.0"

# The version raw recordings (``captures_raw/``) started being saved in. Fixed forever at
# the version that introduced them, while CAPTURE_APP_VERSION moves on: a project created
# before this has captures with no raw recordings behind them, which is worth saying
# plainly rather than leaving as a mystery about a half-empty folder.
RAW_RECORDING_SINCE_VERSION = "1.1.0"
