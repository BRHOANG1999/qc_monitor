"""Pre-ictal CWT sweep engine (thesis subsystem).

Discovers the time-scale at which the pre-ictal feature trajectory is cleanest,
with leave-one-seizure-out CV + a surrogate null so a finding is a result, not
an artifact. Heavy compute runs in a background worker (``worker.py``); results
land in versioned SQLite (``preictal_*`` tables) + a BIDS ``derivatives/``
pocket on disk. See the staged plan in the repo notes.
"""
