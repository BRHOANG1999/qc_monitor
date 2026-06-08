"""Animal id + electrode location parsing for channel names.

Channels in this lab follow the pattern
``[Animal ID][Electrode location]`` — e.g. ``BCH062SLM`` is animal
``BCH062`` with electrode location ``SLM`` (stratum lacunosum-
moleculare). One session can carry several electrodes for the same
animal (e.g. ``BCH062SR`` + ``BCH062SLM``) and the reviewer is free
to pick whichever one looks usable.

The default regex matches "non-digits then digits" as the animal
portion and "the rest" as the electrode location:

  ``BCH040SR``  -> ("BCH040", "SR")
  ``BCH061SLM`` -> ("BCH061", "SLM")
  ``saline``    -> ("saline", None)      # no digits, whole string
  ``test1``     -> ("test1", None)       # no trailing letters

Override the regex via ``config.review_queue.animal_pattern`` if a
different lab convention shows up. The pattern MUST have exactly two
capture groups: ``(animal_id)(electrode_location)``.
"""

from __future__ import annotations

import re
from functools import lru_cache

DEFAULT_PATTERN = r"^(\D*\d+)(\D.*)$"

# stim_copy / saline / test channels never carry an animal record.
# Listed here so the queue + dropdown can skip them even when the
# session_config role tags get stale or absent.
NON_ANIMAL_KEYWORDS: tuple[str, ...] = (
    "stimcopy", "saline", "ref", "reference",
    "ground", "gnd", "blank", "test", "noise",
)


@lru_cache(maxsize=64)
def _compiled(pattern: str) -> re.Pattern:
    return re.compile(pattern)


def split_animal_electrode(name: str | None,
                            pattern: str | None = None
                            ) -> tuple[str, str | None]:
    """Return ``(animal_id, electrode_location_or_None)`` for *name*.

    When *name* doesn't match the regex (no digits, control channel,
    etc.), the whole string is returned as the animal and the
    electrode is ``None``. Empty / falsy input returns ``("", None)``.
    """
    if not name:
        return ("", None)
    pat = _compiled(pattern or DEFAULT_PATTERN)
    m = pat.match(str(name))
    if m and len(m.groups()) >= 2:
        animal = m.group(1)
        electrode = m.group(2) or None
        return (animal, electrode)
    return (str(name), None)


def is_animal_channel(name: str | None) -> bool:
    """True when *name* looks like a real animal recording channel.

    Filters out ``stimCopy`` / ``saline`` / ``ref`` / etc. so the
    queue picker never offers them as a reviewable animal. Match is
    case-insensitive substring.
    """
    if not name:
        return False
    lower = str(name).lower()
    return not any(kw in lower for kw in NON_ANIMAL_KEYWORDS)
