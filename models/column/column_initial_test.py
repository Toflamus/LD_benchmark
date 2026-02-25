from __future__ import annotations

"""Column random-initialization points for benchmark runs.

This module is intentionally *import-safe* (no solver runs at import time).

`POINT_DICT` stores points in *tray-number space* (physical tray numbers).
GDPopt LDSDA/LDBD expects `starting_point` in *external-variable index space*.
For the NT=17 column model in `gdp_column.py`, interior trays are 2..16
(15 candidates). The external-variable index corresponding to tray `t` is
`t - 1`.

So: tray-point (14, 5) -> starting_point (13, 4).
"""

from typing import Iterable

# Keys are stable identifiers for initialization points.
# Values are (reflux_tray, boilup_tray) as physical tray numbers.
POINT_DICT: dict[int, tuple[int, int]] = {
    1: (14, 5),
    2: (14, 4),
    3: (13, 5),
    4: (15, 5),
    5: (14, 6),
    6: (13, 4),
    7: (15, 4),
    8: (13, 6),
    9: (15, 6),
    10: (14, 3),
    11: (12, 5),
    12: (16, 5),
    13: (14, 7),
    14: (13, 3),
    15: (15, 3),
    16: (12, 4),
    17: (16, 4),
    18: (12, 6),
    19: (16, 6),
    20: (13, 7),
    21: (15, 7),
    22: (12, 3),
    23: (16, 3),
    24: (12, 7),
    25: (16, 7),
    26: (14, 2),
    27: (11, 5),
    28: (14, 8),
    29: (13, 2),
    30: (15, 2),
    31: (11, 4),
    32: (11, 6),
    33: (13, 8),
    34: (15, 8),
    35: (12, 2),
    36: (16, 2),
    37: (11, 3),
    38: (11, 7),
    39: (12, 8),
    40: (16, 8),
    41: (10, 5),
    42: (14, 9),
    43: (10, 4),
    44: (10, 6),
    45: (13, 9),
    46: (15, 9),
    47: (11, 2),
    48: (11, 8),
    49: (10, 3),
    50: (10, 7),
    51: (12, 9),
    52: (16, 9),
    53: (10, 2),
    54: (10, 8),
    55: (11, 9),
    56: (14, 10),
    57: (13, 10),
    58: (15, 10),
    59: (12, 10),
    60: (16, 10),
    61: (10, 9),
    62: (11, 10),
    63: (10, 10),
}


def tray_point_to_starting_point(tray_point: tuple[int, int]) -> tuple[int, int]:
    reflux_tray, boilup_tray = tray_point
    return reflux_tray - 1, boilup_tray - 1


def iter_starting_points(keys: Iterable[int] | None = None):
    for k in (POINT_DICT.keys() if keys is None else keys):
        yield k, tray_point_to_starting_point(POINT_DICT[int(k)])
