"""EPM code decoding (MCRCO 1100.1, Enlistment Processing Manual).

Education codes are compositional, not a lookup list: the first two digits are
academic years completed, the third character is the credential category, which
determines the education tier.  See Appendix C and Table 3-6.
"""
from __future__ import annotations

import numpy as np

# third character -> (credential name, tier).  Tier None means "depends on AFQT".
CREDENTIAL = {
    "1": ("LessThanHS", 3),
    "5": ("OtherNonTrad", 2),
    "7": ("Correspondence", 2),
    "8": ("OneSemCollege", 1),
    "B": ("AdultAltDiploma", 1),
    "C": ("OccupCert", 2),
    "D": ("Associate", 1),
    "E": ("GED", 2),
    "F": ("ExitExamFail", 1),
    "G": ("NursingDiploma", 1),
    "H": ("HomeSchool", None),   # Tier 1 if AFQT >= 50 else Tier 2
    "J": ("CertAttendance", 2),
    "K": ("Baccalaureate", 1),
    "L": ("TradHSDiploma", 1),
    "M": ("NearCompletion", 1),
    "N": ("Masters", 1),
    "R": ("PostMasters", 1),
    "S": ("HSSenior", 1),
    "U": ("Doctorate", 1),
    "W": ("FirstProf", 1),
    "X": ("NGYCP_GED", 2),
}

# EPM para 3-6(7): DoD mental group categories.
MENTAL_GROUPS = [("I", 93, 99), ("II", 65, 92), ("IIIA", 50, 64), ("IIIB", 31, 49), ("IV", 0, 30)]

# EPM Table 3-6, minimum AFQT by tier.  Waivers are not authorized.
TIER_MIN_AFQT = {1: 21, 2: 31, 3: 50}

ALPHA_CUTOFF = 50   # alpha = mental group I-IIIA
PASS_CUTOFF = 31    # below this is MG-IV, needing district CO administrative review


def parse_education_code(code, afqt=None):
    """Return (credential, tier, years) for an EPM education code.

    Codes are zero-padded to three characters first: exports drop the leading
    zero, so ``98`` arrives for ``098``.
    """
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return (None, None, None)
    c = str(code).strip().upper().zfill(3)
    if len(c) != 3 or not c[:2].isdigit() or c[2] not in CREDENTIAL:
        return ("UNKNOWN", None, None)
    credential, tier = CREDENTIAL[c[2]]
    if tier is None:  # home school: tier depends on AFQT
        tier = 1 if (afqt is not None and not np.isnan(afqt) and afqt >= 50) else 2
    return (credential, tier, int(c[:2]))


def mental_group(afqt):
    if afqt is None or np.isnan(afqt):
        return None
    for name, lo, hi in MENTAL_GROUPS:
        if lo <= afqt <= hi:
            return name
    return None


def alpha_bravo(afqt):
    """Alpha = mental group I-IIIA (AFQT >= 50); Bravo = IIIB (31-49)."""
    if afqt is None or np.isnan(afqt):
        return None
    if afqt >= ALPHA_CUTOFF:
        return "ALPHA"
    return "BRAVO" if afqt >= PASS_CUTOFF else "BELOW"


def meets_tier_minimum(afqt, tier):
    """EPM Table 3-6 tier-aware minimum, as distinct from the flat 31 rule."""
    if afqt is None or np.isnan(afqt) or tier is None:
        return None
    return bool(afqt >= TIER_MIN_AFQT[int(tier)])
