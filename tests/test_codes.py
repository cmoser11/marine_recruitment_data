import numpy as np
import pytest

from marine_recruitment_data.codes import (alpha_bravo, meets_tier_minimum, mental_group,
                                           parse_education_code)


@pytest.mark.parametrize("code,expected", [
    ("12L", ("TradHSDiploma", 1, 12)),
    ("11S", ("HSSenior", 1, 11)),
    ("16K", ("Baccalaureate", 1, 16)),
    ("10E", ("GED", 2, 10)),
    ("14D", ("Associate", 1, 14)),
    ("138", ("OneSemCollege", 1, 13)),
    ("101", ("LessThanHS", 3, 10)),
])
def test_known_codes(code, expected):
    assert parse_education_code(code, 60) == expected


@pytest.mark.parametrize("raw,years", [("98", 9), ("88", 8), ("9E", 9)])
def test_zero_padding(raw, years):
    """Exports drop the leading zero: 098 arrives as 98."""
    credential, tier, got = parse_education_code(raw, 60)
    assert got == years and credential != "UNKNOWN"


def test_home_school_tier_depends_on_afqt():
    assert parse_education_code("12H", 60)[1] == 1
    assert parse_education_code("12H", 40)[1] == 2


def test_mental_groups():
    assert [mental_group(x) for x in (95, 70, 55, 40, 25)] == ["I", "II", "IIIA", "IIIB", "IV"]


def test_alpha_boundary_is_50_not_51():
    assert alpha_bravo(50) == "ALPHA"
    assert alpha_bravo(49) == "BRAVO"
    assert alpha_bravo(30) == "BELOW"


def test_tier_aware_minimum_differs_from_flat_31():
    # EPM Table 3-6: Tier I minimum is 21, so a 25 qualifies where a flat 31 would not
    assert meets_tier_minimum(25, 1) is True
    assert meets_tier_minimum(25, 2) is False
    assert meets_tier_minimum(40, 3) is False


def test_unknown_and_missing():
    assert parse_education_code("12Z", 60)[0] == "UNKNOWN"
    assert parse_education_code(None)[0] is None
    assert mental_group(np.nan) is None
