import pandas as pd
import pytest

from marine_recruitment_data.ingest import IDENTITY_KEY, to_applicants


def _row(**kw):
    base = dict(rss="MESA", test_date=pd.Timestamp("2025-03-19 12:41"), afqt=51.0,
                race="WHITE", school="PIEDRA VISTA HIGH SCHOOL", source="AREA CANVASSING",
                zipcode="87401", edu_contract=None, edu_code="12L", has_timestamp=True)
    base.update(kw)
    return base


@pytest.fixture
def repeat_group():
    """One applicant reported once per lead source; only two rows carry the contract code."""
    return pd.DataFrame([
        _row(source="AREA CANVASSING"),
        _row(source="AREA CANVASSING", edu_contract="12L"),
        _row(source="ELECTRONIC MEDIUM"),
        _row(source="LIST CANVASS"),
        _row(source="POOLEE"),
        _row(source="PRIORITY PROSPECT CARD", edu_contract="12L"),
    ])


def test_collapses_to_one_applicant(repeat_group):
    out = to_applicants(repeat_group)
    assert len(out) == 1
    assert out.n_rows.iloc[0] == 6
    assert out.n_sources.iloc[0] == 5


def test_contract_code_is_coalesced_not_first_row(repeat_group):
    """First-row-wins would drop the contract code and mislabel the applicant."""
    out = to_applicants(repeat_group)
    assert out.edu_contract.iloc[0] == "12L"
    assert bool(out.contracted.iloc[0]) is True


def test_applicant_id_is_stable_and_unique(repeat_group):
    a = to_applicants(repeat_group).applicant_id.iloc[0]
    b = to_applicants(repeat_group.iloc[::-1]).applicant_id.iloc[0]
    assert a == b, "id must not depend on row order"
    assert a.startswith("MRD-")


def test_distinct_people_stay_separate():
    df = pd.DataFrame([_row(), _row(afqt=72.0)])
    assert len(to_applicants(df)) == 2


def test_derived_outcomes():
    out = to_applicants(pd.DataFrame([_row(afqt=50.0)]))
    assert out.alpha.iloc[0] == 1 and out.passed.iloc[0] == 1
    assert out.mental_group.iloc[0] == "IIIA"
