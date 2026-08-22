import pandas as pd
import pytest

from marine_recruitment_data.features import LEAKY, assert_no_leakage, load_zip_income


def test_zip_table_loads():
    z = load_zip_income()
    assert len(z) > 20000
    assert z.set_index("zipcode").loc["87401", "mean_agi_k"] > 0


def test_leakage_guard_rejects_outcome_derived_columns():
    with pytest.raises(ValueError, match="outcome-derived"):
        assert_no_leakage(["race_g", "contracted"])
    assert_no_leakage(["race_g", "log_agi"])  # clean design passes


def test_contract_code_is_registered_as_leaky():
    # null for 98.8% of sub-31 applicants, so it encodes the outcome
    assert "edu_contract" in LEAKY and "src_poolee" in LEAKY
