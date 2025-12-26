"""Tests for school name matching utilities."""


def test_fuzzy_match_exact():
    """Test exact match scores 100."""
    from mshp_gap.matching import fuzzy_match_score
    assert fuzzy_match_score("PS 1", "PS 1") == 100


def test_fuzzy_match_abbreviation():
    """Test P.S. vs PS matching."""
    from mshp_gap.matching import standardize_school_name
    assert standardize_school_name("P.S. 1") == standardize_school_name("PS 1")

