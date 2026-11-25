import pytest

from aae.score import noninferior_verdict, paired_diff_ci, same_result, verdict


def test_order_and_names_ignored_unless_ordered():
    assert same_result([(1, "a"), (2, "b")], [(2, "b"), (1, "a")], ordered=False)
    assert not same_result([(1, "a"), (2, "b")], [(2, "b"), (1, "a")], ordered=True)


def test_float_tolerance_and_none():
    assert same_result([(0.1 + 0.2,)], [(0.3,)], ordered=False)
    assert same_result([(None,)], [(None,)], ordered=False)
    assert not same_result([(None,)], [(0,)], ordered=False)


def test_column_count_must_match():
    assert not same_result([(1, 2)], [(1,)], ordered=False)


def test_verdict_three_ways():
    assert verdict(8.0, 2.0, 14.0) == "supported"
    assert verdict(1.0, -2.0, 4.0) == "not supported"
    assert verdict(6.0, -1.0, 13.0) == "inconclusive"


def test_paired_ci_basics():
    d, lo, hi = paired_diff_ci([0, 0, 0, 0] * 25, [1, 1, 0, 0] * 25)
    assert d == pytest.approx(50.0) and lo > 0 and hi < 100
    d, lo, hi = paired_diff_ci([], [])
    assert d != d


def test_noninferior_verdict():
    assert noninferior_verdict(-2.0, -4.0, 0.0, 5.0) == "within margin"
    assert noninferior_verdict(3.0, 1.0, 6.0, 5.0) == "within margin"
    assert noninferior_verdict(-10.0, -14.0, -6.0, 5.0) == "behind"
    assert noninferior_verdict(-5.0, -12.0, 2.0, 5.0) == "inconclusive"
