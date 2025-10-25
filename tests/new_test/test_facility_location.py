import torch
import pytest

from submodlib.sub_modularfunctions.regular_submod_functions.facility_location import FacilityLocation


def _make_facility_location():
    """Helper that returns a FacilityLocation instance with a tiny 3x3 kernel."""
    sijs = torch.tensor(
        [
            [1.0, 0.2, 0.4],
            [0.2, 1.0, 0.3],
            [0.4, 0.3, 1.0],
        ],
        dtype=torch.float32,
    )
    return FacilityLocation(n=3, mode="dense", sijs=sijs, device="cpu")


def test_evaluate_matches_manual_computation():
    """The evaluate result should equal the manual sum of per-master maxima."""
    fl = _make_facility_location()
    subset = {0, 2}
    eval_value = fl.evaluate(subset)

    indices = torch.tensor(list(subset), dtype=torch.long)
    expected = torch.max(fl.sijs[:, indices], dim=1)[0].sum().item()

    assert eval_value == pytest.approx(expected)


def test_memoization_consistency_with_naive_calculation():
    """Memoized marginal gain and evaluation must match the naive versions."""
    fl = _make_facility_location()
    subset = {0}
    candidate = 1

    fl.setMemoization(subset)
    gain_memo = fl.marginalGainWithMemoization(subset, candidate)
    gain_naive = fl.marginalGain(subset, candidate)
    assert gain_memo == pytest.approx(gain_naive)

    fl.updateMemoization(subset, candidate)
    subset_with_candidate = subset | {candidate}
    assert fl.evaluateWithMemoization(subset_with_candidate) == pytest.approx(
        fl.evaluate(subset_with_candidate)
    )
