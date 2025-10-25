import torch
import pytest

from submodlib.sub_modularfunctions.regular_submod_functions.graph_cut import GraphCut


def _make_graph_cut(lambda_val=0.5):
    """Helper returning a small GraphCut instance with a fixed kernel."""
    GraphCut.num_neighbors = None  # attributes referenced inside __init__
    GraphCut.separate_rep = False
    GraphCut.data_rep = None

    sijs = torch.tensor(
        [
            [1.0, 0.2, 0.4],
            [0.2, 1.0, 0.3],
            [0.4, 0.3, 1.0],
        ],
        dtype=torch.float32,
    )
    return GraphCut(n=3, mode="dense", lambda_val=lambda_val, ggsijs=sijs, metric="cosine")


def test_graph_cut_evaluate_matches_formula():
    gc_obj = _make_graph_cut(lambda_val=0.5)
    subset = {0, 2}

    value = gc_obj.evaluate(subset)
    idx = torch.tensor(sorted(subset), dtype=torch.long)
    representation = gc_obj.sijs[:, idx].sum()
    diversity = gc_obj.sijs[idx][:, idx].sum()
    expected = representation - gc_obj.lambda_val * diversity
    assert value == pytest.approx(expected.item())


def test_graph_cut_marginal_gain_matches_naive_difference():
    gc_obj = _make_graph_cut(lambda_val=0.3)
    subset = {0}
    candidate = 1

    gain = gc_obj.marginalGain(subset, candidate)
    value_without = gc_obj.evaluate(subset)
    value_with = gc_obj.evaluate(subset | {candidate})
    assert gain == pytest.approx(value_with - value_without)


def test_higher_lambda_penalizes_diversity_more():
    subset = {0, 1}
    gc_low = _make_graph_cut(lambda_val=0.1)
    gc_high = _make_graph_cut(lambda_val=0.9)

    assert gc_high.evaluate(subset) < gc_low.evaluate(subset)
