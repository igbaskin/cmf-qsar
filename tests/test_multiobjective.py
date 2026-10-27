"""A design loop with more than one objective (plan item MC11).

What these tests fix is the arithmetic a chemist would be shown: which candidates are on the
Pareto front and which are dominated, that an objective to be minimised is read the right way
round, that the acquisition adds improvements and not raw values, that a batch is spread by the
posterior covariance as plan item MC2's rule requires, and that the trade reported for a
suggestion names the right gains and the right losses.
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from cmfqsar.multiobjective import (MAXIMISE, MINIMISE, Objective, Trade,  # noqa: E402
                                    acquisition, batch, best_so_far, dominates, improvements,
                                    pareto_front, pareto_layers, report, trade, with_best)

#: potency up, off-target down
OBJ = [Objective("potency", MAXIMISE), Objective("hERG", MINIMISE)]


# ------------------------------------------------------------------- domination


def test_one_candidate_beats_another_only_by_being_better_everywhere():
    signs = [1.0, -1.0]
    assert dominates([9.0, 4.0], [8.0, 5.0], signs)
    assert not dominates([9.0, 6.0], [8.0, 5.0], signs), "better in one, worse in the other"
    assert not dominates([8.0, 5.0], [8.0, 5.0], signs), "the same is not better"
    assert dominates([8.0, 4.0], [8.0, 5.0], signs), "the same in one and better in the other"


def test_the_direction_of_an_objective_is_obeyed():
    up = [Objective("x", MAXIMISE)]
    down = [Objective("x", MINIMISE)]
    v = np.array([[1.0], [2.0]])
    assert list(pareto_front(v, up)) == [1]
    assert list(pareto_front(v, down)) == [0]


# ----------------------------------------------------------------- the front


def test_the_front_and_the_layers_behind_it():
    v = np.array([[9.0, 5.0], [8.0, 4.0], [8.5, 4.5], [7.0, 6.0], [6.0, 7.0]])
    layers = pareto_layers(v, OBJ)
    assert list(layers) == [1, 1, 1, 2, 3]
    assert sorted(pareto_front(v, OBJ)) == [0, 1, 2]


def test_every_candidate_lands_on_exactly_one_layer():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(40, 2))
    layers = pareto_layers(v, OBJ)
    assert layers.min() == 1 and np.all(layers > 0)
    assert len(layers) == 40


def test_candidates_that_are_all_the_same_are_all_on_the_front():
    v = np.ones((5, 2))
    assert len(pareto_front(v, OBJ)) == 5


# ------------------------------------------------------------- the acquisition


def test_the_improvement_is_counted_from_the_best_already_made():
    obj = [Objective("potency", MAXIMISE, best=8.0)]
    v = np.array([[9.0], [7.0]])
    s = np.array([[1e-6], [1e-6]])
    imp = improvements(v, s, obj)
    assert imp[0, 0] == pytest.approx(1.0, abs=1e-3), "a log unit better is a log unit of gain"
    assert imp[1, 0] == pytest.approx(0.0, abs=1e-3), "worse than the best is no gain"


def test_an_objective_to_be_minimised_improves_by_going_down():
    obj = [Objective("hERG", MINIMISE, best=5.0)]
    s = np.array([[1e-6], [1e-6]])
    imp = improvements(np.array([[4.0], [6.0]]), s, obj)
    assert imp[0, 0] == pytest.approx(1.0, abs=1e-3)
    assert imp[1, 0] == pytest.approx(0.0, abs=1e-3)


def test_uncertainty_is_worth_something_even_where_the_mean_is_not():
    obj = [Objective("potency", MAXIMISE, best=8.0)]
    sure = improvements(np.array([[7.0]]), np.array([[0.01]]), obj)[0, 0]
    unsure = improvements(np.array([[7.0]]), np.array([[1.00]]), obj)[0, 0]
    assert unsure > sure, "a candidate that might surprise is worth more than one that cannot"


def test_the_weights_say_what_one_objective_is_worth_against_the_other():
    v = np.array([[9.0, 5.0]])
    s = np.full((1, 2), 1e-6)
    a = [Objective("potency", MAXIMISE, weight=1.0, best=8.0),
         Objective("hERG", MINIMISE, weight=0.0, best=5.0)]
    b = [Objective("potency", MAXIMISE, weight=0.0, best=8.0),
         Objective("hERG", MINIMISE, weight=1.0, best=5.0)]
    assert acquisition(v, s, a)[0] == pytest.approx(1.0, abs=1e-3)
    assert acquisition(v, s, b)[0] == pytest.approx(0.0, abs=1e-3)


def test_the_best_already_made_is_read_off_the_training_values():
    y = np.array([[8.0, 5.0], [9.0, 6.0], [7.0, 4.0]])
    assert best_so_far(y, OBJ) == [9.0, 4.0]
    filled = with_best(OBJ, y)
    assert [o.best for o in filled] == [9.0, 4.0]
    assert [o.name for o in filled] == ["potency", "hERG"]


# ------------------------------------------------------------------ the batch


def test_a_batch_does_not_take_the_same_candidate_twice():
    v = np.array([[9.0, 4.0], [8.9, 4.1], [8.8, 4.2], [5.0, 9.0]])
    obj = with_best(OBJ, np.array([[8.0, 5.0]]))
    covs = [np.eye(4) * 0.25, np.eye(4) * 0.25]
    got = batch(v, covs, obj, 3)
    assert len(got) == 3 and len(set(got)) == 3


def test_the_covariance_spreads_a_batch_as_the_batch_rule_of_mc2_asks():
    """Two candidates the model cannot tell apart must not both be ordered.

    The conditioning acts on the uncertainty, so the test puts every candidate at the best
    already made: the mean part of the improvement is then nil for all three and what is left
    is what the model does not yet know.
    """
    v = np.array([[8.0, 5.0], [8.0, 5.0], [8.0, 5.0]])
    obj = with_best(OBJ, np.array([[8.0, 5.0]]))
    twins = np.array([[1.0, 0.999, 0.0], [0.999, 1.0, 0.0], [0.0, 0.0, 1.0]])
    got = batch(v, [twins.copy(), twins.copy()], obj, 2)
    assert 2 in got, "the third candidate is the only one that adds anything"


def test_the_improvement_a_candidate_promises_is_not_taken_away_by_its_twin():
    """The other side of the same coin, and worth fixing so that it is not read as a bug.

    Two compounds that really are a log unit better than anything made are both worth making,
    whatever the model knows about them; conditioning removes the uncertainty and not the
    promise.  A rule that spread the batch on the means instead would be refusing a compound
    for being like a good one.
    """
    v = np.array([[9.0, 4.0], [9.0, 4.0], [8.5, 4.5]])
    obj = with_best(OBJ, np.array([[8.0, 5.0]]))
    twins = np.array([[1.0, 0.999, 0.0], [0.999, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert sorted(batch(v, [twins.copy(), twins.copy()], obj, 2)) == [0, 1]


def test_a_batch_can_be_taken_from_the_front_alone():
    v = np.array([[9.0, 5.0], [8.0, 4.0], [1.0, 9.0]])
    obj = with_best(OBJ, np.array([[8.0, 5.0]]))
    covs = [np.eye(3), np.eye(3)]
    got = batch(v, covs, obj, 3, only=[0, 1])
    assert sorted(got) == [0, 1], "it cannot order what it was not offered"


# ------------------------------------------------------------------ the trade


def test_a_trade_names_what_is_gained_and_what_is_lost():
    obj = [Objective("potency", MAXIMISE), Objective("hERG", MINIMISE)]
    t = trade([9.0, 6.0], [8.0, 5.0], obj, name="new", against="old")
    assert [n for n, _v in t.gains] == ["potency"]
    assert [n for n, _v in t.loses] == ["hERG"]
    assert t.gains[0][1] == pytest.approx(1.0)
    assert t.loses[0][1] == pytest.approx(-1.0), "a loss is negative in the objective's own way"
    assert "gains potency" in str(t) and "loses hERG" in str(t)


def test_a_change_the_model_cannot_see_is_not_reported_as_a_trade():
    obj = [Objective("potency", MAXIMISE), Objective("hERG", MINIMISE)]
    t = trade([8.1, 5.05], [8.0, 5.0], obj, indifference=0.3)
    assert t.gains == [] and t.loses == []
    assert sorted(t.same) == ["hERG", "potency"]
    assert "no change worth reporting" in str(t)


def test_the_report_gives_one_trade_per_suggestion_against_the_best_made():
    y = np.array([[8.0, 5.0], [9.0, 6.0]])
    obj = with_best(OBJ, y)
    v = np.array([[9.5, 4.0], [7.0, 7.0]])
    trades = report(v, obj, ["a", "b"], [0, 1])
    assert len(trades) == 2
    assert trades[0].name == "a" and [n for n, _v in trades[0].gains] == ["potency", "hERG"]
    assert [n for n, _v in trades[1].loses] == ["potency", "hERG"]
