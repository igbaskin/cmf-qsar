"""Inverse design on prototypes: the generator, the duplicate test, the oracles and the loop
(plan item M11).

What is tested, in the order the module builds it:

* the generator: how many candidates a prototype allows, that the index of a candidate decodes
  to the same assignment every time, that a sample is drawn without replacement, that the two
  modes do what they say, and that a vacancy removes an atom;
* the duplicate test: the symmetry operations of the prototype as permutations of its sites, one
  candidate per orbit, and the descriptor channels as the other test, which must not separate
  two cells that are the same structure;
* the oracles: a tabulated value, an ASE calculator with and without a relaxation, and the call
  count that a design budget is stated in;
* the loop: it asks the oracle once per candidate and never twice, it stops at the budget, it
  improves on its start set on a set where that is possible, a rule that ignores the model does
  worse than one that uses it, minimising and maximising are the same loop with a sign, and a
  closed loop over generated structures with a real calculator runs end to end;
* what the module refuses: an empty pool, an unknown rule, a kernel of the wrong shape, a site
  class with no elements and an index past the end of the enumeration.
"""
import numpy as np
import pytest

from cmfqsar import prototypes as P
from cmfqsar.periodic import INORGANIC_FIELDS, PeriodicSpec, crystal_table, inorganic_fieldset

ase_build = pytest.importorskip("ase.build")
bulk = ase_build.bulk
EMT = pytest.importorskip("ase.calculators.emt").EMT

SPEC = PeriodicSpec(r_max=6.0, step=0.5, sigma=0.5, taper=2.0, background=True)


@pytest.fixture(scope="module")
def fieldset():
    return inorganic_fieldset(elements=[], physchem=INORGANIC_FIELDS,
                              charge_model="electronegativity", alpha=1.0)


@pytest.fixture(scope="module")
def perovskite():
    """The ABO3 prototype with four A and three B elements: 12 compositions."""
    return P.Prototype.by_symbol(P.cubic_perovskite(3.9),
                                 {"Sr": ["Sr", "Ca", "Ba", "Pb"], "Ti": ["Ti", "Zr", "Sn"]})


@pytest.fixture(scope="module")
def host():
    """A 2 x 2 x 1 supercell of cubic fcc copper: 16 sites."""
    return bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 1)


@pytest.fixture(scope="module")
def arrangement(host):
    """Three of the 16 sites taking gold: 560 placements, 10 distinct arrangements."""
    return P.Prototype(host, (P.SiteClass("Cu", tuple(range(len(host))), ("Au",), count=3),))


# --------------------------------------------------------------------------- #
# The generator
# --------------------------------------------------------------------------- #
def test_composition_mode_counts_the_product(perovskite):
    assert perovskite.n_candidates == 4 * 3
    pool = P.candidates(perovskite)
    assert len(pool) == 12
    assert len({c.formula for c in pool}) == 12
    assert all(len(c.atoms) == 5 for c in pool)


def test_arrangement_mode_counts_the_choices(arrangement):
    assert arrangement.n_candidates == 560                    # C(16, 3), one element
    pool = P.candidates(arrangement)
    assert len(pool) == 560
    assert all(sorted(c.atoms.get_chemical_symbols()).count("Au") == 3 for c in pool)


def test_an_index_decodes_to_the_same_candidate_twice(perovskite):
    for index in (0, 5, 11):
        first, second = perovskite.candidate_at(index), perovskite.candidate_at(index)
        assert first.label == second.label
        assert first.atoms.get_chemical_symbols() == second.atoms.get_chemical_symbols()


def test_every_index_gives_a_different_assignment(arrangement):
    seen = {arrangement.candidate_at(i).label for i in range(arrangement.n_candidates)}
    assert len(seen) == arrangement.n_candidates


def test_a_sample_is_drawn_without_replacement(arrangement):
    pool = P.candidates(arrangement, limit=50, seed=3)
    assert len(pool) == 50
    assert len({c.index for c in pool}) == 50
    again = P.candidates(arrangement, limit=50, seed=3)
    assert [c.index for c in pool] == [c.index for c in again]
    other = P.candidates(arrangement, limit=50, seed=4)
    assert [c.index for c in pool] != [c.index for c in other]


def test_a_vacancy_removes_the_atom(host):
    proto = P.Prototype(host, (P.SiteClass("Cu", tuple(range(len(host))), (P.VACANCY,),
                                           count=2),))
    cand = proto.candidate_at(0)
    assert len(cand.atoms) == len(host) - 2
    assert set(cand.atoms.get_chemical_symbols()) == {"Cu"}


def test_two_classes_are_independent(host):
    proto = P.Prototype(host, (P.SiteClass("left", (0, 1), ("Au", "Ag")),
                               P.SiteClass("right", (2, 3), ("Ni", "Pd", "Pt"))))
    assert proto.n_candidates == 2 * 3
    labels = {P.candidates(proto)[i].label for i in range(6)}
    assert len(labels) == 6
    cand = proto.candidate_at(0)
    symbols = cand.atoms.get_chemical_symbols()
    assert symbols[0] == symbols[1] == "Au"
    assert symbols[2] == symbols[3] == "Ni"


def test_the_prototype_of_a_perovskite_is_the_five_atom_cell():
    atoms = P.cubic_perovskite(4.0, "Ba", "Ti", "O")
    assert atoms.get_chemical_symbols() == ["Ba", "Ti", "O", "O", "O"]
    assert np.allclose(atoms.get_cell()[:], 4.0 * np.eye(3))
    assert all(atoms.get_pbc())


# --------------------------------------------------------------------------- #
# Which candidates are the same structure
# --------------------------------------------------------------------------- #
def test_the_symmetry_operations_permute_the_sites(arrangement):
    perms = P.site_permutations(arrangement)
    assert perms.shape[1] == 16
    assert np.array_equal(perms[0], np.arange(16))            # the identity comes first
    assert perms.shape[0] > 1
    for row in perms:
        assert sorted(row.tolist()) == list(range(16))        # every row is a permutation


def test_one_candidate_per_orbit(arrangement):
    pool = P.candidates(arrangement)
    kept, dropped = P.unique(arrangement, pool)
    assert len(kept) == 10
    assert dropped == 560 - 10
    assert len({c.index for c in kept}) == 10


def test_the_reduction_of_one_substituted_site_leaves_one(host):
    proto = P.Prototype(host, (P.SiteClass("Cu", tuple(range(len(host))), ("Au",), count=1),))
    kept, dropped = P.unique(proto, P.candidates(proto))
    assert (len(kept), dropped) == (1, 15)                    # every fcc site is the same site


def test_the_channels_do_not_separate_one_structure_from_itself(arrangement, fieldset):
    """A cell and the same cell shifted by a lattice vector have one channel set."""
    cand = arrangement.candidate_at(0)
    shifted = cand.atoms.copy()
    shifted.positions += shifted.get_cell()[0] / 2.0 + np.array([0.0, 0.0, 0.3])
    shifted.wrap()
    table = crystal_table([cand.atoms, shifted], fieldset, ["as built", "shifted"])
    keep, dropped = P.unique_by_channels(table.molecules, SPEC, table.weights.shape[1])
    assert (len(keep), dropped) == (1, 1)


def test_the_channels_separate_most_arrangements(arrangement, fieldset):
    """The channels give their own key to all but the homometric pairs of the enumeration."""
    kept, _ = P.unique(arrangement, P.candidates(arrangement))
    table = crystal_table([c.atoms for c in kept], fieldset, [c.label for c in kept])
    keep, dropped = P.unique_by_channels(table.molecules, SPEC, table.weights.shape[1])
    assert len(keep) == 8 and dropped == 2
    assert len(P.channel_keys(table.molecules, SPEC, table.weights.shape[1])) == 10


# --------------------------------------------------------------------------- #
# The oracles
# --------------------------------------------------------------------------- #
def test_the_lookup_oracle_reads_and_counts(perovskite):
    pool = P.candidates(perovskite)
    y = np.arange(len(pool), dtype=float)
    oracle = P.LookupOracle(y)
    assert oracle(pool[3]) == 3.0
    assert oracle(pool[0]) == 0.0
    assert oracle.calls == 2
    assert oracle.asked == [3, 0]


def test_the_lookup_oracle_refuses_what_it_has_no_value_for(perovskite):
    oracle = P.LookupOracle({0: 1.0})
    with pytest.raises(ValueError, match="no value is tabulated"):
        oracle(perovskite.candidate_at(1))


def test_the_calculator_oracle_relaxes_and_lowers_the_energy(arrangement):
    cand = arrangement.candidate_at(0)
    fixed = P.CalculatorOracle(EMT, relax=False)
    relaxed = P.CalculatorOracle(EMT, relax=True, steps=40)
    before, after = fixed(cand), relaxed(cand)
    assert after <= before + 1e-9
    assert fixed.calls == relaxed.calls == 1
    assert relaxed.seconds > 0.0
    assert len(relaxed.relaxed[cand.index]) == len(cand.atoms)


def test_the_calculator_oracle_returns_the_energy_per_atom(arrangement):
    cand = arrangement.candidate_at(0)
    per_atom = P.CalculatorOracle(EMT, relax=False)(cand)
    whole = P.CalculatorOracle(EMT, relax=False, per_atom=False)(cand)
    assert whole == pytest.approx(per_atom * len(cand.atoms), rel=1e-9)


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def pool_and_kernel(arrangement, fieldset):
    """The ten distinct arrangements, their kernel, and an energy for each one."""
    kept, _ = P.unique(arrangement, P.candidates(arrangement))
    pool = P.as_candidates([c.atoms for c in kept], [c.label for c in kept])
    table = crystal_table([c.atoms for c in kept], fieldset, [c.label for c in kept])
    K = P.pool_kernel(P.feature_matrix(table.molecules, SPEC, table.weights.shape[1]),
                      P.block_sizes(SPEC, table.weights.shape[1]))
    y = np.asarray([P.CalculatorOracle(EMT, relax=False)(c) for c in pool], float)
    return pool, K, y


def test_the_loop_asks_once_per_candidate(pool_and_kernel):
    pool, K, y = pool_and_kernel
    oracle = P.LookupOracle(y)
    rec = P.design_loop(pool, oracle, start=[0, 1, 2], budget=4, kernel=K, rule="ucb",
                        maximise=False)
    assert len(rec["steps"]) == 4
    assert oracle.calls == 3 + 4
    assert len(set(oracle.asked)) == len(oracle.asked)


def test_the_loop_stops_when_the_pool_runs_out(pool_and_kernel):
    pool, K, y = pool_and_kernel
    rec = P.design_loop(pool, P.LookupOracle(y), start=[0], budget=100, kernel=K, rule="greedy",
                        maximise=False)
    assert len(rec["steps"]) == len(pool) - 1
    assert rec["computed"] == len(pool)


def test_the_loop_finds_the_best_of_a_small_pool(pool_and_kernel):
    pool, K, y = pool_and_kernel
    rec = P.design_loop(pool, P.LookupOracle(y), start=[0, 1, 2], budget=5, kernel=K, rule="ei",
                        maximise=False)
    assert rec["best"] <= float(np.min(y[[0, 1, 2]]))
    assert rec["best"] == pytest.approx(y[rec["best_index"]])
    assert rec["best"] in set(y.tolist())


def test_minimising_is_maximising_the_negative(pool_and_kernel):
    pool, K, y = pool_and_kernel
    low = P.design_loop(pool, P.LookupOracle(y), start=[0, 1, 2], budget=5, kernel=K,
                        rule="greedy", maximise=False)
    high = P.design_loop(pool, P.LookupOracle(-y), start=[0, 1, 2], budget=5, kernel=K,
                         rule="greedy", maximise=True)
    assert [s["taken"] for s in low["steps"]] == [s["taken"] for s in high["steps"]]
    assert low["best"] == pytest.approx(-high["best"])


def test_the_model_based_rule_beats_random_on_a_pool_it_can_rank(pool_and_kernel):
    """Greedy ranking must reach a lower energy than picking at random, on average."""
    pool, K, y = pool_and_kernel
    greedy, random = [], []
    for seed in range(12):
        rng = np.random.default_rng(seed)
        start = sorted(int(i) for i in rng.choice(len(pool), 3, replace=False))
        greedy.append(P.design_loop(pool, P.LookupOracle(y), start=start, budget=2, kernel=K,
                                    rule="greedy", maximise=False, seed=seed)["best"])
        random.append(P.design_loop(pool, P.LookupOracle(y), start=start, budget=2, kernel=K,
                                    rule="random", maximise=False, seed=seed)["best"])
    assert np.mean(greedy) <= np.mean(random)


def test_first_better_reads_the_sign(pool_and_kernel):
    pool, K, y = pool_and_kernel
    rec = P.design_loop(pool, P.LookupOracle(y), start=[0, 1, 2], budget=6, kernel=K,
                        rule="greedy", maximise=False)
    step = P.first_better(rec, rec["start_best"])
    assert step is None or rec["steps"][step - 1]["value"] < rec["start_best"]
    assert P.first_better(rec, float(np.min(y)) - 1.0) is None
    assert "best" in P.summarise(rec)


def test_a_closed_loop_computes_every_candidate_it_proposes(arrangement, fieldset):
    """The loop with a calculator: no table of values anywhere, and the relaxed cells kept."""
    kept, _ = P.unique(arrangement, P.candidates(arrangement))
    table = crystal_table([c.atoms for c in kept], fieldset, [c.label for c in kept])
    K = P.pool_kernel(P.feature_matrix(table.molecules, SPEC, table.weights.shape[1]),
                      P.block_sizes(SPEC, table.weights.shape[1]))
    oracle = P.CalculatorOracle(EMT, relax=True, steps=20)
    rec = P.design_loop(kept, oracle, start=[0, 1], budget=3, kernel=K, rule="ucb",
                        maximise=False)
    assert oracle.calls == 5
    assert len(oracle.relaxed) == 5
    assert all(np.isfinite(s["value"]) for s in rec["steps"])
    assert rec["best"] <= min(s["value"] for s in rec["steps"] + [{"value": rec["start_best"]}])


def test_the_loop_builds_its_own_kernel_when_given_a_spec(arrangement, fieldset):
    kept, _ = P.unique(arrangement, P.candidates(arrangement))
    table = crystal_table([c.atoms for c in kept], fieldset, [c.label for c in kept])
    pool = P.as_candidates(table.molecules, [c.label for c in kept])
    y = np.arange(len(pool), dtype=float)
    rec = P.design_loop(pool, P.LookupOracle(y), start=[0, 1], budget=2, spec=SPEC,
                        n_fields=table.weights.shape[1], rule="variance", maximise=True)
    assert len(rec["steps"]) == 2


# --------------------------------------------------------------------------- #
# What the module refuses
# --------------------------------------------------------------------------- #
def test_an_empty_pool_is_refused():
    with pytest.raises(ValueError, match="pool is empty"):
        P.design_loop([], P.LookupOracle([]), start=[], budget=1, kernel=np.zeros((0, 0)))


def test_an_unknown_rule_is_refused(pool_and_kernel):
    pool, K, y = pool_and_kernel
    with pytest.raises(ValueError, match="unknown rule"):
        P.design_loop(pool, P.LookupOracle(y), start=[0], budget=1, kernel=K, rule="hope")


def test_a_kernel_of_the_wrong_shape_is_refused(pool_and_kernel):
    pool, K, y = pool_and_kernel
    with pytest.raises(ValueError, match="the pool has"):
        P.design_loop(pool, P.LookupOracle(y), start=[0], budget=1, kernel=K[:3, :3])


def test_a_loop_without_a_kernel_or_a_spec_is_refused(pool_and_kernel):
    pool, _K, y = pool_and_kernel
    with pytest.raises(ValueError, match="give the loop a kernel"):
        P.design_loop(pool, P.LookupOracle(y), start=[0], budget=1)


def test_a_start_outside_the_pool_is_refused(pool_and_kernel):
    pool, K, y = pool_and_kernel
    with pytest.raises(ValueError, match="start set names candidate"):
        P.design_loop(pool, P.LookupOracle(y), start=[999], budget=1, kernel=K)


def test_a_site_class_without_elements_is_refused():
    with pytest.raises(ValueError, match="no elements"):
        P.SiteClass("A", (0, 1), ())
    with pytest.raises(ValueError, match="no sites"):
        P.SiteClass("A", (), ("Au",))


def test_a_count_larger_than_the_class_is_refused(host):
    with pytest.raises(ValueError, match="asks for"):
        P.SiteClass("Cu", (0, 1, 2), ("Au",), count=4)


def test_a_site_in_two_classes_is_refused(host):
    with pytest.raises(ValueError, match="belongs to one class"):
        P.Prototype(host, (P.SiteClass("a", (0, 1), ("Au",)),
                           P.SiteClass("b", (1, 2), ("Ag",))))


def test_an_index_past_the_end_is_refused(perovskite):
    with pytest.raises(ValueError, match="allows 12 candidates"):
        perovskite.candidate_at(12)


def test_a_prototype_without_the_element_asked_for_is_refused():
    with pytest.raises(ValueError, match="has no Fe site"):
        P.Prototype.by_symbol(P.cubic_perovskite(3.9), {"Fe": ["Co"]})


def test_an_unknown_descriptor_is_refused():
    with pytest.raises(ValueError, match="no descriptor is known"):
        P.feature_matrix([], "not a spec", 1)
