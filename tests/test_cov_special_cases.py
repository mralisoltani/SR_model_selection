import numpy as np
from deap import gp
from tree_mutation import pset_definition, ground_tree_generator
from utils import calc_Err_in


# --------------------------------------------------------------------------- #
# Force the choice function to return the same residuals, therefore optimism #
# collapses to zero                                                          #
# --------------------------------------------------------------------------- #
def test_cov_zero_with_identical_bootstrap(monkeypatch):
    n_features = 5
    pset, _ = pset_definition(n_features)
    tree = gp.PrimitiveTree.from_string("x1", pset)

    rng = np.random.default_rng(0)
    n = 40
    X = rng.uniform(0, 1, (n, 5))
    noise = rng.normal(0, 0.1, n)
    y = X[:, 0] + noise  # true model  y = x1 + ε

    # Optimism for normal bootstrap should be non-zero
    Err_free, err_app_free, optimism_free = calc_Err_in(
        tree, X, y, B=20, algorithm="lm"
    )
    assert optimism_free > 1e-6, "optimism unexpectedly zero without patch"

    def no_shuffle(a, size=None, replace=True, p=None):
        """Return `a` unchanged (repeated if size > len(a))."""
        arr = np.asarray(a)
        if size is None:
            return arr.copy()
        reps = -(-size // len(arr))
        return np.tile(arr, reps)[:size].copy()

    # Make the choice to return exactly the same vector each time
    monkeypatch.setattr(np.random, "choice", no_shuffle, raising=True)

    Err_fix, err_app_fix, optimism_fix = calc_Err_in(tree, X, y, B=20, algorithm="lm")

    # Apparent error stays the same, optimism collapses to zero
    assert np.isclose(optimism_fix, 0.0, atol=1e-10)
    assert np.isclose(err_app_fix + optimism_fix, err_app_fix, atol=1e-10)
    # Sanity: err_app has to equal the value from the unpatched run
    assert np.isclose(err_app_fix, err_app_free, atol=1e-10)


# --------------------------------------------------------------------------- #
# Easy perfect-fit ⇒ residuals are zero ⇒ optimism must be zero              #
# --------------------------------------------------------------------------- #
def test_cov_zero_when_perfect_fit():
    n_features = 5
    pset, _ = pset_definition(n_features)
    tree = gp.PrimitiveTree.from_string("add(x1,x2)", pset)  # easy model y = x1 + x2

    # perfect-fit dataset
    X = np.zeros((20, 5))
    X[:, 0] = np.linspace(0, 1, 20)
    X[:, 1] = np.linspace(0, 1, 20)
    y = X[:, 0].copy()  # no noise and easy linear model ⇒ residuals == 0

    Err, err_app, optimism = calc_Err_in(tree, X, y, B=10)

    assert np.isclose(err_app, 0.0, atol=1e-12)
    assert np.isclose(optimism, 0.0, atol=1e-12)
    assert np.isclose(Err, 0.0, atol=1e-12)


# --------------------------------------------------------------------------- #
# Complex perfect-fit ⇒ residuals are zero ⇒ optimism must be zero           #
# --------------------------------------------------------------------------- #
def test_cov_zero_with_complex_model():
    n_features = 5
    pset, _ = pset_definition(n_features)
    complex_tree = ground_tree_generator(pset, function_name="friedman")

    rng = np.random.default_rng(123)
    n = 120
    X = rng.random((n, 5))

    ground_func = gp.compile(expr=complex_tree, pset=pset)
    y = np.array([ground_func(*row) for row in X])  # exact targets, σ² = 0

    Err, err_app, optimism = calc_Err_in(complex_tree, X, y, algorithm="lm", B=25)

    assert np.isclose(err_app, 0.0, atol=1e-9)
    assert np.isclose(optimism, 0.0, atol=1e-9)
    assert np.isclose(Err, 0.0, atol=1e-9)
