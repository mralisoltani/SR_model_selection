import math
import random
import copy
from functools import partial
import numpy as np
from deap import gp, base
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import time
from scipy.optimize import minimize
import multiprocessing as mp

# -----------------------------------------------------------------------------
# Primitive Set and Toolbox
# -----------------------------------------------------------------------------


def pset_definition(n_features):
    pset = gp.PrimitiveSet("MAIN", n_features)
    pset.context["np"] = np

    EPS = 1e-6
    _EXP_LIMIT = 80  # np.log(np.finfo(np.float64).max)  # ≈ 709.78

    def protected_div(left, right):
        if np.ndim(left) == 0 and np.ndim(right) == 0:
            return left / right if abs(right) > EPS else left

        left = np.asarray(left, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)

        out_shape = np.broadcast(left, right).shape
        out = np.empty(out_shape, dtype=np.float64)

        mask = np.abs(right) > EPS
        np.copyto(out, np.broadcast_to(left, out_shape))
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            np.divide(left, right, out=out, where=mask)

        return out.item() if out.ndim == 0 else out

    def protected_pow(base, exp):
        """
        ppow(b, p) = exp( clip( p * log(|b| + eps), -C, C ) )
        """
        eps = 1e-16
        C = 30
        base_is_scalar = np.isscalar(base)
        exp_is_scalar = np.isscalar(exp)

        base = np.asarray(base, dtype=float)
        exp = np.asarray(exp, dtype=float)

        abs_base = np.abs(base) + eps

        with np.errstate(
            divide="ignore", invalid="ignore", over="ignore", under="ignore"
        ):
            log_term = np.log(abs_base)
            val = exp * log_term
            clipped = np.clip(val, -C, C)
            out = np.exp(clipped)

        if base_is_scalar and exp_is_scalar:
            return float(out)
        return out

    def protected_sqrt(x):
        return np.sqrt(np.abs(x))

    def protected_exp(x):
        return np.exp(np.clip(x, a_min=None, a_max=_EXP_LIMIT))

    pset.addPrimitive(np.add, 2, name="add")
    pset.addPrimitive(np.subtract, 2, name="sub")
    pset.addPrimitive(np.multiply, 2, name="mul")
    pset.addPrimitive(np.sin, 1, name="sin")
    pset.addPrimitive(np.cos, 1, name="cos")
    pset.addPrimitive(protected_pow, 2, name="protected_pow")
    pset.addPrimitive(protected_sqrt, 1, name="protected_sqrt")
    pset.addPrimitive(protected_div, 2, name="protected_div")
    pset.addPrimitive(protected_exp, 1, name="protected_exp")

    pset.renameArguments(**{f"ARG{i}": f"x{i+1}" for i in range(n_features)})

    pset.addEphemeralConstant("c_u", partial(np.random.uniform, -5, 5))
    pset.addEphemeralConstant("c_n", partial(np.random.normal, 0, 1))
    pset.addEphemeralConstant(
        "c_p", partial(np.random.choice, [0.5, 2.0, 3.0, -0.5, -2.0])
    )
    pset.addEphemeralConstant("c_b", partial(np.random.uniform, -50, 50))

    toolbox = base.Toolbox()
    toolbox.register("expr_full", gp.genFull, pset=pset, min_=1, max_=2)
    toolbox.register("individual", gp.PrimitiveTree, toolbox.expr_full)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("clone", copy.deepcopy)

    return pset, toolbox


def ground_tree_generator(pset, function_name="friedman"):
    # -----------------------------------------------------------------------------
    # Ground‑Truth Functions
    # -----------------------------------------------------------------------------
    f1 = (
        "add("
        "  add("
        "    add("
        "      mul(10, sin(mul(3.14159265, mul(x1, x2)))),"
        "      mul(20, mul(sub(x3, 0.5), sub(x3, 0.5)))"
        "    ),"
        "    mul(10, x4)"
        "  ),"
        "  mul(5, x5)"
        ")"
    )

    f2 = (
        "mul("
        "  protected_exp(mul(-1, protected_pow(sub(x1, 1), 2))),"
        "  protected_pow(add(1.2, protected_pow(sub(x2, 2.5), 2)), -1)"
        ")"
    )
    f3 = (
        "mul("
        "  protected_exp(mul(x1, -1)),"
        "  mul("
        "    protected_pow(x1, 3),"
        "    mul("
        "      cos(x1),"
        "      mul("
        "        sin(x1),"
        "        sub("
        "          mul(cos(x1), protected_pow(sin(x1), 2)),"
        "          1"
        "        )"
        "      )"
        "    )"
        "  )"
        ")"
    )

    f4 = "mul(" f"{f3}," "  sub(x2, 5)" ")"

    f5 = (
        "mul("
        "  mul("
        "    mul(30, sub(x1, 1)),"
        "    sub(x3, 1)"
        "  ),"
        "  protected_pow("
        "    sub("
        "      mul(x1, protected_pow(x2, 2)),"
        "      mul(10, protected_pow(x2, 2))"
        "    ),"
        "    -1"
        "  )"
        ")"
    )

    f6 = (
        "protected_div("
        "  sub("
        "    add("
        "      protected_pow(sub(x1, 3), 4),"
        "      protected_pow(sub(x2, 3), 3)"
        "    ),"
        "    sub(x2, 3)"
        "  ),"
        "  add("
        "    protected_pow(sub(x2, 2), 4),"
        "    10"
        "  )"
        ")"
    )

    f7 = (
        "add("
        "  mul(sub(x1, 3), sub(x2, 3)),"
        "  mul(2, sin(mul(sub(x1, 4), sub(x2, 4))))"
        ")"
    )

    ground_truth_functions = {
        "f1": f1,
        "friedman": f1,
        "f2": f2,
        "kotanchek": f2,
        "f3": f3,
        "salustowicz": f3,
        "f4": f4,
        "salustowicz2d": f4,
        "f5": f5,
        "ratpol3d": f5,
        "f6": f6,
        "ratpol2d": f6,
        "f7": f7,
        "ripple": f7,
    }

    ground_tree = gp.PrimitiveTree.from_string(
        ground_truth_functions[function_name], pset
    )

    return ground_tree


def sanity_check_deap_parse(pset, toolbox):
    true_funcs = {
        "f1": lambda x1, x2, x3, x4, x5: (
            10 * math.sin(math.pi * x1 * x2) + 20 * (x3 - 0.5) ** 2 + 10 * x4 + 5 * x5
        ),
        "f2": lambda x1, x2, x3, x4, x5: math.exp(-((x1 - 1) ** 2))
        * (1.2 + (x2 - 2.5) ** 2) ** -1,
        "f3": lambda x1, x2, x3, x4, x5: math.exp(-x1)
        * x1**3
        * math.cos(x1)
        * math.sin(x1)
        * (math.cos(x1) * math.sin(x1) ** 2 - 1),
        "f4": lambda x1, x2, x3, x4, x5: math.exp(-x1)
        * x1**3
        * math.cos(x1)
        * math.sin(x1)
        * (math.cos(x1) * math.sin(x1) ** 2 - 1)
        * (x2 - 5),
        "f5": lambda x1, x2, x3, x4, x5: 30
        * (x1 - 1)
        * (x3 - 1)
        / (
            (x1 * x2**2 - 10 * x2**2)
            if (x1 * x2**2 - 10 * x2**2) != 0
            else 1e-6
        ),
        "f6": lambda x1, x2, x3, x4, x5: ((x1 - 3) ** 4 + (x2 - 3) ** 3 - (x2 - 3))
        / ((x2 - 2) ** 4 + 10),
        "f7": lambda x1, x2, x3, x4, x5: (x1 - 3) * (x2 - 3)
        + 2 * math.sin((x1 - 4) * (x2 - 4)),
    }

    samples = [
        (
            random.uniform(0, 10),
            random.uniform(0, 10),
            random.uniform(0, 10),
            random.uniform(0, 10),
            random.uniform(0, 10),
        )
        for _ in range(10000)
    ]

    for key, values in true_funcs.items():
        tree = ground_tree_generator(pset, function_name=str(key))
        func = toolbox.compile(expr=tree)
        test_val = func(1.0, 2.5, 3.2, 2.1, 5.1)
        mse = np.mean(
            [
                (func(x1, x2, x3, x4, x5) - true_funcs[key](x1, x2, x3, x4, x5)) ** 2
                for x1, x2, x3, x4, x5 in samples
            ]
        )
        print(f"{key} @ (x1=1.0, x2=2.5) = {test_val:.6f},  MSE: {mse:.10f}")


# -----------------------------------------------------------------------------
# Uniform Data-Points
# -----------------------------------------------------------------------------


def data_points(ground_tree, pset, n_samples):
    ground_func = gp.compile(expr=ground_tree, pset=pset)
    n_points = len(pset.arguments)
    X = np.random.rand(n_samples, n_points)
    Y = np.fromiter((ground_func(*row) for row in X), dtype=float, count=n_samples)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42
    )
    return X_train, X_test, Y_train, Y_test


# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------


def mse_for_tree(individual: gp.PrimitiveTree, pset, X, Y) -> float:
    f = gp.compile(expr=individual, pset=pset)
    y_pred = np.fromiter((f(*row) for row in X), dtype=float, count=len(X))
    r = y_pred - Y
    if not np.all(np.isfinite(r)):
        return 1e10
    if np.any(np.abs(r) > 1e100):
        return 1e10
    return mean_squared_error(Y, y_pred)


def count_numeric_terminals(ind) -> int:
    k = 0
    for node in ind:
        if isinstance(node, gp.Terminal):
            if hasattr(node, "value") and isinstance(node.value, (int, float)):
                k += 1
    return k


def mse_for_constants(theta, ind, const_nodes, pset, X, Y):
    for node, val in zip(const_nodes, theta):
        node.value = float(val)
    try:
        f = gp.compile(expr=ind, pset=pset)
    except Exception:
        return 1e20

    y_pred = np.fromiter((f(*row) for row in X), dtype=float, count=len(X))

    if (not np.all(np.isfinite(y_pred))) or np.any(np.abs(y_pred) > 1e100):
        return 1e20

    r = y_pred - Y
    return float(np.mean(r * r))


def get_constant_nodes(ind):
    nodes = []
    vals = []
    for node in ind:
        if isinstance(node, gp.Terminal) and hasattr(node, "value"):
            if isinstance(node.value, (int, float)):
                nodes.append(node)
                vals.append(float(node.value))
    return nodes, np.array(vals, dtype=float)


def _run_single_restart(args):
    (
        ind,
        X,
        Y,
        x0,
        method,
        options_dict,
    ) = args

    n_features = X.shape[1]
    pset_worker, _ = pset_definition(n_features)

    ind_local = copy.deepcopy(ind)

    const_nodes_local, init_vals_local = get_constant_nodes(ind_local)

    if len(init_vals_local) == 0:
        mse = mse_for_tree(ind_local, pset_worker, X, Y)
        return mse, init_vals_local, ind_local

    def mse_for_constants_local(theta):
        return mse_for_constants(theta, ind_local, const_nodes_local, pset_worker, X, Y)

    res = minimize(
        mse_for_constants_local,
        x0=x0,
        method=method,
        options=options_dict,
    )

    theta_candidate = res.x
    mse_candidate = float(res.fun)

    for node, val in zip(const_nodes_local, theta_candidate):
        node.value = float(val)

    return mse_candidate, theta_candidate, ind_local


def optimise_tree_constants(
    ind,
    pset,
    X,
    Y,
    n_restarts: int = 5,
    method: str = "BFGS",
    n_jobs: int | None = None,
):
    const_nodes, init_vals = get_constant_nodes(ind)
    print(f"Original values: {init_vals}")

    options_dict = {"disp": False}

    if len(init_vals) == 0:
        mse = mse_for_tree(ind, pset, X, Y)
        return ind, mse

    dim = len(init_vals)

    x0_list = []
    for r in range(n_restarts):
        if r == 0:
            x0 = init_vals.copy()
        elif r < round(n_restarts / 2):
            x0 = init_vals + np.random.normal(0, 1, size=dim)
        else:
            x0 = np.random.uniform(-10.0, 10.0, size=dim)
        x0_list.append(x0)

    if n_jobs is None:
        n_jobs = min(n_restarts, mp.cpu_count())

    arg_list = [
        (
            ind,
            X,
            Y,
            x0_list[r],
            method,
            options_dict,
        )
        for r in range(n_restarts)
    ]

    with mp.Pool(processes=n_jobs) as pool:
        results = pool.map(_run_single_restart, arg_list)

    best_mse = np.inf
    best_theta = None
    best_tree = None

    for mse_candidate, theta_candidate, ind_local in results:
        if mse_candidate < best_mse:
            best_mse = mse_candidate
            best_theta = theta_candidate
            best_tree = ind_local

    print(f"Best theta after {n_restarts} restarts: {best_theta}")
    print(f"Best MSE: {best_mse:.8f}")

    return best_tree, best_mse


def count_distinct_features(ind, pset):
    k = 0
    used = set()
    arg_names = list(pset.arguments)
    for node in ind:
        if isinstance(node, gp.Terminal):
            name = str(node.name)
            if name in arg_names:
                k += 1
                used.add(name)
            elif name.startswith("ARG"):
                used.add(name)
                k += 1
    return k


def mutate_subtree_const(
    ind, pset, toolbox, min_depth=1, max_depth=3, min_consts=2, max_tries=50
):
    expr_gen = partial(gp.genFull, pset=pset, min_=min_depth, max_=max_depth)
    for _ in range(max_tries):
        clone = toolbox.clone(ind)
        (mutant,) = gp.mutUniform(clone, expr=expr_gen, pset=pset)
        if count_numeric_terminals(mutant) >= min_consts:
            return mutant
    return mutant


def mutate_subtree(ind, pset, toolbox, min_depth=2, max_depth=10):
    clone = toolbox.clone(ind)
    expr_gen = partial(gp.genFull, pset=pset, min_=min_depth, max_=max_depth)
    (mutant,) = gp.mutUniform(clone, expr=expr_gen, pset=pset)
    return mutant


def ground_truth_perturbations(
    pset,
    toolbox,
    ground_tree,
    X_train=None,
    X_test=None,
    Y_train=None,
    Y_test=None,
    n_mutations: int = 10,
):
    list_perturbations = []
    duplicates_check = set()
    print("Ground‑truth expression as tree:\n", ground_tree, "\n")
    mse_train_ground_truth = mse_for_tree(ground_tree, pset, X_train, Y_train)
    print(f"Ground‑truth train MSE: {mse_train_ground_truth:.8f}\n")

    if all(x is not None for x in [X_train, X_test, Y_train, Y_test]):
        ground_truth_train_mse = mse_for_tree(ground_tree, pset, X_train, Y_train)
        ground_truth_test_mse = mse_for_tree(ground_tree, pset, X_test, Y_test)
        print(
            f"Ground‑truth MSEs   train={ground_truth_train_mse:.8f}"
            f"test={ground_truth_test_mse:.8f}\n"
        )
        i = 0
        while i < n_mutations:
            mutant = mutate_subtree(ground_tree, pset=pset, toolbox=toolbox)
            tr_mse = mse_for_tree(mutant, pset, X_train, Y_train)
            te_mse = mse_for_tree(mutant, pset, X_test, Y_test)
            list_perturbations.append(
                {"id": i, "tree": mutant, "train_mse": tr_mse, "test_mse": te_mse}
            )
            f = gp.compile(expr=mutant, pset=pset)
            pred = np.fromiter(
                (f(*row) for row in X_train), dtype=float, count=len(X_train)
            )
            if np.all(np.isfinite(pred)):
                i += 1
        list_perturbations.append(
            {
                "id": i + 1,
                "tree": ground_tree,
                "train_mse": ground_truth_train_mse,
                "test_mse": ground_truth_test_mse,
            }
        )
    elif any(x is None for x in [X_train, X_test, Y_train, Y_test]):
        i = 0
        mse_train_ground_truth = mse_for_tree(ground_tree, pset, X_train, Y_train)
        mse_threshold = 10
        start_time = time.time()
        while i < n_mutations:
            mutant = mutate_subtree(ground_tree, pset=pset, toolbox=toolbox)
            mse_train_mutant_tmp = mse_for_tree(mutant, pset, X_train, Y_train)

            if mse_train_mutant_tmp < mse_threshold * mse_train_ground_truth:
                print("Before optimization:", mse_train_mutant_tmp)
                print("Mutant is worth tuning parameters, running optimization ...")

                mutant, mse_train_mutant = optimise_tree_constants(
                    mutant, pset, X_train, Y_train, n_restarts=100, method="BFGS"
                )
            else:
                continue

            f = gp.compile(expr=mutant, pset=pset)
            pred = np.fromiter(
                (f(*row) for row in X_train), dtype=float, count=len(X_train)
            )

            mean_pred = np.mean(pred)
            pred_rounded = np.round(pred, decimals=8)
            pred_rounded_tuple = tuple(pred_rounded)

            n_params = count_numeric_terminals(mutant)
            n_featurses = count_distinct_features(mutant, pset)
            if (
                np.all(np.isfinite(pred))
                and mean_pred != 0
                and mean_pred < 1e4
                and mean_pred > -1e4
                and len(mutant) > 5
                and n_featurses >= 2
                and pred_rounded_tuple not in duplicates_check
                and mse_train_mutant < mse_train_ground_truth
            ):
                duplicates_check.add(pred_rounded_tuple)
                list_perturbations.append(
                    {
                        "id": i,
                        "tree": mutant,
                        "mean_train_pred": np.mean(pred),
                        "num_of_params": n_params,
                    }
                )
                print(f"Mutation {i:02d}: ")
                print("   ", mutant, "\n")
                print(f"mse_train_mutant: {mse_train_mutant:.8f}")
                print(f"Elapsed time: {time.time() - start_time:.2f} seconds\n")
                i += 1
        n_params = count_numeric_terminals(ground_tree)
        f = gp.compile(expr=ground_tree, pset=pset)
        pred = np.fromiter(
            (f(*row) for row in X_train), dtype=float, count=len(X_train)
        )
        list_perturbations.append(
            {
                "id": i + 1,
                "tree": ground_tree,
                "mean_train_pred": np.mean(pred),
                "num_of_params": n_params,
            }
        )

    return list_perturbations
