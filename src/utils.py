import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import networkx as nx
import re
from tree_mutation import pset_definition

# from functools import partial
from deap import gp, base

# import random
import copy
import optuna
from scipy.optimize import least_squares
import numpy as np


# ############### Show the output as a tree


def _clean_deap_labels(labels):
    pat = re.compile(r"^protected_")
    # gp.graph gives {node_id: label}; ensure str() in case of numeric terminals
    return {nid: pat.sub("", str(lbl)) for nid, lbl in labels.items()}


def show_as_tree(tree):
    """
    Change the individual to a tree and show
    """
    nodes, edges, labels = gp.graph(tree)
    labels = _clean_deap_labels(labels)

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    pos = graphviz_layout(G, prog="dot")

    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    plt.tight_layout()
    plt.savefig("random_tree.png", dpi=300)
    plt.show()
    print("DEAP output: ", tree)


def _signature(node):
    """Return a hashable description safe for any DEAP node."""
    if isinstance(node, gp.Primitive):
        return ("P", node.name, node.arity)
    elif isinstance(node, gp.Terminal):
        return ("T", getattr(node, "value", getattr(node, "name", str(node))))
    else:
        return ("?", str(node))


def _paths(tree):
    """
    Prefix-order traversal that yields each node's (path_tuple, signature).

    The path is a tuple of child-indices:  () → root, (0,) → root.left, etc.
    """
    stack = []  # [remaining_children, parent_path, next_child_index]
    for node in tree:
        if not stack:
            path = ()
        else:
            rem, parent_path, next_idx = stack[-1]
            path = parent_path + (next_idx,)
            stack[-1][0] -= 1  # consume one child slot
            stack[-1][2] += 1
            if stack[-1][0] == 0:  # parent is now full
                stack.pop()

        yield path, _signature(node)
        arity = getattr(node, "arity", 0)
        if arity:
            stack.append([arity, path, 0])


def show_diff_tree(
    tree1,
    tree2,
    *,
    same_colour="lightblue",
    diff_colour="red",
    perturbed_diff_colour="green",
):
    """Draw tree1 and tree2 and highlight the different nodes in both trees.
    - tree1 diffs: red
    - tree2 diffs: yellow
    """
    dict1 = dict(_paths(tree1))
    dict2 = dict(_paths(tree2))

    diff_paths = {
        p
        for p in dict1.keys() | dict2.keys()
        if p not in dict1 or p not in dict2 or dict1[p] != dict2[p]
    }

    # ---------- Tree 1 (ground truth) ----------
    nodes1, edges1, labels1 = gp.graph(tree1)
    labels1 = _clean_deap_labels(labels1)
    G1 = nx.Graph()
    G1.add_nodes_from(nodes1)
    G1.add_edges_from(edges1)

    # gp.graph preserves prefix order, so align paths with node ids
    path_list1 = [path for path, _ in _paths(tree1)]
    diff_node_ids1 = {nodes1[i] for i, p in enumerate(path_list1) if p in diff_paths}

    pos1 = graphviz_layout(G1, prog="dot")
    node_colors1 = [
        diff_colour if n in diff_node_ids1 else same_colour for n in G1.nodes()
    ]

    print("------------- ground-truth tree -------------")
    plt.figure(figsize=(7, 5))
    nx.draw_networkx_nodes(G1, pos1, node_color=node_colors1, edgecolors="black")
    nx.draw_networkx_edges(G1, pos1)
    nx.draw_networkx_labels(G1, pos1, labels1, font_size=8)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("main_tree.png", dpi=300)
    plt.show()

    # ---------- Tree 2 (perturbed) ----------
    nodes2, edges2, labels2 = gp.graph(tree2)
    labels2 = _clean_deap_labels(labels2)
    G2 = nx.Graph()
    G2.add_nodes_from(nodes2)
    G2.add_edges_from(edges2)

    path_list2 = [path for path, _ in _paths(tree2)]
    diff_node_ids2 = {nodes2[i] for i, p in enumerate(path_list2) if p in diff_paths}

    pos2 = graphviz_layout(G2, prog="dot")
    node_colors2 = [
        perturbed_diff_colour if n in diff_node_ids2 else same_colour
        for n in G2.nodes()
    ]

    print("------------- perturbed tree -------------")
    plt.figure(figsize=(7, 5))
    nx.draw_networkx_nodes(G2, pos2, node_color=node_colors2, edgecolors="black")
    nx.draw_networkx_edges(G2, pos2)
    nx.draw_networkx_labels(G2, pos2, labels2, font_size=8)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("perturbed_tree.png", dpi=300)
    plt.show()

    # ---------- Report ----------
    if diff_paths:
        print("Differing paths (root = ()):")
        for p in sorted(diff_paths):
            print(f"  {p}: {dict1.get(p, '∅')}  ↔  {dict2.get(p, '∅')}")
    else:
        print("No structural differences detected.")


# def show_diff_tree(tree1, tree2, *, same_colour="lightblue", diff_colour="red"):
#     """Draw tree1 and tree2 and highlight the different nodes in tree1"""

#     dict1 = dict(_paths(tree1))
#     dict2 = dict(_paths(tree2))

#     diff_paths = {
#         p
#         for p in dict1.keys() | dict2.keys()
#         if p not in dict1 or p not in dict2 or dict1[p] != dict2[p]
#     }

#     nodes, edges, labels = gp.graph(tree1)
#     G = nx.Graph()
#     G.add_nodes_from(nodes)
#     G.add_edges_from(edges)

#     # gp.graph preserves prefix order → we can re-compute the path list
#     path_list = [path for path, _ in _paths(tree1)]
#     diff_node_ids = {nodes[i] for i, p in enumerate(path_list) if p in diff_paths}

#     pos = graphviz_layout(G, prog="dot")
#     node_colors = [
#         diff_colour if n in diff_node_ids else same_colour for n in G.nodes()
#     ]

#     print("------------- ground-truth tree -------------")
#     nx.draw_networkx_nodes(G, pos, node_color=node_colors, edgecolors="black")
#     nx.draw_networkx_edges(G, pos)
#     nx.draw_networkx_labels(G, pos, labels, font_size=8)
#     plt.axis("off")
#     plt.tight_layout()
#     plt.savefig("main_tree.png", dpi=300)
#     plt.show()

#     print("------------- perturbed tree -------------")
#     show_as_tree(tree=tree2)

#     if diff_paths:
#         print("Differing paths (root = ()):")
#         for p in sorted(diff_paths):
#             print(f"  {p}: {dict1.get(p, '∅')}  ↔  {dict2.get(p, '∅')}")
#     else:
#         print("No structural differences detected.")


def add_tuning_params(expr_str, theta_prefix="theta", bias_prefix="bias"):
    """
    Replace every token x<i> with add(mul(thetak,x<i>),biask)
    (k = first time we meet a *new* variable name).
    """
    mapping = {}
    counter = 0
    var_pat = re.compile(r"\bx(\d+)\b")

    def repl(match):
        nonlocal counter
        x_name = f"x{match.group(1)}"
        if x_name not in mapping:
            mapping[x_name] = counter
            counter += 1
        k = mapping[x_name]
        return f"add(mul({theta_prefix}{k},{x_name})," f"{bias_prefix}{k})"

    new_expr = var_pat.sub(repl, expr_str)
    return new_expr


def problem_set_redefinition(pset):
    max_vars = len(pset.arguments)  # x1 … xn
    max_params = max_vars  # one theta and bias for each variables

    pset = gp.PrimitiveSet("MAIN", max_vars + 2 * max_params)

    pset.renameArguments(**{f"ARG{i}": f"x{i+1}" for i in range(max_vars)})

    extra = {}
    for k in range(max_params):
        extra[f"ARG{max_vars + k}"] = f"theta{k}"
        extra[f"ARG{max_vars + max_params+k}"] = f"bias{k}"
    pset.renameArguments(**extra)

    def protected_div(left, right):
        """
        Safe division that works for both scalars and arrays.
        Returns `left / right` unless |right| < EPS, in which case it returns `left`.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(np.abs(right) > 1e-6, left / right, left)
        # np.where on scalars returns a 0-D ndarray → convert back to Python float
        return float(result) if np.isscalar(left) and np.isscalar(right) else result

    def protected_pow(base, exp):
        # """
        # Safe power that replaces invalid or overflow results with 1.0.
        # Works for scalars and arrays.
        # """
        # with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        #     result = np.power(base, exp)

        # # mask of bad (nan or ±inf) entries
        # bad = ~np.isfinite(result)
        # if bad.any():  # handles both scalar & array
        #     result = np.where(bad, 1.0, result)

        # # squeeze back to scalar if needed
        # return float(result) if np.isscalar(base) and np.isscalar(exp) else result

        """
        Safe, saturated power:

            ppow(b, p) = exp( clip( p * log(|b| + eps), -C, C ) )

        Works elementwise for scalars and arrays.
        Always returns a finite, positive value in [exp(-C), exp(C)].
        """
        eps = 1e-16
        C = 30
        # Remember whether original inputs were scalars
        base_is_scalar = np.isscalar(base)
        exp_is_scalar = np.isscalar(exp)

        base = np.asarray(base, dtype=float)
        exp = np.asarray(exp, dtype=float)

        # |b| + eps to avoid log(0)
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

    _EXP_LIMIT = np.log(np.finfo(np.float64).max)  # ≈ 709.782712893

    def protected_exp(x):
        """
        Safe exp that never overflows:
            • clip the input at +_EXP_LIMIT
            • return ordinary np.exp on the clipped value
        Works for both scalars and arrays.
        """
        return np.exp(np.clip(x, a_min=None, a_max=_EXP_LIMIT))

    pset.addPrimitive(np.add, 2, name="add")
    pset.addPrimitive(np.subtract, 2, name="sub")
    pset.addPrimitive(np.multiply, 2, name="mul")
    # pset.addPrimitive(np.power, 2, name="pow")
    pset.addPrimitive(np.sin, 1, name="sin")
    pset.addPrimitive(np.cos, 1, name="cos")
    # pset.addPrimitive(np.exp, 1, name="exp")
    # pset.addPrimitive(np.sqrt, 1, name="sqrt")
    pset.addPrimitive(protected_pow, 2, name="protected_pow")
    pset.addPrimitive(protected_sqrt, 1, name="protected_sqrt")
    pset.addPrimitive(protected_div, 2, name="protected_div")
    pset.addPrimitive(protected_exp, 1, name="protected_exp")

    toolbox = base.Toolbox()
    toolbox.register("expr_full", gp.genFull, pset=pset, min_=1, max_=2)
    toolbox.register("individual", gp.PrimitiveTree, toolbox.expr_full)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("clone", copy.deepcopy)

    return pset, toolbox


def get_param_names(pset):
    """
    Act as a helper to return parameter name and variable name
    """
    arg_names = pset.arguments  # fixed ordering!
    x_names = [a for a in arg_names if a.startswith("x")]
    param_names = [
        a for a in arg_names if a.startswith("theta") or a.startswith("bias")
    ]

    return x_names, param_names


def tuning_optuna(ind, ppset, func, X, y):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # ---------------------------------------------------------------
    # 0 .  Data
    # ---------------------------------------------------------------
    # X must be shape (n_samples, n_input_vars) in *exact* x1…xN order.
    # y must be shape (n_samples,)
    # Example placeholders:
    # X = np.asarray([...])
    # y = np.asarray([...])

    # ---------------------------------------------------------------
    # 1 .  Identify inputs vs. free parameters
    # ---------------------------------------------------------------
    x_names, param_names = get_param_names(pset=ppset)

    n_x = len(x_names)  # = X.shape[1]
    assert X.shape[1] == n_x, f"X has {X.shape[1]} cols but {n_x} x-variables expected"

    # ---------------------------------------------------------------
    # 2 .  Objective for Optuna
    # ---------------------------------------------------------------
    def objective(trial):
        # 3a  suggest one float per θ / b in the *DECLARED ORDER*
        param_vals = [trial.suggest_float(name, -20, 20) for name in param_names]

        # 3b  evaluate row-by-row (GP lambdas need plain floats, not vectors)
        preds = []
        for row in X:  # row is length n_x
            args = list(row) + param_vals  # x1…xN θ0…θK b0…bK
            preds.append(func(*args))
        preds = np.asarray(preds)

        # 3c  return mean-squared-error (Optuna will minimise)
        return np.mean((preds - y) ** 2)

    # ---------------------------------------------------------------
    # 3 .  Run the study
    # ---------------------------------------------------------------
    study = optuna.create_study(direction="minimize")
    # 1) build the identity‐trial dict
    identity = {f"theta{k}": 1.0 for k in range(int(len(param_names) / 2))}
    identity.update({f"bias{k}": 0.0 for k in range(int(len(param_names) / 2))})

    # 2) force Optuna to try it
    study.enqueue_trial(
        identity
    )  # optuna will only finds the parameters that make better MSE than original tree.

    # 3) now run the real optimization
    study.optimize(objective, n_trials=1000, show_progress_bar=True)

    best_params = study.best_trial.params  # dict { 'theta0':…, 'bias0':…, … }
    best_mse = study.best_value

    print("⬇︎  Best affine parameters")
    for k in param_names:
        print(f"{k:7s} = {best_params[k]: .6f}")
    print(f"Best MSE = {best_mse:.6f}")

    return best_params, best_mse


def tuning_lm(ind, ppset, func, X, y):
    # ---------------------------------------------------------------
    # 0 .  Data (use your own X_train, Y_train)
    # ---------------------------------------------------------------
    # X  shape (n_samples, n_input_vars)
    # y  shape (n_samples,)

    # ---------------------------------------------------------------
    # 1 .  Compile GP individual
    # ---------------------------------------------------------------
    _, param_names = get_param_names(pset=ppset)

    # Initial guess: identity transform (theta=1, bias=0)
    initial_params = []
    bounds_lower, bounds_upper = [], []
    for pname in param_names:
        if pname.startswith("theta"):
            initial_params.append(1.0)
            bounds_lower.append(-20)
            bounds_upper.append(20)
        else:  # bias
            initial_params.append(0.0)
            bounds_lower.append(-20)
            bounds_upper.append(20)

    # ---------------------------------------------------------------
    # 2 .  Define residual function for Levenberg-Marquardt
    # ---------------------------------------------------------------
    def residuals(params, X, y):
        preds = []
        for row in X:
            args = list(row) + list(params)
            preds.append(func(*args))
        preds = np.asarray(preds)
        return preds - y  # residuals (not squared!)

    # ---------------------------------------------------------------
    # 3 .  Run Levenberg–Marquardt optimization
    # ---------------------------------------------------------------
    result = least_squares(
        residuals,
        x0=initial_params,
        args=(X, y),
        method="lm"  # Levenberg-Marquardt algorithm
        # LM algorithm doesn't support bounds directly; remove if 'lm' is used
        # If bounds are crucial, use method='trf' instead
    )

    best_params_lm = result.x
    best_mse_lm = np.mean(result.fun**2)

    # ---------------------------------------------------------------
    # 4 .  Report Results
    # ---------------------------------------------------------------
    print("⬇︎ Best affine parameters (LM):")
    for name, value in zip(param_names, best_params_lm):
        print(f"{name:7s} = {value: .6f}")
    print(f"Best MSE (LM) = {best_mse_lm:.6f}")

    return best_params_lm, best_mse_lm


def tuning_params(ind, ppset, X, y, algorithm="lm"):
    func = gp.compile(expr=ind, pset=ppset)
    if algorithm == "lm":
        print("Running Levenberg–Marquardt algorithm ----------------")
        best_params, best_mse = tuning_lm(ind, ppset, func, X, y)
    elif algorithm == "optuna":
        print("Running Optuna algorithm ----------------")
        best_params, best_mse = tuning_optuna(ind, ppset, func, X, y)
        best_params = np.array(
            list(best_params.values())
        )  # we need array of values for substitution
    else:
        print("Please choose the correct algorithm's name")

    _, param_names = get_param_names(pset=ppset)

    return best_params, best_mse, param_names


def evaluate_with_tuned_params(ind, ppset, X, y, best_params):
    # This function just checks if the parameter values are correct or not.

    _, param_names = get_param_names(pset=ppset)

    # Prepare parameters in correct order
    if isinstance(best_params, dict):
        optimized_params = [best_params[name] for name in param_names]
    else:
        optimized_params = [
            best_params[param_names.index(name)] for name in param_names
        ]

    # Compile the optimized GP individual
    optimized_func = gp.compile(expr=ind, pset=ppset)

    # Compute predictions with optimized params
    predictions = np.array(
        [optimized_func(*(list(row) + optimized_params)) for row in X]
    )

    # Compute MSE explicitly
    mse_optimized = np.mean((predictions - y) ** 2)

    print(f"MSE with optimized parameters: {mse_optimized:.6f}")

    # Identity parameters: theta=1, bias=0
    identity_params = []
    for pname in param_names:
        identity_params.append(1.0 if "theta" in pname else 0.0)

    pred_identity = np.array(
        [optimized_func(*(list(row) + identity_params)) for row in X]
    )

    mse_identity = np.mean((pred_identity - y) ** 2)

    print(f"MSE with identity parameters (original): {mse_identity:.6f}")
    print(f"Optimized vs identity MSE difference: {mse_identity - mse_optimized:.6f}")


def substitute_params_with_constants(tree, pset, best_params, param_names):
    """
    Replace thetas and biases parameters in the tree with constants
    using their values after optimization.
    """
    new_tree = gp.PrimitiveTree(tree)

    # Build fallback mapping: ARGi → actual name
    fallback_mapping = {f"ARG{i}": name for i, name in enumerate(pset.arguments)}
    param_values_dict = dict(zip(param_names, best_params))  # or from Optuna
    for i, node in enumerate(new_tree):
        if isinstance(node, gp.Terminal):
            if isinstance(node.value, str) or node.value is None:
                internal_name = node.name  # e.g., "ARG5"
                mapped_name = fallback_mapping.get(internal_name, internal_name)

                if mapped_name in param_values_dict:
                    value = param_values_dict[mapped_name]
                    new_tree[i] = gp.Terminal(value, False, ret=type(value))

    return new_tree


def tree_fit(ind, pset, algorithm, X, y):
    """
    This function receives a tree and fine-tune the parameters based on the data points
    and return the same tree with added tuned parameters
    """

    # Add tuning parameters to the tree
    new_str = add_tuning_params(str(ind))
    ppset, toolbox = problem_set_redefinition(pset=pset)
    new_tree = gp.PrimitiveTree.from_string(new_str, ppset)
    # Fine-tune the tree based on the data points
    best_params, best_mse, param_names = tuning_params(
        new_tree, ppset, X, y, algorithm=algorithm
    )
    # Substitute the tuned parameters in the tree and predict value for each data point
    tuned_tree = substitute_params_with_constants(
        new_tree, ppset, best_params, param_names
    )
    # Reset the problem set to make sure the tree is exactly the same as original tree

    return tuned_tree


def calc_Err_in(tree, X, y, algorithm="lm", B=5, n_features=5):
    # np.random.seed(seed)
    pset, _ = pset_definition(n_features)
    tuned_tree = tree_fit(tree, pset, algorithm, X, y)
    pset, _ = pset_definition(n_features)
    tuned_func = gp.compile(expr=tuned_tree, pset=pset)
    mu_hat = np.array([tuned_func(*row) for row in X])
    resid = y - mu_hat

    y_star = np.empty((len(y), B))
    mu_star = np.empty_like(y_star)

    for b in range(B):
        eps = np.random.choice(resid, size=len(resid), replace=True)
        y_s = mu_hat + eps
        boot_tree = tree_fit(tree, pset, algorithm, X, y_s)
        pset, _ = pset_definition(n_features)
        tuned_func_b = gp.compile(expr=boot_tree, pset=pset)
        mu_b = np.array([tuned_func_b(*row) for row in X])
        y_star[:, b], mu_star[:, b] = y_s, mu_b

    y_bar = y_star.mean(axis=1, keepdims=True)
    cov_i = (mu_star * (y_star - y_bar)).sum(axis=1) / (B - 1)
    err = np.sum((y - mu_hat) ** 2)
    optimism = 2 * cov_i.sum()
    Err = err + optimism
    print(f"\nApparent total squared error: {err:.4f}")
    print(f"Bootstrap optimism correction (2·∑ ĉov_i): {optimism:.4f}")
    print(f"Prediction‑error estimate (apparent + optimism): {Err:.4f}")
    return Err, err, optimism
