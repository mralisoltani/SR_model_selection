import sys
from collections import defaultdict
import numpy as np
import pandas as pd


# minimization
def avg_test_error(df, criteria, ixs_true, ixs_pred, g_idx):
    return df.iloc[ixs_pred].MSE_test_opt.mean()


# maximization
def precision_at_k(df, criteria, ixs_true, ixs_pred, g_idx):
    return len(set(ixs_true).intersection(ixs_pred)) / len(ixs_pred)


def win_check(df, criteria, ixs_true, ixs_pred, g_idx):
    g_set = set(g_idx)
    pred_orig_ids = set(
        df.iloc[ixs_pred]["Index"].to_numpy()
    )  # use original stable IDs

    return 1.0 if (pred_orig_ids & g_set) else 0.0


def avg_size(df, criteria, ixs_true, ixs_pred, g_idx):
    return df.iloc[ixs_pred]["Number_of_nodes"].mean()


if len(sys.argv) < 4:
    print("Usage: python eval_modelsel.py k csv-file test-col")
    # python eval_modelsel.py 3 results_f1_100_10_WO.csv "SSE_test_orig"
    sys.exit(-1)

# maximum number of top-k solutions
k = int(sys.argv[1])
# CSV file with the data
csv_file = sys.argv[2]
# column that contains the test set measure used to select the true best model
test_col = sys.argv[3]
name = csv_file.split(".csv")[0]
# ground truth index
ground_truth_index = [100]


# columns of the CSV file with the results and also the criteria to evaluate
criterias = [
    "MSE_train_opt",
    "AIC",
    "BIC",
    "AICc",
    "MDL",
    "Err_in",
]


df = pd.read_csv(csv_file)
df["Index"] = df.index  # original (stable) IDs for finding win_check

# now you can clean/filter/reset safely
df = df.replace([np.inf, -np.inf, 2e301], np.nan).dropna().reset_index(drop=True)
q_high = df[test_col].quantile(0.95)
print(f"Filtering out models with {test_col} > {q_high}")
df = df[df[test_col] <= q_high].reset_index(drop=True)


for i in range(1, k + 1):
    results = defaultdict(list)
    # evaluate the metrics for different k
    true_set = df[test_col].nsmallest(i).index.values
    # which measures to calculate
    measures = {
        "avg_test_error": avg_test_error,
        "precision_at_k": precision_at_k,
        "ground_truth_hit": win_check,
        "avg_size": avg_size,
    }
    for criteria in criterias:
        c_set = df[criteria].nsmallest(i).index.values

        results["dataset"].append(name)
        results["criteria"].append(criteria)
        for m_name, m_fun in measures.items():
            results[m_name].append(
                m_fun(df, criteria, true_set, c_set, ground_truth_index)
            )

    results = pd.DataFrame(results)
    # results.to_csv(f"results/report_{name}_{i}.csv")
