#!/bin/bash

k="${1:-7}"  # use first arg, default to 3 if not given
python eval_modelsel.py "$k" BFGS_cluster_results_f1_100_10.csv "MSE_test_opt"
python eval_modelsel.py "$k" BFGS_cluster_results_f2_100_2.csv  "MSE_test_opt"
python eval_modelsel.py "$k" BFGS_cluster_results_f3_100_1.csv  "MSE_test_opt"
python eval_modelsel.py "$k" BFGS_cluster_results_f4_100_2.csv  "MSE_test_opt"
python eval_modelsel.py "$k" BFGS_cluster_results_f5_100_3.csv  "MSE_test_opt"
python eval_modelsel.py "$k" BFGS_cluster_results_f6_100_2.csv  "MSE_test_opt"
python eval_modelsel.py "$k" BFGS_cluster_results_f7_100_2.csv  "MSE_test_opt"