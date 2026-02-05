# Model Selection in Symbolic Regression
This reposity includes the final version of the codes for generating the results of paper called "A Comparative Study of Model Selection Criteria for Symbolic Regression".
Please go step-by-step through the code based on the guide as follows:

## Dataset Construction

Change the directory into the cloned repository and run the following code to generate the datasets used in the paper:

```bash
cd src
python generate_datasets.py
```
This script will generate the datasets for each of the ground-truth functions and save them in ``` src/data ``` folder.


## Symbolic Model Generation

To generate the symbolic perturbed models from each of the ground-truth function, run the following code:

```bash
python export_pop.py --n_m (#number_of_mutations) --n_f (#number_of_features) --f_n (#function_Name)
```

#number_of_mutations : The number of required models to be generated from the ground-truth function

#number_of_features : The number of features of the function. For f1, we used 10 features instead of original 5.

#function_Name : The number of ground-truth function based on the order presented in the paper (f1-f7).

The output of export_pop.py will be an *.operon file that contains all expressions generated from that ground-truth function with file name as
"(function-name)_(number-of-mutations)_(number-of-features).operon".

For example, for generating 100 models for function ``` f1 ```, you can use the following code:

```bash
python export_pop.py --n_m 100 --n_f 10 --f_n f1
```

You can easily generate the models for all ground-truth functions by running the following shell script from the same folder:

```bash
chmod +x export_pop_all.sh
./export_pop_all.sh
```

This script uses the same ``` export_pop.py ``` as above, and generates 100 perturbations from each ground truth function and save them separately as .operon files.


## Compute Selection Metrics

After generating the datasets and perturbed models from the ground-truth functions, we have to calculate the ranking of the models based on each selection metric. You can do so by running the following script for each of the functions:

```bash
python srtools.py (#function_Name)
```
You can write the name of the functions that you would like to calculate the metric for, or just leave it empty, so the script will calculate the rankings for all of the seven functions. The output of this script is a ``` .csv ``` file with the name like ``` BFGS_cluster_results_f1_100_10.csv ```. For example, to calculate the metrics for function 1, you can use the following code:

```bash
python srtools.py f1
```

## Performance Calculation of Each Metrics
In this step, we have the ranking results for each of the functions based on each of the selection metrics. To see how good a metric is, we need to evaluate each of the four performance metrics for each of the seven functions based on each selection metric. The related script should be run as follows:

```bash
python eval_modelsel.py (#k) (csv_file_from_previous_stage) (name_of_the_test_column)
```

This script evaluates the performance of the rankings based on the ground-truth noiseless target ranking that in our case comes from ``` "MSE_test_opt" ``` column. ``` #k ``` shows the number of models from each function that we want to evaluate the performance for. For instance, to evaluate the performance on results for the first best ``` 50 ``` models for ``` f1 ```, you can run the following script:

```bash
python eval_modelsel.py "50" BFGS_cluster_results_f1_100_10.csv "MSE_test_opt"
```

The output ``` .csv ``` files will be saved in ``` results ``` folder.

If you want to have the evaluation for all the functions, then just run the following shell:

```bash
chmod +x eval_modelsel.sh
./eval_modelsel.sh
```

## Plotting the Final Results

The final step is to depict the results in a more intuitive way. Use the following code to run the plotting script:

```bash
python perf_plot.py
```
The output plots will be saved in a ``` .pdf ``` format in ``` plots ``` folder.
