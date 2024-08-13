# Implicit Regularization of Attention Models Trained on Mirror Descent

This is the codebase for my SuperUROP project titled "Implicit Regularization of Attention Models Trained on Mirror Descent". To reproduce the results, we have the `bash_scripts` directory that has several bash code to run the experiments:
- `run-local-{p}.sh` for $p=$ 2, 3, and 1\_75
     These three bash scripts each trains the single-layer attention model with the $l_p$ Mirror Descent algorithm for 100 times for 100 different synthetically generated datasets, and calculates the 100 solutions for the $l_p$ -ATT-SVM problems. Once completed, it trains the model one more time for the first dataset but with the initialization being at zero. All results will be saved in the `result/convergence/p{p}` folder.
- `run-correlations-W.sh`
     This bash script is used to calculate the correlation coefficient between the iterates of the $l_p$ Mirror Descent algorithm and the $l_q$-ATT-SVM solutions for all $p,q\in\set\{1.75,2,3\}$, results are saved in `result/correlation/{q}-{p}`

The above bash scripts presents the results in the result directory, which can be visualized in the `plotter.ipynb` file.

## Acknowledgement

Certain parts of the code is adapted from the [TF-as-SVM](https://github.com/umich-sota/TF-as-SVM) repository
