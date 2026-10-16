# Optimised Ensemble Classification with PSO

MATLAB research implementation of ensemble classification using particle swarm optimisation and class-based clustering.

## Research reference

Z. Jan, J. C. Munoz, and A. Ali, "A Novel Method for Creating an Optimized Ensemble Classifier by Introducing Cluster Size Reduction and Diversity", IEEE Transactions on Knowledge and Data Engineering, 2020.

Please cite the associated paper when using this work.

## Start here

1. Open the repository in MATLAB.
2. Review the datasets and experiment configuration in `mainProgram.m`.
3. Ensure the required machine learning, neural network and optimisation toolboxes are installed.
4. Run `mainProgram` from the repository directory.

The `Problem` variable selects datasets. The existing experiment varies the cluster setting from 2 to 10 and saves cross-validation results as CSV. Very large cluster counts can leave too few samples to train classifiers.

## Code map

| File | Responsibility |
|---|---|
| `mainProgram.m` | Experiment entry point and dataset selection |
| `runTraining.m` | Training workflow |
| `trainClassifiers.m` | Base classifier construction |
| `classifierSelectionPSO.m` | Optimisation-based classifier selection |
| `fusion.m`, `fusionPSO.m` | Ensemble prediction combination |
| `saveResults.m` | Experiment output |

## Reproducibility

This is a historical research implementation. MATLAB/toolbox versions and the selected datasets affect reproducibility; validate the environment before comparing results. Repository organisation updates do not constitute a rerun or independent reproduction of the paper.

[Profile and project directory](https://github.com/zohaibjan)
