# Optimised Ensemble Classification with PSO

MATLAB research implementation of ensemble classification using particle swarm optimisation and class-based clustering.

## Research reference and attribution

Z. Jan, J. C. Munoz, and A. Ali, "A Novel Method for Creating an Optimized Ensemble Classifier by Introducing Cluster Size Reduction and Diversity", IEEE Transactions on Knowledge and Data Engineering, 2020.

Please cite the relevant publication and retain source author notices. Existing dataset and third-party terms still apply; this cleanup does not grant a new licence.

## Data and environment

Bundled `P-Data/*.mat` files contain numeric feature matrix `X` (samples × features) and label column `y` (samples × 1). The driver loads the dataset names listed in `Problem` from the MATLAB path.

**MATLAB dependencies:** Statistics and Machine Learning Toolbox; Deep Learning Toolbox; Global Optimization Toolbox (`particleswarm`).

No exact original MATLAB/toolbox version is recorded. Use MATLAB rather than assuming GNU Octave compatibility for the learner APIs.

## Start here

1. Open this repository as the MATLAB current folder. Use a separate session/path for each research repository: helper filenames overlap across projects.
2. Supply or extract the data described above. Check feature/label row counts and class labels.
3. Edit `Problem` in `mainProgram.m` (or `dataSetNames.m` where used), and start with one dataset and one repeat.
4. Run the synthetic utility checks, then inspect `mainProgram.m` before starting the full experiment:

```matlab
results = runtests('tests');
assertSuccess(results);
rng(42, 'twister');
mainProgram;
```

The seed is a starting point for recording runs; nested seed resets, parallel workers and toolbox versions can still affect reproducibility. `runMultiple.m` performs 50 full driver repetitions and is not the quick-start command.

## Workflow and output

`mainProgram.m` selects datasets and the 2–10 outer experiment setting. `runTraining.m` creates ten folds and a validation holdout. `generateClustersV2.m` chooses per-class cluster counts; `trainClassifiers.m` builds learners; `classifierSelectionPSO.m` selects an ensemble; `fusion.m` scores held-out predictions.

`saveResults.m` appends rows to `results.csv` in the current folder and writes the header only when creating the file. Driver-level CSV writes are sequential to avoid concurrent append races. Archive old output before a new configuration; rows do not contain a complete environment/configuration record. Committed CSV files are historical outputs, not freshly validated benchmarks.

## Review notes and reproducibility limits

The outer `eachClass` sweep is overridden by silhouette-based selection inside `generateClustersV2.m`; do not interpret the CSV K column as the actual chosen per-class cluster count. The historical generator clusters labelled rows and mixes raw/scaled centroid conventions. Review that method against the paper before changing it. ANN predictions use output indices, so verify contiguous labels starting at 1. Auxiliary `accuracyOfPSO.m` refers to `confusionmatStats`, which is not supplied here.

The cleanup aligns function/file names, restores visible warnings, makes CSV handles close on errors, and tests constant-feature normalisation plus CSV append behaviour. MATLAB experiments and the new utility tests have **not** been executed in the cleanup environment, which has no MATLAB installation. Static parsing is not a reproduction of the paper.

[Profile and project directory](https://github.com/zohaibjan)
