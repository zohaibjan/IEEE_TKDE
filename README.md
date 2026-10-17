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

The silhouette search now excludes the label column, bounds K to available distinct feature rows, handles tiny classes and records feature-only centroids. Balancing and routing use the same raw feature space. The outer `eachClass` sweep is still overridden by silhouette selection; the CSV setting is not the fitted per-class K. ANN outputs now map through the training label vocabulary. PSO uses the same >0.4 selection rule during scoring and return, with a deterministic nonempty fallback. Auxiliary `accuracyOfPSO.m` still references an unprovided `confusionmatStats`; it is not the main path.

## Runtime follow-up — October 2026

BKS is combination method 10 (method 9 remains Dempster–Shafer). BKS matches observed training patterns directly instead of allocating an exponential table, uses the supplied training labels, and falls back to majority voting for unseen patterns. This is an implementation correction, not a reproduced research score.

## Correctness review — October 2026

This pass fixes implementation defects; it does not validate the associated research results. Changes can alter model pools, predictions and measured scores. Re-run the experiments before comparing corrected code with historical CSVs or published results.

- Label conversion uses numeric category names, not categorical indices.
- Voting rejects empty/nonfinite inputs; a prediction failure is reported rather than counted as a fabricated class-one vote.
- Neural models store their training labels in `net.userdata.classLabels`. Retrain old models or restore that exact training vocabulary before prediction; inferring it from a validation subset is unsafe.
- Synthetic checks cover normalization, CSV append behavior and the new prediction/selection boundaries. Additional cluster checks run where applicable.

See [the code review](docs/code-review.md) for repository-specific changes, tests and remaining limitations.

**Validation:** 22 regression/smoke tests passed in MATLAB R2024b on Linux through [GitHub Actions](https://github.com/zohaibjan/IEEE_TKDE/actions/workflows/matlab-tests.yml). Tests include real toolbox execution on synthetic data. Full research datasets and published accuracy results have not been reproduced. See [validation evidence and limits](docs/validation.md).

[Profile and project directory](https://github.com/zohaibjan)
