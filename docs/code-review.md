# Code review and validation scope

The silhouette search now excludes the label column, bounds K to available distinct feature rows, handles tiny classes and records feature-only centroids. Balancing and routing use the same raw feature space. The outer `eachClass` sweep is still overridden by silhouette selection; the CSV setting is not the fitted per-class K. ANN outputs now map through the training label vocabulary. PSO uses the same >0.4 selection rule during scoring and return, with a deterministic nonempty fallback. Auxiliary `accuracyOfPSO.m` still references an unprovided `confusionmatStats`; it is not the main path.

## Changed behavior and acceptance

The fixes preserve original source attribution and the documented experiment entry points. They change previously erroneous edge cases and may change reported results. Stored outputs were not rewritten or relabelled as new results.

Run in a clean MATLAB session with only this repository on the path:

```matlab
results = runtests('tests');
assertSuccess(results);
```

The test fixtures are synthetic. `FixedClassifier`, where supplied, implements `predict` without fitting a model and is on the path only for the test session. Majority ties choose the smallest numeric label. Empty or invalid votes fail explicitly. ANN vocabularies are learned from training labels, not held-out labels. PSO thresholds retain the repository's setting, with one shared nonempty mask policy for objective scoring and final return. Random selection, where used, has at most pool-size times elimination-budget attempts.

## Executed MATLAB validation

Actual SVM/tree fitting and ANN training, particleswarm consistency, bounded feature-only silhouette clustering, majority/BKS fusion and core utilities.

BKS is combination method 10 (method 9 remains Dempster–Shafer). BKS matches observed training patterns directly instead of allocating an exponential table, uses the supplied training labels, and falls back to majority voting for unseen patterns. This is an implementation correction, not a reproduced research score.

See [validation.md](validation.md) for the exact run, release and boundaries.

## Remaining validation

- Compatibility beyond the tested MATLAB R2024b/toolbox release.
- Full training runs with the documented datasets, preserving training/validation/test separation.
- Numerical quality, random seeds, paper-method agreement, runtime and memory consumption.
- Optional historical optimizer/diversity/alternate-driver files are not a validated library.

The regression/smoke suite passed in MATLAB R2024b using MathWorks GitHub Actions. Synthetic fixtures check implementation behavior; full research experiments, historical accuracy and performance benchmarks have not been reproduced.
