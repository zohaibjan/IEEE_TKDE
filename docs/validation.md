# Executed validation

Date: 17 October 2026. [Successful GitHub Actions run 39000804720](https://github.com/zohaibjan/IEEE_TKDE/actions/runs/39000804720) at commit `485002c43383f9507368f668cd79cafcc96a0854`.

- MATLAB R2024b on Ubuntu 22.04.
- 22 tests passed; zero failed or incomplete.
- 29 root function files loaded/compiled by MATLAB without executing the experiment entry scripts.
- Installed toolboxes: Statistics and Machine Learning, Deep Learning, Global Optimization, Optimization (dependency), and Parallel Computing.

## What was executed

Actual SVM/tree fitting and ANN training, particleswarm consistency, bounded feature-only silhouette clustering, majority/BKS fusion and core utilities.

Toolbox tests fit real models and/or call real solvers. Deterministic predict-compatible fixtures are also used to isolate selection and voting behavior. Smoke tests use small synthetic datasets and bounded solver/epoch budgets; default research budgets remain available.

## Changes and behavior

BKS is combination method 10 (method 9 remains Dempster–Shafer). BKS matches observed training patterns directly instead of allocating an exponential table, uses the supplied training labels, and falls back to majority voting for unseen patterns. This is an implementation correction, not a reproduced research score.

## Reproduce

Use a clean MATLAB session with only this repository on the path:

```matlab
results = runtests('tests');
assertSuccess(results);
```

The workflow runs tests on MATLAB changes and pull requests, and supports manual dispatch. It has read-only repository permissions and performs no deployment.

## Limits

These tests do not execute every function branch or reproduce a paper's full datasets, published accuracy, runtime or memory results. Experiment scripts and unsupported auxiliary paths still need their documented inputs/dependencies. Root function loading is a compilation check, not full code coverage. Consult [code-review.md](code-review.md) for repository-specific limitations. Historical result CSVs were not regenerated or relabelled.
