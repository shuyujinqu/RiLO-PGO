# RiLO-PGO Open-Source Package

This package is the compact public version of the project.

It keeps only:

- the RiLO-PGO pipeline and the utility functions it needs
- a disturbed SE(3) `.g2o` benchmark suite for quick testing
- two clean entry scripts for public use

It intentionally removes:

- comparison pipelines for other back-end methods
- paper-only plotting scripts
- historical drafts, cached figures, and temporary experiments
- unrelated branches such as EuRoC maintenance code and old archive folders

## Folder Layout

```text
rilo_pgo_open_source/
  README.md
  main_run_single_case.m
  main_run_rilo_suite.m
  datasets/
    robust_suite_g2o/
  results/
  src/
    run_rilo_case.m
    posegraphSLAM_Ours.m
    posegraphSLAM_RiemannianAA_GNC_vMF.m
    posegraphSLAM_gd.m
    posegraphSLAM_LM.m
    ...
```

## Quick Start

Open MATLAB in `rilo_pgo_open_source` and run:

```matlab
main_run_single_case
```

This will:

- load one disturbed graph from `datasets/robust_suite_g2o`
- optimize it with RiLO-PGO
- save `optimized.g2o`, `metrics.txt`, `result.mat`, and `traj3d.png`

## Batch Run

To process the full disturbed suite:

```matlab
main_run_rilo_suite
```

Outputs are written to:

- `results/suite_run/rilo_suite_results.mat`
- `results/suite_run/summary.csv`
- one subfolder per dataset with the optimized graph and a simple figure

## Notes

- The public package defaults to `use_sl = false` to keep setup simple.
- `posegraphSLAM_Ours.m` still contains the original internal structure, including the PieADMM floor and weighted LM refinement used by RiLO-PGO.
- If you later want to add SL-based hyperparameter prediction back into the public package, the current structure leaves room for that extension without changing the entry scripts.
