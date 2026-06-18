# RiLO-PGO Compact Reference Implementation

This repository is a compact public reference package for RiLO-PGO:

> Loop-outlier-aware Riemannian pose-graph optimization for robust visual localization.

The package is intentionally small. It is designed to make the proposed
backend refinement mechanism inspectable and runnable without exposing the
full internal research repository, historical experiment scripts, paper-only
plotting utilities, or unrelated baseline implementations.

## What Is Included

- Core RiLO-PGO MATLAB implementation.
- PieADMM-style floor solving used as the default floor path in this package.
- LM refinement from the floor solution.
- Loop-specific residual scoring.
- Soft and hard loop-edge reweighting.
- Unified conservative candidate selection.
- Two representative `.g2o` pose-graph examples for a quick smoke test.

The full candidate pool implemented here is:

1. `pie_floor`
2. `pie_unweighted_lm`
3. `pie_loop_reweight_soft`
4. `pie_loop_reweight_hard`

All candidates are evaluated on the same original unweighted graph metrics.
The floor solution is always retained as a fallback.

## What Is Not Included

- Complete internal experiment-management scripts.
- Historical drafts and cached paper figures.
- Full plotting pipelines used only for manuscript preparation.
- Large raw-data archives.
- Complete third-party SLAM systems.
- Baseline implementations that are not needed to inspect the RiLO-PGO logic.

This scope keeps the release focused on the algorithm introduced in the paper.

## Folder Layout

```text
RiLO-PGO/
  README.md
  LICENSE
  main_run_single_case.m
  main_run_minimal_suite.m
  datasets/
    minimal_g2o/
      parking-garage__loop_outlier_10.g2o
      tinyGrid3D__loop_outlier_10.g2o
  results/
  src/
    posegraphSLAM_Ours.m
    posegraphSLAM_gd.m
    posegraphSLAM_LM.m
    parse_g2o_se3quat.m
    metric_edge_sum.m
    metric_fml.m
    ...
```

## Quick Start

Open MATLAB in this folder and run:

```matlab
main_run_single_case
```

This script loads one disturbed pose graph, runs the RiLO-PGO backend
refinement pipeline, and writes:

- `optimized.g2o`
- `metrics.txt`
- `result.mat`
- `traj3d.png`

to `results/single_case/`.

To run both public example graphs:

```matlab
main_run_minimal_suite
```

The batch summary is written to `results/suite_run/summary.csv`.

## Conservative Selection Rule

The public implementation uses one fixed candidate-selection rule for all
examples. The default thresholds are:

```text
eps_f_degrade  = 0.01
eps_theta_max  = 0.15
eps_theta_soft = 0.05
delta_f        = 0.01
delta_t        = 0.10
```

For each non-floor candidate, the rule rejects the candidate if it degrades
the original graph objective or rotational consistency beyond the above
tolerances. Among admissible candidates, the selected solution is the one
with the smallest original-graph `fML`; ties are broken by smaller
translation loss and then by the less aggressive candidate order:

```text
unweighted LM -> soft weighted LM -> hard weighted LM
```

## Notes

- The public examples are intended as runnable algorithm demonstrations, not
  as a replacement for the complete experimental archive.
- Runtime values from this compact package should be interpreted as practical
  end-to-end reference timings for the included implementation and machine.
- If redistribution is permitted, additional processed pose-graph inputs can
  be added under `datasets/minimal_g2o/` without changing the entry scripts.

## License

This compact reference implementation is released under the MIT License.
