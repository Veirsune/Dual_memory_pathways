# Manuscript evidence map

The manuscript describes completed experiments. Manuscript preparation performed no new model training and did not select new checkpoints.

| Paper component | Primary source under `/workspace/experiments/` | Packaged evidence |
|---|---|---|
| Clean SHD reproduction context | `shd-accuracy/REPORT.md` | `shd-clean-report.md` |
| SHD vector / wide-scalar control | `shd-vector-memory/metrics.csv`, `REPORT.md` | `shd-capacity.csv`, `shd-capacity-report.md` |
| SHD supervision ablation | `shd-supervision-ablation/seed-metrics.csv`, `protocol.json` | `shd-supervision.csv`, `shd-protocol.json` |
| SHD fixed/static/dynamic reading | `shd-final-readout/REPORT.md` | `shd-readout-report.md` |
| Gesture official-test comparison | `final-study/block3-summary.csv` | `gesture.csv`, `gesture-report.md`, `gesture-protocol.json` |
| Full SSC comparison | `block5-ssc/metrics.csv`, `paired-statistics.json` | `ssc.csv`, `ssc-statistics.json`, protocol/report |
| Full S/PS-MNIST comparison | `block6-mnist/metrics.csv`, `paired-statistics.json` | `mnist.csv`, `mnist-statistics.json`, protocol/report |
| Gesture operation estimates | `final-study/hardware-summary.csv` | `hardware-operations.csv`, `operations-report.md` |
| Fixed-point core and measured cycles | `block7-hardware/cycles.csv`, `results/trained-replay.json` | `hardware-cycles.csv`, `hardware-replay.json`, `rtl-report.md` |

Every copied file has its SHA-256 recorded in `evidence/source-manifest.json`. The source snapshots support independently regenerating the main tables and plots. Rounded values in prose follow those source outputs; percentage-point differences are computed before rounding.

## Full experiment archives (remain in the research workspace)

| Archive | SHA-256 |
|---|---|
| `final-study/final-research-deliverable.tar.gz` | `9ada141a33b51271dd753d250d3fe6580b95d5726aa976937523dc1a2abf249b` |
| `block5-ssc/block5-ssc-deliverable.tar.gz` | `76b3b24796ac23cabaab09e51c7dc1a35a62dd4f98ad4f89ea4d4d000be9179d` |
| `block6-mnist/block6-mnist-deliverable.tar.gz` | `ae43931830d0d89b11ed35fcad15d3b409cee63dbf5b498337a7f0784901058e` |
| `block7-hardware/block7-hardware-deliverable.tar.gz` | `780977597683faad5988b506318486298b82914eaa44de8651bc73ec3cc6c779` |

The small manuscript ZIP does not duplicate trained checkpoints, full per-example predictions, datasets, or large waveforms. Those remain available in the listed archives. The manuscript's numerical evidence is included directly as compact CSV/JSON snapshots.

## Bibliographic checks

Crossref DOI metadata was retrieved for LSTM, GRU, the Heidelberg datasets, DVS Gesture, MNIST, surrogate-gradient learning and SpikingJelly; see `evidence/reference-metadata.json`. LMU, HiPPO, S4, GLIF and TC-LIF were cross-checked against the source paper's reference list. LMU, S4 and Mamba landing-page titles were also checked. The DMP reference is explicitly arXiv:2512.07602v3 (May 2, 2026); the manuscript does not assume a journal publication. The MNIST paper's year is 1998 and the Heidelberg journal issue is 2022, using verified metadata rather than copying inaccuracies in another reference list.

## Result selection

The main cross-dataset tables use the full frozen SSC, Gesture and MNIST comparisons, not pilot subsets or best seeds. Historical SHD controls are used to explain mechanism and are labelled by their actual supervision recipe. Whole-sequence, endpoint, auxiliary-head, gate-coefficient, operation-estimate and RTL-cycle quantities remain separately identified. Strong-noise failures and the final readout study are included in the text/appendix.

## Manuscript artifacts

The paper is `GDSM_Paper.pdf`, with editable source in `main.tex` and an independent source package in `GDSM_LaTeX.zip`. It contains six core method equations, four figures, seven tables and 16 references. Source metrics are preserved in the evidence files above.

Detailed protocols, exploratory diagnostics and complete supplementary tables are preserved in `EXPERIMENT_DETAILS.md`. The PDF retains the longest middle-gap condition for each dataset and both disturbance types, without changing any recorded values. The memory-core cycle plot remains a repository asset; the PDF reports its key counts in Appendix A.

## Training curves

Figure 3 uses clean-input, whole-sequence validation accuracy from 42 original stage histories, packaged under `evidence/training-history/`. Source locations and SHA-256 hashes are listed in `evidence/training-history-manifest.json`. `training-curve-points.csv` contains 96 aggregates. All seeds are included (SSC: 801; MNIST: 42, 43, 44), and uncertainty is sample standard deviation, not a confidence interval. The archived validation functions are in `training-history/ssc_study.py` and `training-history/mnist_run.py`. No corruption-specific validation trajectory was recorded, and no missing epochs or intermediate checkpoints are reconstructed.

## Notation alignment

The English and Chinese manuscripts now follow the original DMP Methods equations (7)–(11) and tensor-dimension table: layer l, step [k], membrane u, drive x, memory m, projection W_x, mapping f_x and matrices Abar/Bbar. The proposed model adds candidate p and write gate g and extends x to a vector. ReLU remains documented as the actual implementation of f_x in Appendix A. Fast and memory currents are written separately, with their combined implementation bias retained in the text. See `NOTATION.md` for the code-variable correspondence. No trained model or archived numerical result was changed.

Figure 4 uses the official SSC test predictions in `evidence/ssc-position/predictions.npz`. The directory also contains the fixed protocol, inference script, plotted utterance and per-condition counts. From the paper directory, `make ssc-position` redraws the panels and regenerates the accuracy CSV.
