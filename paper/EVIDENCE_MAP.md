# Manuscript evidence map

The manuscript describes completed experiments. Manuscript preparation performed no new model training and did not select new checkpoints.

| Paper component | Primary source under `/workspace/experiments/` | Packaged evidence |
|---|---|---|
| Clean SHD reproduction context | `shd-accuracy/REPORT.md` | `shd-clean-report.md` |
| SHD vector / wide-scalar control | `shd-vector-memory/metrics.csv`, `REPORT.md` | `shd-capacity.csv`, `shd-capacity-report.md` |
| SHD supervision ablation | `shd-supervision-ablation/seed-metrics.csv`, `protocol.json` | `shd-supervision.csv`, `shd-protocol.json` |
| SHD fixed/static/dynamic reading | `shd-final-readout/REPORT.md` | `shd-readout-report.md` |
| Gesture official-test comparison | `final-study/block3-summary.csv` | `gesture.csv`, `gesture-report.md`, `gesture-protocol.json` |
| Ordinary recognition and timed interference | Repository `large_scale/paired_effects/` | `paired-effects.csv`, `paired-intervals.csv`, `mnist.csv` |
| Carried word-prefix state on SSC | Repository `large_scale/ssc_matched200/midpoint/` | `ssc-midpoint.json`, `ssc-midpoint.npz` |
| Full SSC comparison | `block5-ssc/metrics.csv`, `paired-statistics.json` | `ssc.csv`, `ssc-statistics.json`, protocol/report |
| Full S/PS-MNIST comparison | `block6-mnist/metrics.csv`, `paired-statistics.json` | `mnist.csv`, `mnist-statistics.json`, protocol/report |
| Gesture operation estimates | `final-study/hardware-summary.csv` | `hardware-operations.csv`, `operations-report.md` |
| Fixed-point core and measured cycles | `block7-hardware/cycles.csv`, `results/trained-replay.json` | `hardware-cycles.csv`, `hardware-replay.json`, `rtl-report.md` |

The archived inputs listed by the asset builder have their SHA-256 recorded in `evidence/source-manifest.json`. The source snapshots support independently regenerating the main tables and plots. Rounded values in prose follow those source outputs; percentage-point differences are computed before rounding.

## Full experiment archives (remain in the research workspace)

| Archive | SHA-256 |
|---|---|
| `final-study/final-research-deliverable.tar.gz` | `9ada141a33b51271dd753d250d3fe6580b95d5726aa976937523dc1a2abf249b` |
| `block5-ssc/block5-ssc-deliverable.tar.gz` | `76b3b24796ac23cabaab09e51c7dc1a35a62dd4f98ad4f89ea4d4d000be9179d` |
| `block6-mnist/block6-mnist-deliverable.tar.gz` | `ae43931830d0d89b11ed35fcad15d3b409cee63dbf5b498337a7f0784901058e` |
| `block7-hardware/block7-hardware-deliverable.tar.gz` | `780977597683faad5988b506318486298b82914eaa44de8651bc73ec3cc6c779` |

The source ZIP includes the tables, selected per-example predictions and both manuscript sources. The repository links the trained checkpoints and evaluation code for the paired comparisons.

## Bibliographic checks

Crossref DOI metadata was retrieved for LSTM, GRU, the Heidelberg datasets, DVS Gesture, MNIST, surrogate-gradient learning and SpikingJelly; see `evidence/reference-metadata.json`. LMU, HiPPO, S4, GLIF and TC-LIF were cross-checked against the source paper's reference list. LMU, S4 and Mamba landing-page titles were also checked. The DMP reference is explicitly arXiv:2512.07602v3 (May 2, 2026); the manuscript does not assume a journal publication. The MNIST paper's year is 1998 and the Heidelberg journal issue is 2022, using verified metadata rather than copying inaccuracies in another reference list.

## Result selection

Table 4 covers the full official test sets of SHD, SSC, DVS Gesture, S-MNIST and PS-MNIST, with recognition scored at the event boundary. Historical SHD controls are used to explain mechanism and are labelled by their actual supervision recipe. Whole-sequence, endpoint, auxiliary-head, gate-coefficient, operation-estimate and RTL-cycle quantities remain separately identified. Further noise-shift and switching results are recorded in the corresponding study reports.

## Manuscript artifacts

The paper is `GDSM_Paper.pdf`, with editable source in `main.tex` and an independent source package in `GDSM_LaTeX.zip`. Section 2 defines the model and training objective in six numbered equations. Source metrics are preserved in the evidence files above.

Detailed protocols, exploratory diagnostics and complete supplementary tables are preserved in `EXPERIMENT_DETAILS.md`. The full middle-gap tables and the memory-core cycle plot are repository assets. Section 5 reports the measured replay cycles and the estimated full-network operation count.

## Training curves

Figure 3 uses clean-input, whole-sequence validation accuracy from 42 original stage histories, packaged under `evidence/training-history/`. Source locations and SHA-256 hashes are listed in `evidence/training-history-manifest.json`. `training-curve-points.csv` contains 96 aggregates. All seeds are included (SSC: 801; MNIST: 42, 43, 44), and uncertainty is sample standard deviation, not a confidence interval. The archived validation functions are in `training-history/ssc_study.py` and `training-history/mnist_run.py`. No corruption-specific validation trajectory was recorded, and no missing epochs or intermediate checkpoints are reconstructed.

## Notation alignment

The English and Chinese manuscripts now follow the original DMP Methods equations (7)–(11) and tensor-dimension table: layer l, step [k], membrane u, drive x, memory m, projection W_x, mapping f_x and matrices Abar/Bbar. The proposed model adds candidate p and write gate g and extends x to a vector. The implementation uses ReLU for f_x, as recorded in `NOTATION.md`. Fast and memory currents are written separately, with their combined implementation bias retained in the text. See `NOTATION.md` for the code-variable correspondence. No trained model or archived numerical result was changed.

The separate word-end placement experiment uses the official SSC test predictions in `evidence/ssc-position/predictions.npz`. The directory also contains the fixed protocol, inference script, plotted utterance and per-condition counts. From the paper directory, `make ssc-position` redraws the panels and regenerates the accuracy CSV.

Figure 4 uses `evidence/ssc-speech/` for matched whole-utterance recognition at four background rates. The directory includes the fixed protocol, predictions, first test input, evaluator and accuracy counts.

Figure 5 uses the matched within-event and post-event differences. The 200-epoch clean-trained SSC model in Table 4 and the 200+12 model in the midpoint intervention are separate fitted models. For the latter, only frames 126–250 contribute scores. The manuscript audit reconstructs the four midpoint accuracies and both net counts from the archived predictions.
