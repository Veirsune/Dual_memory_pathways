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

## Revision v2

Figure 3 uses full-network effective MAC estimates for clean Gesture and appended L80 silence, generated from `hardware-operations.csv`. Exact values and arithmetic are in `full-network-mac-claims.json`. The measured memory-core cycle figure is retained in the appendix. Evaluation chronology and distribution-shift details are retained in the appendix; the Discussion ends with two future-direction sentences.

## Naming revision v3

The manuscript names the architecture Gated Dual-State Memory (GDSM). Historical result keys and source reports retain `commit`; this is a presentation-only rename of the same evaluated model. No weights, metrics or protocols changed.
