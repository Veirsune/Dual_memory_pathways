# Source release checks

The following checks were run from a clean checkout containing this source release:

- `python -m unittest discover -s tests -v`: 14 tests passed. Covers streaming causality, state reset, checkpoint/optimizer continuation, memory and CPU-LIF gradients, visual model behavior and gate causality.
- `python -m mnist_study.verify --device cpu --output /tmp/gdsm-mnist-verification.json`: passed. Candidate recurrence and gradients match the independent recurrent reference; original scalar DMP forward equivalence and all three families' finite training gradients pass. Numeric results are in `mnist-cpu.json`.
- Both new data preparation entry points accept `--help`; Python source compilation passed. Their encoding/split operations are preserved from the archived study scripts, with input/output paths exposed as arguments.
- Staged whitespace checks passed. Evidence snapshots retain their exact bytes and recorded SHA-256 hashes.

Hardware regression and trained-trace RTL results are the previously completed study results in `../paper/evidence/rtl-report.md`; publishing this code did not rerun the full hardware suite or retrain any model.
