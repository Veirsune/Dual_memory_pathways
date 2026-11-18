# Matched SSC training

Train FSNN, the released two-layer SSC DMP and GDSM from scratch. The recognition comparison uses 200 clean epochs for every model. A separate 12-epoch disturbance continuation starts from each model's own best validation checkpoint. See protocol.json for the settings fixed before training. The earlier paper's 40+12 runs remain a separate experiment.

Prepare SSC with prepare.py RAW_DIR DATA_DIR SPLIT for train, valid and test. DATA_DIR must exist. Training expects those files in $SSC_ROOT/data. Put the released ssc/dual_pathway.py on the Python import path as upstream_ssc.py and include the project root on PYTHONPATH. Use the project's SpikingJelly clock_driven environment (PyTorch 2.0, SpikingJelly 0.0.0.0.14 with CuPy for CUDA). Run train.py with SSC_ROOT set to the experiment directory. The script writes per-epoch validation, resumable optimizer checkpoints, and the selected model for each phase into results/. After training it writes test.json for all six selected checkpoints.

For the two-model 256-wide clean recognition comparison, set SSC_PROTOCOL to the absolute path of width256.json and set SSC_ROOT to a fresh experiment directory containing the same data. Both models train for 200 epochs. Each history records training and validation time in seconds per epoch.
