# SSC noise timing

The protocol fixes the three final mixed-stage checkpoints, the full official test set, the noise distribution and the five burst positions. Sample order is the original SSC test order. No checkpoint fitting or noise tuning is performed.

Each 250-frame utterance retains all its spike counts. For an overlap of r frames, the same 80-frame Poisson burst begins at frame 250-r (zero-based) and the sequence ends at 330-r. Counts add in overlapping bins. The readout averages the final 20 logits. Thus increasing overlap also shortens the noise-only tail and moves scoring closer to the word end. At full overlap only the last 80 speech frames are corrupted.

Use large_scale/prepare.py with the official SSC files to reconstruct test-x.npy and test-y.npy. Place the frozen mixed-stage final.pt files in the input directory under the names fsnn.pt, dmp.pt and commit.pt. The evaluator checks their SHA-256 values and the input hashes before inference. From the repository root run:

    PYTHONPATH=. python paper/evidence/ssc-position/evaluate.py INPUT_DIR OUTPUT_DIR

predictions.npz contains one prediction per utterance per checkpoint and condition. example.npz stores test utterance 0 and its exact burst. To redraw the figure and accuracy.csv from these two archives:

    python paper/scripts/draw_ssc_position.py

The run used one Colab Tesla T4. Clean top-1 predictions agree with the archived SSC run on all 20,382 utterances for FSNN and GDSM and on 20,381 for DMP. All three clean accuracies agree to two decimal places. The new experiment uses a fresh shared NumPy realization of the same Poisson distribution as the original tail experiment.
