# Midpoint stable-state intervention on SSC

One frozen mixed-trained GDSM (200 clean epochs plus 12 mixed epochs). All 20,382 official test utterances have 250 frames. Frames 1–125 are clean. Frames 126–250 are either clean or corrupted with independent additive Poisson counts, rate 0.125 per channel per frame. Both arms start from exactly the same saved prefix states and receive identical suffix frames. The reset arm zeros both stable states once before frame 126. All other states carry forward. Writes resume immediately, including the write from the previous candidate. Predictions use only the mean logits over frames 126–250.

| Suffix | Keep m | Reset m | Keep − reset, pp [95% CI] | Only keep correct | Only reset correct |
|---|---:|---:|---:|---:|---:|
| Clean | 32.83% | 28.84% | +3.99 [+3.44, +4.55] | 1899 | 1086 |
| Poisson 0.125 | 17.73% | 15.33% | +2.40 [+1.89, +2.93] | 1324 | 835 |

The difference between the two keep-minus-reset effects (noisy minus clean) is -1.59 pp [-2.24, -0.96]. All intervals resample the same 498 speaker clusters across arms and conditions, holding the checkpoint and noise realization fixed. Resetting a learned state to zero tests the total effect of carrying that state across this boundary. It does not identify its semantic content or rule out a transient response to the reset.

Prediction files store the test labels, speaker groups and four paired predictions per utterance. The normal batched implementation and an independent batched implementation of the reset were used to check the stepwise suffix predictions on the first 16 samples per condition. See results.json for numerical checks. To reproduce, use the repository's SpikingJelly environment, the matched sum-pooled test frames prepared by ../prepare.py, and the official raw HDF5 test file for speaker IDs. Run from the repository root:

    PYTHONPATH=. python large_scale/ssc_matched200/midpoint/evaluate.py DATA_DIR RAW_TEST_H5 large_scale/ssc_matched200/midpoint/gdsm-mixed-best.pt --batch 256 --threads 3
