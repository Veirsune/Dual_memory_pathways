# SSC word recognition with continuous background spikes

For each official SSC test utterance, the sequence contains the original 250 speech frames. Independent synthetic Poisson events are added to every channel and frame, without clipping the resulting counts. The burst generator does not insert time or shift the speech. The classifier averages the 250 score vectors and picks the highest-scoring word. Model weights come from the original mixed training run (seed 801); the networks receive identical inputs.

The three noise increments use independent PCG64 streams. Reusing each increment at higher rates keeps the input paired across rates. The evaluation script verifies the data and checkpoint hashes recorded in protocol.json. To rerun, prepare the input directory as described in ../ssc-position/README.md, then run from the repository root:

    PYTHONPATH=. python paper/evidence/ssc-speech/evaluate.py INPUT_DIR OUTPUT_DIR

The archived predictions and example are sufficient to redraw the figure using the following command:

    python paper/scripts/draw_ssc_speech.py
