# Experiment and implementation details

Supporting protocols and diagnostics for the paper. Numerical records are in [evidence/](evidence/); complete clean and middle-interruption tables are in [tables/clean.tex](tables/clean.tex) and [tables/middle.tex](tables/middle.tex). This document is not part of the paper PDF.

## Implementation and training

The models are implemented in SpikingJelly (bibliography keys: `spikingjelly`) and trained with surrogate gradients (bibliography keys: `surrogate`).

### State dynamics and initialization
All neural, memory and feature-history states start at zero for a new sequence. An inserted interruption does not reset any state. The candidate transition uses the original DMP Legendre construction and zero-order-hold discretization with a unit time step. For indices $i,j=0,\ldots,d-1$, its continuous parameters are
$$
 A^{\mathrm c}_{ij}=\frac{2i+1}{\theta}
 \begin{cases}-1,&i<j,\\(-1)^{i-j+1},&i\ge j,\end{cases}
 \qquad b^{\mathrm c}_i=\frac{(2i+1)(-1)^i}{\theta}.
$$
Discretization gives $\bar A$ and $\bar b$. The implementation stores a projection $\widetilde W_u$, bias $\widetilde b_u$ and fixed diagonal scaling $D$, with $D_{ii}=\bar b_i/c_i$ and positive constants $c_i$ evenly spaced from 0.5 to 1.5. We use the equivalent notation $W_u=C^{-1}\widetilde W_u$ and $b_u=C^{-1}\widetilde b_u$, where $C=\operatorname{diag}(c_i)$. Positive homogeneity of ReLU gives $D\operatorname{ReLU}(\widetilde W_u s_t+\widetilde b_u)=\bar b\odot\operatorname{ReLU}(W_u s_t+b_u)$, so Eq. (2) describes exactly the implemented drive. For initialization, stored row $i$ and its bias are $c_i$ times the scalar DMP weights and bias; the effective initial drive therefore matches DMP. The projection rows subsequently train independently. In paired SSC and MNIST starts, $W_m$ is initialized to zero so that the additional stable read does not alter the initial current.

Local gate weights start at zero, with bias $\log(0.95/0.05)$. The permission MLP's output weights start at zero and its output bias at $+2$ in the task-supervised comparisons. These initially open gates learn to close through training. Historical SHD protection studies use separately calibrated initialization.

### Gate features
Local activity normalization uses a scale bounded below by 0.05. Mean activity and absolute mean change are divided by this scale; variance is divided by its square. Nonzero occupancy supplies the fourth feature. Equation (5) concatenates these features with their previous EMA.

The permission MLP uses raw-input features and their previous EMA. For auditory input, four global statistics and 14 pooled frequency bands produce a $36\to32\to2d$ MLP, whose output is split between the two hidden layers. Gesture uses four global statistics and $2\times4\times4$ polarity/spatial bins, giving a $72\to32\to d$ MLP; pooled frame counts are divided by a further 16 for these features. MNIST uses the four scalar-pixel statistics and an $8\to32\to2d$ MLP. Pixel variance is identically zero but the shared feature construction is retained. Each MLP uses a ReLU hidden layer. Permission is one when its output logit is nonnegative; its backward derivative is replaced by the sigmoid derivative.

### SSC and Gesture
SSC events are binned into 250 frames between each utterance's first and last event. Consecutive groups of five channels are reduced by a maximum, yielding 140 channels. The two hidden LIF layers have time constant 2, threshold 1 and dropout 0.2. The memory window is $\theta=40$. Training uses seed 801, batch size 512, AdamW, weight decay 0.05 and gradient clipping at 1. Learning rates start at 0.003 for 40 clean epochs and 0.001 for 12 mixed epochs, with cosine decay to $5\times10^{-5}$ in each stage. Both stages use Eq. (6). Training shifts input channels by up to 12 in either direction. Evaluation noise seed 901 supplies paired inputs to all models, with shorter gaps taken as prefixes of the longest noise realization.

Gesture uses subjects 1--19 for training, subjects 20--23 for validation and official test subjects 24--29. The event mirror has millisecond timestamp precision. Counts are grouped into 100 normalized frames and averaged over $4\times4$ spatial blocks, producing two-polarity $32\times32$ inputs. Each $3\times3$ convolution is followed by LIF activation and $2\times2$ average pooling. The memory branch is attached to the 256-unit fully connected layer, with $\theta=300$; output scores are membrane potentials. This protocol uses 100 frames rather than the original DMP study's 1,000-frame Gesture setting.

Each model starts mixed training from its best validation checkpoint within 50 clean epochs. Fine-tuning uses seed 701, Adam, learning rate $10^{-4}$, batch size 4 and 20 fixed epochs with Eq. (6). Input-byte hashes confirm identical augmented batches across models. The 1$\times$ noise rate is estimated from training activity; 3$\times$ appears only in stress evaluation.

### Sequential MNIST
The official training images are stratified into 54,000 training and 6,000 validation examples with split seed 1701. Pixels are normalized to $[0,1]$. PS-MNIST uses one permutation generated with seed 1702, shared across all splits and training seeds. The two 200-unit LIF layers use time constant 2, threshold 1, no dropout and $\theta=300$.

Seeds 42, 43 and 44 each train all three models on both tasks. Adam uses batch size 512, no weight decay and gradient clipping at 1. The 50-epoch clean stage uses whole-sequence cross-entropy, with constant learning rate 0.001 on S-MNIST and 0.005 on PS-MNIST. The 12-epoch mixed stage uses Eq. (6) and learning rate 0.0003. Paired starts use the default linear-layer initialization, rather than the original DMP study's truncated-normal initialization. Evaluation noise seed is 2701. The 18 fits took 7,932.53 seconds on an A100 80 GB, followed by 319.1 seconds of evaluation.

**Evaluation chronology.**
SHD was used iteratively during architecture development, and SSC's official test set had also been evaluated in earlier project stages. For the comparisons reported here, protocols and scheduled checkpoints were fixed before evaluation. All 18 MNIST models were frozen before official-test evaluation for that block.

## SHD component and supervision controls

SHD development uses 6,245 training utterances, 640 validation utterances and a 1,271-example confirmation subset drawn from the official training release; training excludes speakers 0 and 2. Official-test evaluation contains 2,264 utterances. Inputs have 140 channels and 100 normalized frames, with two 128-unit LIF hidden layers, dropout 0.2 and $\theta=40$. A separate DMP baseline trained on the full training release reached $91.95\pm0.64$% clean accuracy. That score belongs to DMP and uses a different training split from the memory studies.

Content training runs for 80 epochs with an auxiliary stable-memory classification loss of weight 1.0. The scalar, wide-scalar and vector controls in the archived results (`tab:capacity`) share a historical protected recipe that also includes noise-detector calibration and explicit opening/leakage objectives. The wide-scalar control uses $M\to d\to1$ projections with ReLU after each projection. It increases parameter count while retaining scalar drive.

The separate supervision study compares task supervision alone, task supervision with exact closure on empty raw frames, and the same model with additional noise-position-derived gate losses (the archived results (`tab:supervision`)). The last arm jointly adds gate classification and leakage penalties. All arms retain the auxiliary class loss, start from matched class-trained sources and receive freshly initialized permission networks. Thus the task-supervised arms do not inherit a trained noise detector.

Each arm uses three fine-tuning seeds (501--503), 30 epochs, batch size 128 and AdamW with learning rate 0.001, weight decay 0.05, gradient clipping at 1 and cosine decay to $5\times10^{-5}$. Training mixes 25% single words, 37.5% middle interruptions and 37.5% paired words. Gaps are silent or rate-0.25 Poisson noise, with lengths 20, 40 or 80. Channel shifts range up to 12. Paired words share a speaker and have different class labels; SHD's 20 labels encode both digit and language.

At $d=30$, the zero-input rule raises silence retention from 29.52% to 67.82%. Adding gate labels yields little further improvement in noise retention and lowers the average new-word score. This comparison motivates the class-supervised recipe used in the main transfer study. Exact closure remains conditional on the learned decision: the mean direct old-state coefficient during noise is 0.706 for the $d=30$ zero-rule model, rather than a guaranteed retention threshold for every noisy sequence.

## Additional results and analyses

### Recognition and middle interruptions
the archived results (`tab:clean`) reports whole-sequence accuracy before and after mixed training. the archived results (`tab:middle`) lists all standard-intensity middle interruptions. Because the event resumes after a middle gap, these tests permit recovery from later evidence; they answer a different question from tail retention. At the unseen L160 gap on MNIST, GDSM's middle-silence scores are slightly below DMP, whereas its middle-noise scores are higher. Tail protection therefore does not imply a uniform advantage on every interruption placement.

### Unfamiliar noise and event switching
Uniform-noise training does not cover every disturbance. Concentrating SHD noise in 28 adjacent channels at rate 1.25 preserves the global mean intensity of uniform rate-0.25 noise, but lowers the $d=30$ zero-rule model's L80 retention from 60.16% to $6.91\pm0.38$%. Gesture's unseen 3$\times$ noise also removes the tail advantage at L80 (the archived results (`tab:strong`)). For the corresponding Gesture middle test, GDSM's whole-sequence accuracy exceeds DMP, while its final-20-frame accuracy is 18.31 points lower. These outcomes identify noise distribution and recovery speed as useful targets for further development.

A frozen-memory SHD study tests whether reading can accelerate switching without erasing stable content. It fixes the entire backbone and first-layer reading, training only three final-layer read multipliers for 30 epochs with additional new-word class queries at frames 20 and 40. A static reader learns three logits; a dynamic reader uses a $12\to16\to3$ MLP on causal input/context and memory statistics. Both return $3\,\mathrm{softmax}(\cdot)$, initialized to unit coefficients. Candidate, stable and write-gate traces remain identical across readers for the same encoded stream.

With the original fixed reader, first-20-frame new-word accuracy is $12.22\pm0.97$%, while the old label is predicted on $27.72\pm4.56$% of examples. New-word final-20-frame accuracy rises to $71.72\pm4.46$%. Static prefix reading yields $12.28\pm0.87$% early accuracy and $58.94\pm8.17$% noise L80 retention; dynamic prefix reading gives $12.69\pm0.45$% and $61.50\pm8.36$%. The fixed reader's retention is $60.16\pm8.76$%. These small early-read improvements support using the simpler fixed reader for the main comparisons while leaving event-switching adaptation as a distinct problem.

### Paired uncertainty
SSC's GDSM--DMP tail gains at L120 are 51.41 points for silence and 48.88 for noise. Speaker-cluster bootstrap 95% intervals are $[49.49,53.25]$ and $[46.96,50.73]$. Gesture's corresponding subject-cluster intervals are reported in the main text. These intervals describe test-population sampling conditional on the fitted models; they do not estimate variation across training seeds.

For MNIST, the archived results (`tab:paired`) gives the three paired seed differences and Student-$t$ intervals with two degrees of freedom. Every observed longest-tail difference is positive, but three seeds leave broad intervals for active-noise conditions. The S-MNIST silence interval excludes zero. Clean-stage GDSM--DMP differences average 0.71 points on S-MNIST (95% interval $[-0.16,1.59]$) and 1.35 on PS-MNIST ($[-1.65,4.34]$).

## Hardware implementation details

### Local cycles and network operations
The local cycle comparison uses four input channels, $N=32$ read outputs, permission hidden width $H_q=32$, and matched Q11.12 precision and row scheduling. DMP takes $1+d+N$ processing cycles per frame. Dense GDSM takes $4d+H_q+2N$; a fully closed stable gate permits cached reading and reduces this to $4d+H_q+N$. Two handshake cycles per transaction are excluded.

The resulting counts at $d=10/30$ are 43/63 for DMP, 136/216 for dense GDSM and 104/184 for closed cached GDSM (the memory-core cycle plot). The local memory branch is more expensive with GDSM. Whole-network MAC accounting includes the larger fast backbone and gives the smaller model-level overhead shown in the full-network MAC plot; these two quantities measure different scopes and cannot be treated as interchangeable latency ratios.

See [memory-core cycle plot](figures/hardware.png).

Parameter counts provide another view of cost. SSC uses 39,075 parameters for FSNN, 47,281 for DMP and 66,495 for GDSM. The corresponding MNIST counts are 42,610, 59,213 and 86,778. Gesture uses 1,085,356 for DMP and 1,259,395 for GDSM. Equal memory dimension therefore does not mean equal total parameter count; the SHD wide-scalar control addresses parameter count for the input-drive comparison specifically.

### Fixed-point replay
The memory core uses signed 24-bit Q11.12 values, wide intermediate products, arithmetic shifts and saturation. Sigmoid is a 257-entry lookup table over $[-8,8]$ with spacing $1/16$. The implemented threshold is $82/4096$ and the EMA coefficient is $3277/4096$. Any open stable coordinate triggers the complete stable read; reuse occurs when all coordinates close.

The trained trace uses the seed-42 S-MNIST mixed checkpoint, the first official-test image (digit 7), the first $d=40$ memory layer and four read coordinates. Its 944 steps comprise 784 pixel steps, 80 silent steps and 80 clipped rate-0.1 noise steps. Native replay matches all 117,056 checked integer-reference values. It takes 185,528 processing cycles, compared with 188,800 for dense execution: 818 skipped stable reads save 3,272 cycles, or 1.73%.

Against floating-point execution, mean absolute errors are 0.00548 for the candidate, 0.00508 for stable memory, 0.03341 for read current and 0.000087 for gates. One of 37,760 coordinate decisions differs at the hard gate boundary, producing a maximum gate error of 0.4072. Exact agreement with the fixed-point reference thus establishes correct integer execution, while the floating-point comparison measures quantization effects. No physical power or energy measurement is inferred from these counts.

## Complete supplementary tables

### SHD supervision

| $d$ | Training | Clean | Silence L80 | Noise L80 | New word |
| --- | --- | --- | --- | --- | --- |
| 10 | Task | $85.62\pm2.06$ | $31.38\pm3.19$ | $41.64\pm9.83$ | $73.28\pm2.51$ |
| 10 | Task + zero rule | $86.31\pm1.11$ | $58.42\pm4.47$ | $46.29\pm8.23$ | $72.97\pm1.60$ |
| 10 | Task + zero rule + gate labels | $86.10\pm1.85$ | $60.38\pm4.03$ | $53.47\pm4.17$ | $69.05\pm1.82$ |
| 30 | Task | $87.38\pm0.98$ | $29.52\pm6.19$ | $56.96\pm12.77$ | $75.89\pm1.27$ |
| 30 | Task + zero rule | $87.99\pm0.87$ | $67.82\pm5.48$ | $60.16\pm8.76$ | $75.63\pm1.59$ |
| 30 | Task + zero rule + gate labels | $87.99\pm0.19$ | $66.84\pm7.17$ | $60.81\pm13.50$ | $73.29\pm0.27$ |

### Clean classification

| Task | Stage | FSNN | DMP | GDSM |
| --- | --- | --- | --- | --- |
| SSC | Clean | 33.97 | 48.71 | 58.36 |
| SSC | Mixed | 33.98 | 47.89 | 58.83 |
| DVS Gesture | Mixed | 85.61 | 86.74 | 86.74 |
| S-MNIST | Clean | 45.71 $\pm$ 2.10 | 97.27 $\pm$ 0.93 | 97.99 $\pm$ 0.60 |
| S-MNIST | Mixed | 14.84 $\pm$ 1.37 | 94.39 $\pm$ 1.56 | 96.42 $\pm$ 0.75 |
| PS-MNIST | Clean | 38.12 $\pm$ 1.27 | 92.16 $\pm$ 0.87 | 93.51 $\pm$ 1.67 |
| PS-MNIST | Mixed | 27.46 $\pm$ 1.59 | 89.00 $\pm$ 1.82 | 89.63 $\pm$ 2.79 |

### All middle interruptions

| Task | Interruption | $L$ | FSNN | DMP | GDSM |
| --- | --- | --- | --- | --- | --- |
| SSC | Silent | 40 | 33.77 | 38.62 | 53.97 |
| SSC | Silent | 80 | 33.36 | 37.03 | 53.04 |
| SSC | Silent | 120 | 33.09 | 36.66 | 51.79 |
| SSC | Poisson | 40 | 33.57 | 43.47 | 53.66 |
| SSC | Poisson | 80 | 33.26 | 42.67 | 52.39 |
| SSC | Poisson | 120 | 32.79 | 42.05 | 50.94 |
| DVS Gesture | Silent | 20 | 85.98 | 85.61 | 86.36 |
| DVS Gesture | Silent | 40 | 85.61 | 85.23 | 88.26 |
| DVS Gesture | Silent | 80 | 85.23 | 83.71 | 86.36 |
| DVS Gesture | Poisson | 20 | 85.48 | 85.98 | 85.73 |
| DVS Gesture | Poisson | 40 | 85.61 | 86.74 | 85.98 |
| DVS Gesture | Poisson | 80 | 85.73 | 85.35 | 86.36 |
| S-MNIST | Silent | 40 | 15.18 $\pm$ 1.62 | 88.45 $\pm$ 1.80 | 93.66 $\pm$ 0.18 |
| S-MNIST | Silent | 80 | 15.31 $\pm$ 1.60 | 89.85 $\pm$ 3.01 | 93.68 $\pm$ 2.59 |
| S-MNIST | Silent | 160 | 15.72 $\pm$ 1.68 | 81.15 $\pm$ 9.20 | 78.97 $\pm$ 11.12 |
| S-MNIST | Poisson | 40 | 13.90 $\pm$ 1.67 | 89.12 $\pm$ 1.55 | 93.67 $\pm$ 0.12 |
| S-MNIST | Poisson | 80 | 13.08 $\pm$ 1.62 | 91.56 $\pm$ 1.44 | 95.03 $\pm$ 0.65 |
| S-MNIST | Poisson | 160 | 11.51 $\pm$ 1.73 | 88.46 $\pm$ 2.19 | 92.39 $\pm$ 0.43 |
| PS-MNIST | Silent | 40 | 28.15 $\pm$ 1.08 | 85.44 $\pm$ 1.93 | 85.93 $\pm$ 2.84 |
| PS-MNIST | Silent | 80 | 27.58 $\pm$ 1.03 | 85.66 $\pm$ 2.27 | 84.84 $\pm$ 5.06 |
| PS-MNIST | Silent | 160 | 25.09 $\pm$ 1.08 | 81.65 $\pm$ 4.57 | 79.23 $\pm$ 11.18 |
| PS-MNIST | Poisson | 40 | 27.58 $\pm$ 1.34 | 85.24 $\pm$ 1.57 | 85.71 $\pm$ 2.07 |
| PS-MNIST | Poisson | 80 | 27.65 $\pm$ 1.44 | 84.80 $\pm$ 1.72 | 84.92 $\pm$ 2.74 |
| PS-MNIST | Poisson | 160 | 27.44 $\pm$ 1.34 | 81.87 $\pm$ 4.39 | 82.55 $\pm$ 4.56 |

### Gesture strong noise

| Test | $L$ | FSNN | DMP | GDSM |
| --- | --- | --- | --- | --- |
| Middle | 20 | 84.47 | 81.82 | 83.59 |
| Middle | 40 | 82.20 | 63.13 | 74.87 |
| Middle | 80 | 64.27 | 19.19 | 54.17 |
| Tail | 20 | 9.09 | 9.09 | 18.69 |
| Tail | 40 | 9.09 | 9.09 | 12.50 |
| Tail | 80 | 9.09 | 9.09 | 9.09 |

### MNIST paired uncertainty

| Task / gap | Seed differences | Paired 95% interval |
| --- | --- | --- |
| S / silence | 75.34, 50.71, 83.91 | $[27.17,112.80]$ |
| S / noise | 17.02, 7.10, 41.49 | $[-22.10,65.84]$ |
| PS / silence | 51.31, 22.36, 53.26 | $[-0.68,85.30]$ |
| PS / noise | 24.56, 2.57, 24.25 | $[-14.19,48.45]$ |


## Validation learning curves

The clean-input validation functions return whole-sequence (`mean`) and final-20-frame (`last20`) accuracy; Figure 3 plots only `mean`. SSC evaluates every 10 epochs and at each stage end (40 clean + 12 mixed); MNIST evaluates every 5 epochs and at each stage end (50 clean + 12 mixed). The epoch axis continues across stages. Validation uses all 9,981 SSC or 6,000 MNIST validation examples. The three MNIST seeds are averaged at matching epochs, with sample standard deviation; SSC has a single seed. No smoothing or invented intermediate evaluations are used. Detailed test-stage results remain distinct from these validation curves.
