# Commit memory-layer functional hardware comparison

This extension keeps the published DMP implementation unchanged and adds a
synthesizable fixed-point **memory-layer core**, not a complete Commit classifier
or the paper's 22FDX physical implementation.

`CommitMemoryCore.scala` implements:

- ReLU vector input projection, full-matrix candidate recurrence and diagonal drive.
- Delayed commit `m[t] = (1-g[t])*m[t-1] + g[t]*p[t-1]`.
- Learned local gate from four current input statistics and their previous EMA.
- Learned scalar-pixel permission adapter (8→32→d), matching one output slice of
  the existing MNIST Commit permission MLP. No timestep or corruption label input.
- Independent candidate/stable reads, including the candidate-read bias.
- Stable-read cache that skips the stable read stage only if **all** gates close.
- Clear/reset of memory, causal history and cached reads; inputs latched on accept.

All coefficients/activations use signed24-bit Q11.12 by default. Products and sums
are widened; arithmetic right shift rounds down, followed by saturation at each
specified projection/state/read boundary. Gates represent0..4096 and are computed
by a257-entry sigmoid LUT on[-8,8], with1/16 logit spacing and nearest lookup.
The local threshold isceil(.02×4096)=82; permission uses logit>=0 and raw pixel>0.
EMA usesround(.8×4096)=3277. No learning or backpropagation occurs in RTL.

Weights are explicit ports and must remain stable until clear/reset. They are not
22FDX SRAM macros. The implementation has row-wise computation stages; its cycle
counts are for this implementation and not for the original optimized DMP core.
The `commit=false` mode provides a matched scalar-drive memory baseline with the
same state precision and row scheduling. Candidate read is always recomputed.

The original public core uses a different integer arithmetic/dataflow contract,
including its own LIF leak and subtractive-reset rules. Passing its tests does
not establish equivalence to the trained PyTorch classifier. This extension tests
the memory contribution **before** the fast-path sum/LIF, not whole-network accuracy.

## Run

Requires Java, sbt1.9.8, a C++ compiler/make Verilator5.006 and firtool1.37.0 (tested).

```bash
sbt 'testOnly dmpsnn.SpikeIntegrationSpec dmpsnn.ScalarDriveSpec dmpsnn.MemoryUpdateSpec dmpsnn.NeuronBankSpec dmpsnn.SpikeVectorEncoderSpec dmpsnn.DmpCoreSpec dmpsnn.DmpGoldenModelSpec dmpsnn.DmpMultiLayerSpec'
COMMIT_HW_RESULTS=/path/results sbt 'testOnly dmpsnn.CommitMemorySpec'
```

The comparison checks against an independent arbitrary-precision Scala oracle:
long silence/noise, delayed writes, partial-coordinate closure, signed saturation,
random streams, resets and input latching. Three modes run on identical inputs:
scalar DMP, dense-read Commit and cached-read Commit. Cached/dense outputs must
match exactly. The diagnostic gate weights are constructed test fixtures; these
cases are not a claim of learned noise detection or classification accuracy.

A separate trained-weight replay validates one real MNIST sample and80+80 appended
silence/noise steps, all40 candidate/stable/gate dimensions and the first4 readout
coordinates from layer1:

```bash
python tools/export_commit_trace.py --checkpoint /path/final.pt --data /path/mnist.npz --output /path/trained-trace.json
COMMIT_TRAINED_TRACE=/path/trained-trace.json COMMIT_HW_RESULTS=/path/results sbt 'testOnly dmpsnn.TrainedCommitSpec'
sbt 'runMain dmpsnn.CommitMemoryVerilog' 'runMain dmpsnn.MatchedDmpMemoryVerilog' 'runMain dmpsnn.TrainedCommitMemoryVerilog'
python tools/native_replay.py --fixture /path/trained-trace.json --rtl generated-commit-trained/CommitMemoryCore.sv --output /path/native
```

The export script needs the existing repository `mnist_study`, `commit_memory`
and `vision_tasks` packages with their PyTorch environment. Integer expected
trajectories are calculated independently in Python/NumPy and Scala/BigInt;
The Scala test checks its oracle against the Python fixture. The native Verilator driver then checks the generated RTL against the same fixture, including cycle counts and reset replay. It avoids slow per-cycle port bookkeeping in the Scala test wrapper. Float-versus-fixed errors are reported separately.
Default `sbt test` cancels the optional trained replay if no fixture is specified.

## Interpretation

Cycle reduction is a real functional scheduling result. `stableReadRows` counts
executed stable-read stages, **not measured SRAM transactions or transistor
switches**. We do not claim operand isolation, physical clock gating, post-layout
area, operating frequency or energy savings. The gate/candidate/fast paths and
extra registers remain costs. This is a foundation for subsequent physical
implementation, not a replacement for the paper's 22FDX flow.
