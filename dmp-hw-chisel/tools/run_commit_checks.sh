#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${SBT:=sbt}"
: "${COMMIT_HW_RESULTS:=$PWD/functional-results}"
export COMMIT_HW_RESULTS
mkdir -p "$COMMIT_HW_RESULTS"
"$SBT" 'testOnly dmpsnn.SpikeIntegrationSpec dmpsnn.ScalarDriveSpec dmpsnn.MemoryUpdateSpec dmpsnn.NeuronBankSpec dmpsnn.SpikeVectorEncoderSpec dmpsnn.DmpCoreSpec dmpsnn.DmpGoldenModelSpec dmpsnn.DmpMultiLayerSpec' 'testOnly dmpsnn.CommitMemorySpec'
if [[ -n "${COMMIT_TRAINED_TRACE:-}" ]]; then
  "$SBT" 'testOnly dmpsnn.TrainedCommitSpec'
fi
"$SBT" 'runMain dmpsnn.CommitMemoryVerilog' 'runMain dmpsnn.MatchedDmpMemoryVerilog' 'runMain dmpsnn.TrainedCommitMemoryVerilog'

if [[ -n "${COMMIT_TRAINED_TRACE:-}" ]]; then
  python3 tools/native_replay.py --fixture "$COMMIT_TRAINED_TRACE" --rtl generated-commit-trained/CommitMemoryCore.sv --output "$(dirname "$COMMIT_HW_RESULTS")/native"
fi
