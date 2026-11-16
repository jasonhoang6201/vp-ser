#!/bin/zsh
# IEMOCAP, seed 42, 5 folds: full system on SenseVoice-FT transcripts with the frozen Whisper large-v3-turbo encoder,
# so the two-ASR comparison uses the same audio encoder as the Whisper-transcript system.
# Results: combined_asr_ctx_whisper_turbo*.
# Usage (from the project root): caffeinate -i zsh notebooks/tools/run_iemocap_sv_turbo_enc.sh
RUNS_REPO=jasonhoang6201/iemocap-runs-s42
source "${0:A:h}/chain_lib.sh"
setup

I='--set DATASET="iemocap"'

for k in 5 1 2 3 4; do
  f=_f$k; [[ $k == 5 ]] && f=""
  step 21_combined 21_svasr_turbo_s42$f $M/combined_asr_ctx_whisper_turbo$f/metrics_test.json "" ${=I} --set TEST_SESSION=$k \
    --set 'ASR="sensevoice"' --set 'AUDIO="whisper_turbo"' --set 'RUNS=["asr_ctx_e2v"]'
done
echo "ALL DONE $(date +%H:%M)"
