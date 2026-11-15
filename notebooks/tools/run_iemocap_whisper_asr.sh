#!/bin/zsh
# IEMOCAP, seed 42, 5 folds: the proposed system with Whisper large-v3-turbo FT as the ASR block.
#   Per fold: 15 Whisper turbo fine-tune + transcripts for every turn -> 21 Whisper-audio system and the ablations
#   that read ASR transcripts -> 21 WavLM-audio system. e2v_only and gt_ctx do not read ASR text, so the runs of
#   run_iemocap_s42.sh are reused. Results: combined_wasr_* next to the SenseVoice-transcript combined_asr_*.
# Usage (from the project root): caffeinate -i zsh notebooks/tools/run_iemocap_whisper_asr.sh
RUNS_REPO=jasonhoang6201/iemocap-runs-s42
source "${0:A:h}/chain_lib.sh"
setup

I='--set DATASET="iemocap"'
K=cache/iemocap/combined

for k in 5 1 2 3 4; do
  f=_f$k; [[ $k == 5 ]] && f=""
  S="--set TEST_SESSION=$k"
  step 15_whisper_ft 15_wasr_s42$f cache/iemocap/asr_whisper_ft$f.csv "" ${=I} ${=S} --set ZERO_SHOT=False
  step 21_combined 21_wasr_whisper_s42$f $K/wasr_whisper${f}_s42.npz "" ${=I} ${=S} --set 'ASR="whisper"' \
    --set 'AUDIO="whisper"' --set 'RUNS=["asr_ctx_e2v", "asr_ctx", "asr_e2v"]'
  step 21_combined 21_wasr_wavlm_s42$f $K/wasr_ctx_wavlm${f}_s42.npz "" ${=I} ${=S} --set 'ASR="whisper"' \
    --set 'AUDIO="wavlm"' --set 'RUNS=["asr_ctx_e2v"]'
done
echo "ALL DONE $(date +%H:%M)"
