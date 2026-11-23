#!/bin/zsh
# IEMOCAP, seed 42, 5 folds: the ablations that read Whisper-FT transcripts, rerun on the transcripts of the
# Whisper-FT run whose checkpoints are published (master-122025/eaasr-iemocap). Per fold: without audio and without
# context (Whisper turbo encoder), then the full system with the WavLM encoder. The full system itself, without text
# and the ground-truth rows already exist or do not read ASR transcripts.
# Results: combined_wasr_ctx*, combined_wasr_whisper_turbo*, combined_wasr_ctx_wavlm*.
# Usage (from the project root), one chain per VM since folds are independent:
#   SESSION=eaasr FOLDS="5 1 2" caffeinate -i zsh notebooks/tools/run_iemocap_wasr_rerun.sh
RUNS_REPO=jasonhoang6201/iemocap-runs-s42
FOLDS=${FOLDS:-"5 1 2 3 4"}
source "${0:A:h}/chain_lib.sh"
setup

I='--set DATASET="iemocap"'
K=cache/iemocap/combined

for k in ${=FOLDS}; do
  f=_f$k; [[ $k == 5 ]] && f=""
  S="--set TEST_SESSION=$k"
  step 21_combined 21_wrr_turbo_s42$f $K/wasr_whisper_turbo${f}_s42.npz "" ${=I} ${=S} --set 'ASR="whisper"' \
    --set 'AUDIO="whisper_turbo"' --set 'RUNS=["asr_ctx", "asr_e2v"]'
  step 21_combined 21_wrr_wavlm_s42$f $K/wasr_ctx_wavlm${f}_s42.npz "" ${=I} ${=S} --set 'ASR="whisper"' \
    --set 'AUDIO="wavlm"' --set 'RUNS=["asr_ctx_e2v"]'
done
echo "ALL DONE $(date +%H:%M)"
