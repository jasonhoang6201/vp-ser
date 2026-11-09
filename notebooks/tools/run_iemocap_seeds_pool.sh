#!/bin/zsh
# Resumable IEMOCAP follow-up, 3 seeds per run:
#   1. Seeds 1, 2 on folds 1-4 for Whisper and WavLM (seed 0 is reused from its saved run).
#   2. Pooling check on session 5: attention over Whisper frames against masked mean on the same frames.
#   3. Whisper + prosody on all 5 folds.
# Usage (from the project root): caffeinate -i zsh notebooks/tools/run_iemocap_seeds_pool.sh
source "${0:A:h}/chain_lib.sh"
setup

I='--set DATASET="iemocap"'
# These steps rewrite existing metrics files, so the last seed's run file is the only marker that proves they finished.
K=cache/iemocap/combined

for k in 1 2 3 4; do
  S="--set TEST_SESSION=$k"
  step 21_combined 21_combined_iemocap_whisper_3s_f$k $K/asr_ctx_whisper_f${k}_s2.npz "" ${=I} ${=S} \
    --set 'AUDIO="whisper"' --set 'SEEDS=[0, 1, 2]' --set 'RUNS=["asr_ctx_e2v"]'
  step 21_combined 21_combined_iemocap_wavlm_3s_f$k $K/asr_ctx_wavlm_f${k}_s2.npz "" ${=I} ${=S} \
    --set 'AUDIO="wavlm"' --set 'SEEDS=[0, 1, 2]' --set 'RUNS=["asr_ctx_e2v"]'
done

for p in fmean attn; do
  step 21_combined 21_combined_iemocap_whisper_$p $K/asr_ctx_whisper_${p}_s2.npz "" ${=I} \
    --set 'AUDIO="whisper"' --set "POOL=\"$p\"" --set 'SEEDS=[0, 1, 2]' --set 'RUNS=["asr_ctx_e2v"]'
done

for k in 5 1 2 3 4; do
  f=_f$k; [[ $k == 5 ]] && f=""
  S="--set TEST_SESSION=$k"
  step 21_combined 21_combined_iemocap_pros_whisper$f $K/asr_ctx_pros_whisper${f}_s2.npz "" ${=I} ${=S} \
    --set 'AUDIO="whisper"' --set 'SEEDS=[0, 1, 2]' --set 'RUNS=["asr_ctx_pros_e2v"]'
done
echo "ALL DONE $(date +%H:%M)"
