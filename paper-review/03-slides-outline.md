# Slide outline (draft 2, 2026-11-16)

Follows `03-outline.md`. About 14 slides for a 15-minute talk. Language: English (same as the report). IEMOCAP only.
Numbers are final (5-fold means). Build the deck with the `do-pptx` skill once this outline and the report are
approved. Results as tables, not charts.

| # | Title | Content | Visual |
|---|---|---|---|
| 1 | Title | Title, name, course, date | none |
| 2 | Emotion-aware ASR: what and why | Output = words + emotion; one example utterance with transcript and label | one example box |
| 3 | Problem | Joint single-model training trades one task against the other (Gao et al. 2023); cascades lose cues to ASR errors | 2 short bullets |
| 4 | Our approach: decouple, choose step by step | One sentence per module: ASR front-end, text + context, audio encoder, gated fusion | Fig. 1 (`figures/fig_overview`) |
| 4b | How one utterance flows | Sizes at each step, gate weights, class probabilities | Fig. 2 (`figures/fig_flow`) |
| 5 | Data and protocol | IEMOCAP 4 classes, 5531 utterances, 5-fold leave-one-session-out; WER, UA | class-count table |
| 6 | Step 1: ASR zero-shot | SenseVoice 11.20 vs Whisper turbo 13.19 WER | Table III, zero-shot column |
| 7 | Step 2: ASR fine-tuned | Whisper turbo 8.32 vs SenseVoice 9.73: Whisper becomes the front-end | Table III, both columns |
| 8 | Step 3: audio encoder | Whisper large-v3 81.17 vs WavLM 80.24 UA | Table IV |
| 9 | Step 4: ablation | Full 81.17; w/o text −8.76, w/o audio −3.63, w/o context −2.32 | Table V |
| 10 | How much do ASR errors cost? | Ground truth 78.57 vs transcripts 77.54 (text only): 1.03 UA at 8.3% WER | 2-row table |
| 11 | What each module helps | Gate on audio 0.47; audio helps angry (+6.7), context helps happy (+8.8) | Table VI |
| 12 | Comparison with published work | Table I with protocol caveat | Table I |
| 13 | Limitations and next steps | Acted speech, one run, possible Whisper pre-training overlap; spontaneous speech (MSP-Podcast) next | bullets |
| 14 | Summary | 3 takeaways: Whisper FT 8.32 WER; 81.17 UA; text > audio > context, ASR errors cost ~1 UA | 3 boxes |

Backup slides (after Q&A): UA per fold (Table VII), training hyper-parameters, SenseVoice-transcript full system
(81.44, within fold variance).
