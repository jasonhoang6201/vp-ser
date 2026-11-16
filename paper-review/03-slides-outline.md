# Slide outline (draft 2, 2026-11-16)

Follows `03-outline.md`. About 13 slides for a 15-minute talk. Language: English (same as the report). IEMOCAP only.
Numbers are final (5-fold means). Build the deck with the `do-pptx` skill once this outline and the report are
approved. Results as tables, not charts.

| # | Title | Content | Visual |
|---|---|---|---|
| 1 | Title | Title, name, course, date | none |
| 2 | Emotion-aware ASR: what and why | Output = words + emotion; one example utterance with transcript and label | one example box |
| 3 | Problem | Joint training: conflicting gradients, the shared model drifts toward ASR (Gao et al. 2023); cascades lose cues to ASR errors | 2 short bullets |
| 4 | Our approach: decouple, choose step by step | One sentence per module: ASR module, text + context, audio encoder, gated fusion | Fig. 1 (`figures/fig_overview`) |
| 4b | How one utterance flows | Sizes at each step, gate weights, class probabilities | Fig. 2 (`figures/fig_flow`) |
| 5 | Data and protocol | IEMOCAP 4 classes, 5531 utterances, 5-fold leave-one-session-out; WER, UA | class-count table |
| 6 | Step 1: ASR zero-shot | SenseVoice 11.20 vs Whisper turbo 13.19 WER | Table III, zero-shot column |
| 7 | Step 2: ASR fine-tuned | Whisper turbo 8.32 vs SenseVoice 9.73: Whisper becomes the ASR module | Table III, both columns |
| 8 | Step 3: audio encoder | Whisper large-v3-turbo encoder 81.80 vs WavLM 80.24 UA; one Whisper for both roles | Table IV |
| 9 | Step 4: ablation | Full 81.80; w/o text −9.20, w/o context −4.97, w/o audio −4.26 | Table V |
| 10 | How much do ASR errors cost? | Ground truth 78.57 vs transcripts 77.54 (text only): 1.03 UA at 8.3% WER, vs audio +4.26, context +4.97 | 2 tables |
| 11 | What each module helps | Gate on audio 0.53; audio helps neutral (+8.0), context helps happy (+7.1) | Table VI |
| 12 | Comparison with published work | Table I with protocol caveat | Table I |
| 13 | Conclusion | 3 takeaways: Whisper FT 8.32 WER; 81.80 UA with one Whisper; text > context > audio, ASR errors cost ~1 UA. One line of future work: spontaneous speech (MSP-Podcast) | 3 boxes + 1 line |

Slide 10 also shows the full system on SenseVoice-FT transcripts with the same encoder (81.29 vs 81.80).
The deck adds a Background part (6 slides: SER, history, cues, datasets and metrics, four directions, challenges)
before slide 2; see `slides/flow.md` for the final 21-slide order.

Backup slides (after Q&A): UA per fold (Table VII), training hyper-parameters.
