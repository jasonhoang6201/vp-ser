# Outline: IEEE report (draft 3, 2026-11-16)

Format: IEEE conference style (`IEEEtran`, two columns), no page limit, English. IEMOCAP only.
Skeleton follows the user's earlier report (`~/Desktop/nlp-ck/overleaf_nlp_ck/main.tex`):
Background with a landscape table and an "Implications for our study" paragraph, Method as an ordered list of
components with tensor sizes, one flow figure on an example input, Results with a Δ column, Analysis paragraphs titled
by the question they answer.

Rules: architecture is shown with figures (draw.io); every result is a table. All numbers below are final
(`notebooks/PROGRESS.md`, "Đợt Whisper ASR"), mean over the 5 leave-one-session-out folds.

Source keys: `PROGRESS` = `notebooks/PROGRESS.md`; `KB` = `.claude/research/papers.md`.

---

**Title:** Emotion-Aware Speech Recognition with Separate ASR and Emotion Models

## Abstract (~150 words)

Order: problem → gap → approach → findings, stated in words (no number list; numbers start in Section V).
Same pattern as Whisper (Radford et al.) and Li et al. SLT 2024, the closest work.

Draft:
> Emotion-aware speech recognition must output both what a speaker says and how they feel. Training one model for
> both tasks lets their conflicting objectives interfere, while a pipeline that recognizes emotion from automatic transcripts
> inherits recognition errors. We study a system that keeps the two tasks in separate models: a fine-tuned speech
> recognizer produces the transcript, and an emotion classifier combines that transcript, the preceding dialogue
> turns and a frozen speech encoder through a gated fusion. Each component is chosen by a controlled comparison,
> from the recognizer to the audio encoder, and every part of the emotion model is then removed in turn. On IEMOCAP
> with leave-one-session-out evaluation, fine-tuning changes which recognizer is best, the Whisper encoder is a
> better audio branch than WavLM, and text, audio and context each improve emotion recognition, in that order of
> importance. Replacing automatic transcripts with ground truth gives only a small further gain, which suggests that
> at the error rates of a fine-tuned recognizer, the audio branch matters more than further ASR improvements.

## I. Introduction (~0.5 column)

- Emotion-aware ASR outputs the words and the speaker's emotion. Source: KB section 1.
- Two families: one model for both tasks, or ASR then SER. Gap: in joint training the two objectives differ, their
  gradients often conflict and the shared network tends to focus on ASR (Gao et al. 2023, citing Yu et al. 2020);
  a cascade loses cues to ASR errors, by an amount that depends on WER
  (Li et al. 2024).
- What we do: a decoupled system chosen step by step, then an ablation of every SER module. Contributions:
  1. a controlled comparison of two ASR models, zero-shot and fine-tuned, on the same folds;
  2. a text + audio + context SER system on ASR transcripts at 81.17 UA, with one Whisper for ASR and audio;
  3. an ablation that measures each module, and the cost of ASR errors (0.72 UA at 9.6% WER).

## II. Background

- **Joint models.** Multi-task CTC (Bansal et al.), ENT, SenseVoice emotion tokens, VIB-Emo; gradient conflict
  between ASR and SER (Gao et al. 2023). Source: KB rows + note 7.
- **SER from ASR transcripts.** Li et al. SLT 2024 (WER ~10% close to ground truth); MF-AED-AEC ICASSP 2024.
- **Speech encoders and context.** EmoBox (Whisper encoder > WavLM); GenSEC Task 3 (previous turns help).
- **Table I: earlier systems on IEMOCAP** (landscape table). Columns: system, uses ASR transcript?, uses audio?,
  context?, WER, UA, protocol. Rows: Li et al. 2024, MF-AED-AEC, VIB-Emo, EmoBox Whisper probe, PL-Distill, Crab,
  ours (9.56, 81.17). Caption: protocols differ, not directly comparable. Source: KB rows with numbers.
- **Implications for our study** (paragraph): (1) joint training suffers from gradient conflict and drifts toward ASR (Gao et al. 2023), so we decouple and can pick
  each part on its own metric; (2) Whisper-style encoders beat WavLM in EmoBox, so we test both; (3) context helps
  text SER, so we add previous turns; (4) hardware: one L4 GPU, so the audio encoder stays frozen.

## III. Method

- **Fig. 1 (`figures/fig_overview`)**: audio → Whisper large-v3-turbo (fine-tuned ASR) → transcript; transcript →
  dialogue history → RoBERTa-large; audio → Whisper turbo encoder (frozen, original checkpoint); gated fusion → classifier → emotion;
  auxiliary heads (training only).
- **Components, in the order they run:**
  - ASR module: Whisper large-v3-turbo (809M, 4 decoder layers), full fine-tune per fold; transcribes every turn.
  - Context: current turn plus the 3 previous turns (all speakers, labeled or not), as a RoBERTa sentence pair,
    left-truncated to 192 tokens; context dropped when the current turn alone is too long.
  - Text encoder: RoBERTa-large fine-tuned on ASR transcripts (not ground truth); [CLS] → 1024-d.
  - Audio encoder: encoder of the pre-trained Whisper large-v3-turbo, frozen; 33 hidden states, mean over real frames (50 frames/s), learned
    softmax layer weights → 1280-d → LayerNorm, Linear, tanh → 1024-d.
  - Gated fusion: g = softmax(W[h_text; h_audio]); z = g_text·h_text + g_audio·h_audio.
  - Classifier: Linear 1024 → 4. Loss: class-weighted CE on z plus auxiliary CE on h_text and h_audio.
- **One Whisper, two roles** (one sentence): the fine-tuned turbo is the ASR picked in step 2; the encoder of the
  original turbo checkpoint is the audio encoder picked in step 3.
- **Fig. 2 (`figures/fig_flow`)**: one utterance through the model with tensor shapes and formulas.
- **Trained vs frozen**: Whisper turbo and RoBERTa fine-tuned; layer weights, projection, gate, classifier, auxiliary
  heads trained from scratch; original Whisper turbo encoder frozen; auxiliary heads dropped at inference.
- **Why auxiliary losses** (one sentence, no dataset name): without them the gate can collapse to one branch and the
  other encoder stops learning; the auxiliary CE keeps both branches discriminative.

## IV. Data and Setup

- **Data.** IEMOCAP, 4 classes (excited merged into happy), 5531 labeled of 10039 turns. **Table II**: utterances
  per class and session.
- **Splitting.** Leave-one-session-out, 5 folds; dev = random 10% of training sessions; each ASR is re-fine-tuned per
  fold, so test transcripts never come from a model that heard that session.
- **Metrics.** WER with the Whisper English normalizer; UA (primary), WA, macro/weighted F1; epoch chosen on dev UA
  (SER) or dev WER (ASR).
- **Training.** One NVIDIA L4. RoBERTa lr 1e-5, heads 1e-4, 5 epochs, batch 16. SenseVoice 10 epochs, lr 2e-4.
  Whisper turbo full fine-tune 3 epochs, lr 1e-5, batch 16.

## V. Results (mean over 5 folds)

**Table III: step 1-2, ASR (WER %, lower is better)**

| ASR | Zero-shot | Fine-tuned | Δ |
|---|---|---|---|
| SenseVoice-Small | **11.20** | 9.73 | −1.47 |
| Whisper large-v3-turbo | 13.19 | **9.56** | −3.63 |

Text: SenseVoice wins zero-shot, Whisper wins after fine-tuning, so the ASR module is Whisper turbo FT.
Per-emotion WER (FT Whisper): neutral 8.4, happy 16.0, angry 4.7, sad 7.8 (happy/excited speech hardest). Happy and
sessions 1, 3 are raised by 4 laughter/breathing-only test turns on which FT Whisper repeats one token.

**Table IV: step 3, audio encoder** (Whisper-FT transcripts + context + audio)

| Audio encoder | UA | WA | wF1 | Δ UA |
|---|---|---|---|---|
| WavLM-large | 79.93 | 78.89 | 78.55 | −1.24 |
| Whisper large-v3-turbo | **81.17** | **79.98** | **79.88** | |

**Table V: step 4, ablation**

| System | UA | WA | wF1 | Δ UA |
|---|---|---|---|---|
| Full | **81.17** | **79.98** | **79.88** | |
| w/o audio | 77.91 | 76.70 | 76.67 | −3.26 |
| w/o context | 75.89 | 74.64 | 74.34 | −5.28 |
| w/o text | 72.60 | 71.10 | 70.50 | −8.57 |
| Ground-truth text, w/o audio (upper bound for ASR) | 78.57 | 77.24 | 77.20 | +0.66 vs w/o audio |

## VI. Analysis (paragraph titles are the questions)

- **Which module matters most?** Text (−8.57) > context (−5.28) > audio (−3.26). Every module helps on average.
- **How much does ASR quality matter for emotion recognition?** (1) Ceiling: ground truth vs Whisper-FT transcripts
  costs 0.72 UA with audio (81.89 vs 81.17) and 0.66 without (78.57 vs 77.91), so removing all ASR errors (9.6% WER
  → 0) gains less than one UA point, less than audio (+3.26) or context (+5.28). (2) Between ASR models, same turbo
  encoder: full system with SenseVoice-FT transcripts (9.7% WER) 81.29 vs 81.17, within fold variance. The ASR is
  chosen on WER, its own task. Agrees with Li et al. 2024.
- **What does the gate learn?** Mean audio weight 0.51 in the full system (0.44-0.63 per fold), 0.28 with WavLM,
  0.71 without context (weaker text, more audio). **Table VI**: recall per emotion.

  | System | Neutral | Happy | Angry | Sad |
  |---|---|---|---|---|
  | Full | **69.3** | **84.0** | **87.5** | **83.9** |
  | w/o audio | 66.6 | 81.6 | 80.1 | 83.4 |
  | w/o context | 65.8 | 74.2 | 81.3 | 82.2 |
  | w/o text | 62.1 | 67.1 | 80.7 | 80.4 |

  Audio mostly helps angry (+7.4); context mostly helps happy (+9.8) and angry (+6.2).
- **Is the gain consistent across folds?** **Table VII**: UA per fold.

  | System | S1 | S2 | S3 | S4 | S5 |
  |---|---|---|---|---|---|
  | Full | 78.89 | 84.79 | 78.55 | 82.33 | 81.31 |
  | WavLM encoder | 76.85 | 83.76 | 78.00 | 78.96 | 82.06 |
  | w/o audio | 75.02 | 82.88 | 75.99 | 75.64 | 80.00 |
  | w/o context | 75.81 | 73.47 | 70.81 | 79.63 | 79.71 |
  | w/o text | 70.05 | 75.49 | 71.06 | 70.76 | 75.63 |

  Audio, context and text each help in 5/5 folds; the Whisper encoder beats WavLM in 4/5 (not S5).

## VII. Conclusion (3-4 sentences)

Summary of the four steps and the module ranking, then one future-work sentence: the system is evaluated on acted
speech (IEMOCAP); spontaneous speech, e.g. MSP-Podcast, is left for future work.

## References (~15)

IEMOCAP (Busso 2008), SenseVoice (An 2024), Whisper (Radford 2023), WavLM (Chen 2022), RoBERTa (Liu 2019),
Li et al. SLT 2024, MF-AED-AEC (He et al. ICASSP 2024), EmoBox (Ma et al. Interspeech 2024), ENT, VIB-Emo,
PL-Distill (ICASSP 2026), Crab, GenSEC Task 3 papers (2), Gao et al. Interspeech 2023, Bansal et al.

## Figures and tables

| # | Content | Kind | Status |
|---|---|---|---|
| Fig. 1 | High-level modules | draw.io, `figures/fig_overview` | drawn (Whisper turbo ASR) |
| Fig. 2 | Example utterance through the model, tensor sizes | draw.io, `figures/fig_flow` | drawn |
| Table I | Earlier systems on IEMOCAP | table | numbers in KB |
| Table II | Class counts per session | table | from manifest |
| Table III | ASR: WER zero-shot vs fine-tuned | table | final |
| Table IV | Audio encoder | table | final |
| Table V | Ablation | table | final |
| Table VI | Recall per emotion | table | final |
| Table VII | UA per fold | table | final |

## Not included (decided)

MELD (any result), prosody text prefix, attention pooling, emotion2vec (IEMOCAP leakage), concat vs gated,
multiple seeds.
