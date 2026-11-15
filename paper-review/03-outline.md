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

**Title:** Decoupled Emotion-Aware Speech Recognition: A Fine-Tuned Whisper Front-End with Context- and
Audio-Aware Emotion Recognition

## Abstract (~150 words)

Task (transcript + emotion from one utterance). We decouple ASR and SER and build the system in four steps, each
step keeping the better option: (1) zero-shot ASR, (2) fine-tuned ASR: Whisper large-v3-turbo reaches 8.32% WER
(SenseVoice-Small 9.73%); (3) audio encoder: frozen Whisper large-v3 beats WavLM-large (81.17 vs 80.24 UA);
(4) ablation. Final system: 81.17% UA on IEMOCAP 4-class, 5-fold leave-one-session-out. Removing text costs 8.76 UA,
audio 3.63, dialogue context 2.32; replacing ASR transcripts with ground truth gains only 1.03 UA.

## I. Introduction (~0.5 column)

- Emotion-aware ASR outputs the words and the speaker's emotion. Source: KB section 1.
- Two families: one model for both tasks, or ASR then SER. Gap: joint models trade one task against the other
  (Gao et al. 2023, gradient conflict); a cascade loses cues to ASR errors, by an amount that depends on WER
  (Li et al. 2024).
- What we do: a decoupled system chosen step by step, then an ablation of every SER module. Contributions:
  1. a controlled comparison of two ASR front-ends, zero-shot and fine-tuned, on the same folds;
  2. a text + audio + context SER system on ASR transcripts at 81.17 UA;
  3. an ablation that measures each module, and the cost of ASR errors (1.03 UA at 8.3% WER).

## II. Background

- **Joint models.** Multi-task CTC (Bansal et al.), ENT, SenseVoice emotion tokens, VIB-Emo; gradient conflict
  between ASR and SER (Gao et al. 2023). Source: KB rows + note 7.
- **SER from ASR transcripts.** Li et al. SLT 2024 (WER ~10% close to ground truth); MF-AED-AEC ICASSP 2024.
- **Speech encoders and context.** EmoBox (Whisper encoder > WavLM); GenSEC Task 3 (previous turns help).
- **Table I: earlier systems on IEMOCAP** (landscape table). Columns: system, uses ASR transcript?, uses audio?,
  context?, WER, UA, protocol. Rows: Li et al. 2024, MF-AED-AEC, VIB-Emo, EmoBox Whisper probe, PL-Distill, Crab,
  ours (8.32, 81.17). Caption: protocols differ, not directly comparable. Source: KB rows with numbers.
- **Implications for our study** (paragraph): (1) joint models trade tasks (literature), so we decouple and can pick
  each part on its own metric; (2) Whisper-style encoders beat WavLM in EmoBox, so we test both; (3) context helps
  text SER, so we add previous turns; (4) hardware: one L4 GPU, so the audio encoder stays frozen.

## III. Method

- **Fig. 1 (`figures/fig_overview`)**: audio → Whisper large-v3-turbo (fine-tuned ASR) → transcript; transcript →
  dialogue history → RoBERTa-large; audio → Whisper large-v3 encoder (frozen); gated fusion → classifier → emotion;
  auxiliary heads (training only).
- **Components, in the order they run:**
  - ASR front-end: Whisper large-v3-turbo (809M, 4 decoder layers), full fine-tune per fold; transcribes every turn.
  - Context: current turn plus the 3 previous turns (all speakers, labeled or not), as a RoBERTa sentence pair,
    left-truncated to 192 tokens; context dropped when the current turn alone is too long.
  - Text encoder: RoBERTa-large fine-tuned on ASR transcripts (not ground truth); [CLS] → 1024-d.
  - Audio encoder: Whisper large-v3 encoder, frozen; 33 hidden states, mean over real frames (50 frames/s), learned
    softmax layer weights → 1280-d → LayerNorm, Linear, tanh → 1024-d.
  - Gated fusion: g = softmax(W[h_text; h_audio]); z = g_text·h_text + g_audio·h_audio.
  - Classifier: Linear 1024 → 4. Loss: class-weighted CE on z plus auxiliary CE on h_text and h_audio.
- **Two Whisper models, two roles** (one sentence): turbo is the ASR picked in step 2; the encoder is large-v3
  picked in step 3 (turbo prunes the decoder, the encoder has the same architecture).
- **Fig. 2 (`figures/fig_flow`)**: one utterance through the model with tensor shapes and formulas.
- **Trained vs frozen**: Whisper turbo and RoBERTa fine-tuned; layer weights, projection, gate, classifier, auxiliary
  heads trained from scratch; Whisper large-v3 frozen; auxiliary heads dropped at inference.
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
| Whisper large-v3-turbo | 13.19 | **8.32** | −4.87 |

Text: SenseVoice wins zero-shot, Whisper wins after fine-tuning, so the front-end is Whisper turbo FT.
Per-emotion WER (FT Whisper): neutral 8.4, happy 12.2, angry 4.7, sad 7.1 (happy/excited speech hardest).

**Table IV: step 3, audio encoder** (Whisper-FT transcripts + context + audio)

| Audio encoder | UA | WA | wF1 | Δ UA |
|---|---|---|---|---|
| WavLM-large | 80.24 | 78.88 | 78.74 | −0.94 |
| Whisper large-v3 | **81.17** | **80.02** | **79.84** | |

**Table V: step 4, ablation**

| System | UA | WA | wF1 | Δ UA |
|---|---|---|---|---|
| Full | **81.17** | **80.02** | **79.84** | |
| w/o audio | 77.54 | 76.53 | 76.33 | −3.63 |
| w/o context | 78.85 | 77.42 | 77.27 | −2.32 |
| w/o text | 72.41 | 70.97 | 70.29 | −8.76 |
| Ground-truth text, w/o audio (upper bound for ASR) | 78.57 | 77.24 | 77.20 | +1.03 vs w/o audio |

## VI. Analysis (paragraph titles are the questions)

- **Which module matters most?** Text (−8.76) > audio (−3.63) > context (−2.32). Every module helps on average.
- **How much do ASR errors cost?** Same text-only setting: ground truth 78.57 vs Whisper-FT transcripts 77.54,
  1.03 UA at 8.3% WER. Better ASR has at most ~1 UA of room; adding audio (+3.63) matters more. Agrees with
  Li et al. 2024.
- **Does the lowest-WER ASR also give the best SER?** Same full system with SenseVoice-FT transcripts: 81.44 UA
  (+0.27, 3/5 folds). Within fold variance (std ~3), consistent with the 1.03 ceiling above: at WER 8-10% the choice
  of ASR barely moves SER, so the ASR is chosen on WER, its own task.
- **What does the gate learn?** Mean audio weight 0.47 in the full system (0.32-0.59 per fold), 0.31 with WavLM,
  0.60 without context (weaker text, more audio). **Table VI**: recall per emotion.

  | System | Neutral | Happy | Angry | Sad |
  |---|---|---|---|---|
  | Full | 69.3 | **85.4** | **87.6** | 82.5 |
  | w/o audio | 64.1 | 82.7 | 80.9 | 82.5 |
  | w/o context | **70.7** | 76.6 | 86.0 | 82.1 |
  | w/o text | 62.5 | 66.1 | 80.9 | 80.1 |

  Audio mostly helps angry (+6.7) and neutral; context mostly helps happy (+8.8).
- **Is the gain consistent across folds?** **Table VII**: UA per fold.

  | System | S1 | S2 | S3 | S4 | S5 |
  |---|---|---|---|---|---|
  | Full | 77.68 | 83.97 | 76.73 | 82.13 | 85.36 |
  | WavLM encoder | 75.96 | 85.15 | 76.87 | 80.32 | 82.88 |
  | w/o audio | 73.38 | 82.14 | 74.43 | 76.77 | 81.01 |
  | w/o context | 76.66 | 84.00 | 77.08 | 77.89 | 78.63 |
  | w/o text | 72.14 | 73.76 | 70.61 | 69.50 | 76.05 |

  Audio and text help in 5/5 folds; context in 3/5 (S2, S3 slightly better without it), so context is the least
  stable gain.

## VII. Limitations

Acted speech only; one run per configuration; Whisper may have seen IEMOCAP in pre-training; context uses reference
turn order and segmentation, no diarization; ASR and SER are not trained jointly.

## VIII. Conclusion (3-4 sentences)

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
