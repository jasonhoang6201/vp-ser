"""Build the defense deck from flow.md.

  <venv>/bin/python paper-review/slides/build_deck.py OUT.pptx
"""
import os
import sys

sys.path.insert(0, os.path.expanduser('~/.claude/skills/do-pptx/scripts'))
from pptx_kit import *  # noqa: E402,F401,F403

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, '..', 'figures')
OUT = sys.argv[1]
deck = Deck()
SPEECH = speech_from_flow(os.path.join(HERE, 'flow.md'))
PB, P1, P2, P3, P4, P5 = ('Part 1. Background', 'Part 2. Problem', 'Part 3. Method', 'Part 4. Experiments',
                          'Part 5. Analysis', 'Part 6. Conclusion')


def notes(s):
    # flow.md numbers its slides in deck order, so the slide just added has the matching speech block.
    Deck.notes(s, SPEECH[deck.n])


def bar(s, x, y, w, h, spec, color=TEAL, size=15):
    rect(s, x, y, w, h, TINT[color], None, r=0.3)
    text(s, x + 0.2, y, w - 0.4, h, spec, size=size, color=DK, anchor='m')


# 1. Title
# The kit's cover title box fits one line at 54pt; this title needs two, so the cover is laid out here.
s = deck._new(deck.blob_big + deck.blob_br)
text(s, 0.68, 1.2, 9.0, 1.9, 'Emotion-Aware Speech Recognition', size=48, bold=True, color=DK, line=0.95)
text(s, 0.70, 3.15, 8.6, 0.5, 'with Separate ASR and Emotion Models', size=24, color=GREY)
rect(s, 0.70, 3.9, 1.2, 0.06, TEAL, None)
text(s, 0.70, 4.2, 9.0, 0.4, 'Voice Processing, final project', size=17, color=DK)
text(s, 0.70, 4.65, 9.0, 0.35, 'IEMOCAP, 5-fold leave-one-session-out', size=13, color=GREY)
text(s, 5.0, 5.75, 6.2, 0.9, ['Name  [[grey|Student ID]]'], size=18, color=DK, align='r', space=6)
notes(s)

# 2. Outline
s = deck.toc('Emotion-Aware Speech Recognition',
             ['Background', 'Problem', 'Method', 'Experiments', 'Analysis', 'Conclusion'])
for sh in s.shapes:
    if sh.has_text_frame and sh.text_frame.text == 'Mục lục':
        sh.text_frame.paragraphs[0].runs[0].text = 'Outline'
notes(s)

# 3. SER
s = deck.slide(PB, 'Speech emotion recognition')
card(s, 0.667, 2.0, 6.6, 4.75)
text(s, 0.92, 2.15, 6.1, 0.4, 'Two kinds of labels', size=16, bold=True, color=GREY)
for k, (tag, desc, ttl, meta, col) in enumerate([
        ('Categorical', 'a few basic emotions, e.g. anger, happiness, sadness, fear; IEMOCAP: 4 classes',
         'An Argument for Basic Emotions', 'Ekman  |  Cognition and Emotion 1992', TEAL),
        ('Dimensional', 'continuous scores: valence (negative to positive), arousal (calm to excited), dominance',
         'A Circumplex Model of Affect', 'Russell  |  J. Personality and Social Psychology 1980', TEAL2)]):
    y = 2.7 + k * 2.0
    pill(s, 0.92, y, 1.9, 0.45, tag, col, 14)
    text(s, 3.0, y - 0.05, 4.05, 0.95, desc, size=14, color=DK)
    chip(s, 0.92, y + 0.95, 6.1, ttl, meta, h=0.62)
card(s, 7.5, 2.0, 5.12, 4.75)
text(s, 7.75, 2.15, 4.6, 0.4, 'Applications', size=16, bold=True, color=GREY)
for k, a in enumerate(['Call centers and customer service', 'Mental health monitoring', 'Voice assistants and robots',
                       'Education and e-learning', 'Driver monitoring']):
    y = 2.7 + k * 0.6
    rect(s, 7.75, y, 4.62, 0.48, WHITE, LIGHT, 0.5, r=0.3)
    text(s, 7.95, y, 4.3, 0.48, a, size=14, color=DK, anchor='m')
chip(s, 7.75, 5.85, 4.62, 'Speech Emotion Recognition: Two Decades in a Nutshell', 'Schuller  |  Communications of the ACM 2018',
     h=0.68)
notes(s)

# 4. History
s = deck.slide(PB, 'History of SER')
eras = [
    ('2000s', 'Hand-crafted features', 'MFCC, pitch, energy + GMM, HMM or SVM', TEAL,
     [('Hidden Markov Model-based Speech Emotion Recognition', 'Schuller et al.  |  ICASSP 2003'),
      ('The INTERSPEECH 2009 Emotion Challenge', 'Schuller et al.  |  Interspeech 2009'),
      ('The Geneva Minimalistic Acoustic Parameter Set (GeMAPS)', 'Eyben et al.  |  IEEE TAC 2016')]),
    ('2014-2019', 'Deep learning', 'DNN, CNN, LSTM on spectrograms or raw waveform; attention pooling', TEAL2,
     [('SER Using Deep Neural Network and Extreme Learning Machine', 'Han et al.  |  Interspeech 2014'),
      ('Adieu Features? End-to-End SER Using a Deep Convolutional Recurrent Network', 'Trigeorgis et al.  |  ICASSP 2016'),
      ('Automatic SER Using Recurrent Neural Networks with Local Attention', 'Mirsamadi et al.  |  ICASSP 2017')]),
    ('2020-2023', 'Self-supervised encoders', 'pre-trained on unlabeled speech, then fine-tuned or frozen', TEAL,
     [('wav2vec 2.0: Self-Supervised Learning of Speech Representations', 'Baevski et al.  |  NeurIPS 2020'),
      ('Emotion Recognition from Speech Using wav2vec 2.0 Embeddings', 'Pepino et al.  |  Interspeech 2021'),
      ('WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing',
       'Chen et al.  |  IEEE JSTSP 2022')]),
    ('2024-', 'Audio LLMs', 'one model answers questions about speech, emotion included', MAUVE,
     [('SALMONN: Towards Generic Hearing Abilities for LLMs', 'Tang et al.  |  ICLR 2024'),
      ('Qwen2-Audio Technical Report', 'Chu et al.  |  arXiv 2024'),
      ('emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation',
       'Ma et al.  |  Findings of ACL 2024')]),
]
for k, (yrs, head, desc, col, papers) in enumerate(eras):
    x = 0.667 + k * 3.03
    card(s, x, 1.95, 2.88, 4.85)
    pill(s, x + 0.15, 2.08, 1.5, 0.42, yrs, col, 13)
    text(s, x + 0.15, 2.6, 2.6, 0.4, head, size=15, bold=True, color=col)
    text(s, x + 0.15, 3.0, 2.6, 0.75, desc, size=12, color=DK)
    for j, (t, m) in enumerate(papers):
        chip(s, x + 0.12, 3.8 + j * 0.98, 2.64, t, m, h=0.88)
    if k < 3:
        arrow(s, x + 2.9, 2.29, x + 3.02, 2.29, INK, 1.25)
notes(s)

# 5. Cues
s = deck.slide(PB, 'Emotion cues in speech')
cues = [('Acoustic', 'How it is said', ['prosody: pitch, energy, speaking rate', 'voice quality', 'spectrum: MFCC, spectrogram'],
         'audio-only SER', TEAL),
        ('Lexical', 'What is said', ['the words of the utterance', 'needs ASR or a manual transcript', 'e.g. "I am so happy"'],
         'text-only SER', TEAL2),
        ('Context', 'What came before', ['previous turns of the dialogue', 'the other speaker\'s reaction',
                                         'the same words can carry different emotions'],
         'conversational SER', MAUVE)]
for k, (h, sub, items, used, col) in enumerate(cues):
    x = 0.667 + k * 4.05
    card(s, x, 2.0, 3.85, 4.75)
    text(s, x + 0.3, 2.2, 3.3, 0.5, h, size=24, bold=True, color=col)
    text(s, x + 0.3, 2.75, 3.3, 0.4, sub, size=15, color=GREY)
    for j, it in enumerate(items):
        y = 3.35 + j * 0.75
        rect(s, x + 0.3, y, 3.25, 0.6, WHITE, LIGHT, 0.5, r=0.25)
        text(s, x + 0.45, y, 3.0, 0.6, it, size=13, color=DK, anchor='m')
    pill(s, x + 0.3, 5.95, 3.25, 0.5, f'used by: {used}', col, 13)
notes(s)

# 6. Datasets
s = deck.slide(PB, 'Datasets and metrics')
grid(s, 0.667, 1.85, [1.75, 0.65, 2.9, 2.15, 2.2, 2.3], [
    ['Dataset', 'Lang.', 'Speech', 'Size', 'Labels', 'Source'],
    ['**IEMOCAP**', 'EN', 'acted dialogues, scripted + improvised', '~12 h, 10 actors', 'categorical + dimensional',
     'Busso et al., LRE 2008'],
    ['MSP-Podcast', 'EN', 'natural, podcast recordings', '409 h (latest release)', 'categorical + dimensional',
     'Lotfian, Busso, IEEE TAC 2019'],
    ['MELD', 'EN', 'TV-series dialogues (Friends)', 'multi-party dialogues', '7 emotions', 'Poria et al., ACL 2019'],
    ['RAVDESS', 'EN', 'acted, fixed sentences', '24 actors, 7356 files', '8 emotions',
     'Livingstone, Russo, PLoS ONE 2018'],
    ['CREMA-D', 'EN', 'acted, 12 sentences', '91 actors, 7442 clips', '6 emotions', 'Cao et al., IEEE TAC 2014'],
    ['EmoDB', 'DE', 'acted, fixed sentences', '10 actors', '7 emotions', 'Burkhardt et al., Interspeech 2005'],
    ['VLSP 2025', 'VI', 'shared task: ASR + SER', '', 'transcript + emotion', 'VLSP 2025']],
    row_h=0.5, size=12, head_size=12, aligns=['l', 'l', 'l', 'l', 'l', 'l'])
for k, (m, d, col) in enumerate([('WER', 'errors / reference words', TEAL), ('UA', 'mean recall over classes', TEAL2),
                                 ('WA', 'overall accuracy', TEAL), ('F1', 'macro or weighted', TEAL2)]):
    x = 0.667 + k * 3.0
    rect(s, x, 6.2, 2.85, 0.55, TINT[col], None, r=0.3)
    text(s, x + 0.2, 6.2, 2.55, 0.55, f'[[{"teal" if col == TEAL else "teal2"}|**{m}**]]  {d}', size=13, color=DK,
         anchor='m')
notes(s)

# 7. Directions
s = deck.slide(PB, 'Emotion-aware ASR: four directions')
dirs = [('1', 'Robust ASR on emotional speech', 'emotional speech raises WER; errors depend on word type and emotion',
         [('ASR and Emotional Speech: A Word-Level Investigation', 'Li et al.  |  Interspeech 2023')]),
        ('2', 'Joint ASR + SER model', 'one network with an ASR head and an emotion head, or emotion tokens',
         [('Speech Emotion Recognition with Multi-Task Learning', 'Cai et al.  |  Interspeech 2021'),
          ('FunAudioLLM: Voice Understanding and Generation (SenseVoice)', 'An et al.  |  arXiv 2024')]),
        ('3', 'ASR then SER', 'emotion from the transcript, often fused with the audio',
         [('Speech Emotion Recognition with ASR Transcripts', 'Li et al.  |  SLT 2024'),
          ('MF-AED-AEC: SER by Fusion, ASR Error Detection and Correction', 'He et al.  |  ICASSP 2024')]),
        ('4', 'Audio LLMs', 'an LLM reads speech or ASR output and returns emotion, with reasons',
         [('Revise, Reason, and Recognize: LLM-Based Emotion Recognition', 'Li et al.  |  arXiv 2024'),
          ('LLM-Empowered Fine-Grained Speech Descriptors (VIB-Emo)', 'Chen et al.  |  arXiv 2025')])]
for k, (n, h, d, papers) in enumerate(dirs):
    x = 0.667 + (k % 2) * 6.1
    y = 1.95 + (k // 2) * 2.45
    card(s, x, y, 5.85, 2.3)
    badge(s, x + 0.2, y + 0.18, int(n), [TEAL, TEAL2, TEAL, MAUVE][k], size=0.45, fs=14)
    text(s, x + 0.8, y + 0.13, 3.5 if n == '3' else 4.9, 0.55, h, size=17, bold=True, color=DK, anchor='m')
    if n == '3':
        pill(s, x + 4.35, y + 0.18, 1.3, 0.42, 'this work', TEAL, 12)
    text(s, x + 0.25, y + 0.72, 5.4, 0.4, d, size=12, color=GREY)
    for j, (t, m) in enumerate(papers):
        chip(s, x + 0.25, y + 1.12 + j * 0.56, 5.35, t, m, h=0.5)
notes(s)

# 8. Challenges
s = deck.slide(PB, 'Open challenges')
chal = [('Acted vs natural speech', 'models trained on acted corpora drop on spontaneous speech',
         'future work on MSP-Podcast', TEAL),
        ('Ambiguous labels', 'annotators often disagree; labels come from a majority vote',
         '4 standard classes, excited merged into happy', TEAL2),
        ('Unseen speakers', 'a test speaker must not appear in training', 'leave-one-session-out, 5 folds', TEAL),
        ('Results hard to compare', 'papers differ in split, dev set, number of classes and metric',
         'dev set for model selection, 5-fold mean', TEAL2)]
for k, (h, d, ours, col) in enumerate(chal):
    x = 0.667 + (k % 2) * 6.1
    y = 1.95 + (k // 2) * 2.45
    card(s, x, y, 5.85, 2.3)
    text(s, x + 0.3, y + 0.2, 5.3, 0.45, h, size=19, bold=True, color=col)
    text(s, x + 0.3, y + 0.75, 5.3, 0.7, d, size=14, color=DK)
    pill(s, x + 0.3, y + 1.6, 5.25, 0.48, f'ours: {ours}', col, 13)
notes(s)

# 3. Task
s = deck.slide(P1, 'Task definition')
card(s, 0.667, 2.2, 3.4, 4.2)
text(s, 0.667, 2.4, 3.4, 0.4, 'Input', size=16, bold=True, color=GREY, align='c')
b = rect(s, 1.0, 3.75, 2.75, 1.0, WHITE, LIGHT, 0.75, r=0.2)
label(b, 'One utterance (audio)', 18, DK, True)
for i, (ttl, out, metric, col) in enumerate([
        ('Output 1', 'Transcript', 'ASR, metric **WER** (lower is better)', TEAL),
        ('Output 2', 'Emotion: neutral, happy, angry, sad', 'SER, metric **UA** (mean recall over classes)', TEAL2)]):
    y = 2.2 + i * 2.2
    arrow(s, 4.1, 4.25, 5.2, y + 1.0, INK, 1.75)
    card(s, 5.3, y, 7.32, 2.0)
    pill(s, 5.55, y + 0.25, 1.6, 0.45, ttl, col, 14)
    text(s, 5.55, y + 0.85, 6.8, 0.5, out, size=22, bold=True, color=col)
    text(s, 5.55, y + 1.38, 6.8, 0.45, metric, size=16, color=DK)
notes(s)

# 4. Problem
s = deck.slide(P1, 'Problem')
fams = [
    ('Joint model: one network, two heads',
     [('Two-stage Finetuning of Wav2vec 2.0 for SER with ASR and Gender Pretraining', 'Gao et al.  |  Interspeech 2023'),
      ('Emotion Neural Transducer for Fine-Grained Speech Emotion Recognition', 'Shen et al.  |  ICASSP 2024')],
     'conflicting gradients, the shared network drifts toward ASR'),
    ('ASR then SER: emotion from the transcript',
     [('Speech Emotion Recognition with ASR Transcripts', 'Li et al.  |  SLT 2024'),
      ('MF-AED-AEC: SER by Multimodal Fusion, ASR Error Detection and Correction', 'He et al.  |  ICASSP 2024')],
     'ASR errors reach the emotion model'),
]
for i, (h, papers, lim) in enumerate(fams):
    x = 0.667 + i * 6.1
    card(s, x, 2.0, 5.85, 4.6)
    text(s, x + 0.25, 2.15, 5.4, 0.8, h, size=19, bold=True, color=TEAL if i == 0 else TEAL2, anchor='m')
    for k, (t, m) in enumerate(papers):
        chip(s, x + 0.25, 3.15 + k * 0.85, 5.35, t, m, h=0.68)
    rect(s, x + 0.25, 5.0, 5.35, 1.3, TINT[MAUVE], None, r=0.2)
    text(s, x + 0.45, 5.0, 5.0, 1.3, f'[[mauve|**Limitation:**]] {lim}', size=16, color=DK, anchor='m')
notes(s)

# 5. Overview
s = deck.slide(P2, 'System overview')
pic(s, os.path.join(FIG, 'fig_overview.png'), 0.667, 2.15, w=11.95)
notes(s)

# 6. Flow
s = deck.slide(P2, 'One utterance through the model')
pic(s, os.path.join(FIG, 'fig_flow.png'), 0.667, 1.85, h=5.0)
card(s, 7.0, 2.0, 5.62, 4.6)
text(s, 7.25, 2.15, 5.1, 0.4, 'Trained vs frozen', size=16, bold=True, color=GREY)
rows = [('Fine-tuned', 'Whisper large-v3-turbo (ASR), RoBERTa-large', TEAL),
        ('Frozen', 'encoder of the original Whisper large-v3-turbo (features cached once)', GREY),
        ('From scratch', 'layer weights, projection, gate, classifier, auxiliary heads', TEAL2),
        ('Training only', 'auxiliary heads, dropped at inference', MAUVE)]
for k, (tag, desc, col) in enumerate(rows):
    y = 2.7 + k * 0.95
    pill(s, 7.25, y, 1.75, 0.45, tag, col if col != GREY else TEAL2, 13)
    text(s, 9.15, y - 0.1, 3.3, 0.8, desc, size=14, color=DK, anchor='m')
notes(s)

# 7. Data
s = deck.slide(P3, 'Data and protocol')
grid(s, 0.667, 2.0, [1.4, 1.25, 1.15, 1.1, 1.0, 1.1], [
    ['Session', 'Neutral', 'Happy', 'Angry', 'Sad', 'Total'],
    ['1', '384', '278', '229', '194', '1085'],
    ['2', '362', '327', '137', '197', '1023'],
    ['3', '320', '286', '240', '305', '1151'],
    ['4', '258', '303', '327', '143', '1031'],
    ['5', '384', '442', '170', '245', '1241'],
    ['Total', '1708', '1636', '1103', '1084', '5531']],
    row_h=0.6, size=17, aligns=['l', 'r', 'r', 'r', 'r', 'r'], bold_rows=(6,))
items = [('IEMOCAP', '10 actors, 5 sessions; 5531 labeled of 10039 turns (unlabeled turns used as context)'),
         ('5 folds', 'leave one session out: test speakers never seen in training'),
         ('Dev set', 'random 10% of the training sessions; ASR re-fine-tuned in every fold')]
for k, (h, d) in enumerate(items):
    y = 2.0 + k * 1.55
    card(s, 7.95, y, 4.67, 1.4)
    text(s, 8.2, y + 0.12, 4.2, 0.4, h, size=17, bold=True, color=[TEAL, TEAL2, MAUVE][k])
    text(s, 8.2, y + 0.55, 4.2, 0.8, d, size=14, color=DK)
notes(s)

# 8. ASR
s = deck.slide(P3, 'Steps 1-2: ASR (WER %)')
# Step pills sit exactly over their WER columns, so the columns of the table set their x.
cw = [3.35, 1.1, 1.1, 1.1, 0.9]
cx = [0.667 + sum(cw[:i]) for i in range(len(cw))]
pill(s, cx[2] + 0.03, 2.05, cw[2] - 0.16, 0.36, 'Step 1', TEAL2, 13)
pill(s, cx[3] + 0.03, 2.05, cw[3] - 0.16, 0.36, 'Step 2', TEAL, 13)
grid(s, 0.667, 2.15, cw, [
    ['ASR model', 'Params', 'Zero-shot', 'Fine-tuned', 'Δ'],
    ['SenseVoice-Small', '234M', '**11.20**', '9.73', '−1.47'],
    ['Whisper large-v3-turbo', '809M', '13.19', '**8.32**', '−4.87']],
    row_h=0.72, size=18, head_size=13, aligns=['l', 'c', 'c', 'c', 'c'])
card(s, 0.667, 4.85, 7.45, 1.9)
text(s, 0.92, 5.05, 7.0, 0.35, 'Chosen ASR module', size=13, bold=True, color=GREY)
text(s, 0.92, 5.45, 7.0, 0.5, 'Whisper large-v3-turbo, fine-tuned per fold', size=20, bold=True, color=TEAL)
text(s, 0.92, 6.05, 7.0, 0.55, 'SenseVoice-Large is not publicly released; Small is the open checkpoint', size=13,
     color=DK)
card(s, 8.45, 2.0, 4.17, 4.75)
text(s, 8.7, 2.15, 3.7, 0.4, 'Fine-tuned WER by emotion (%)', size=15, bold=True, color=GREY)
grid(s, 8.7, 2.6, [1.35, 1.3, 1.05], [
    ['Emotion', 'SenseVoice', 'Whisper'],
    ['Neutral', '10.5', '**8.4**'],
    ['Happy', '12.5', '**12.2**'],
    ['Angry', '6.8', '**4.7**'],
    ['Sad', '8.4', '**7.1**']],
    row_h=0.62, size=17, head_size=13, aligns=['l', 'c', 'c'])
pill(s, 8.7, 5.95, 3.7, 0.5, 'happy speech is hardest for both', MAUVE, 13)
notes(s)

# 9. Encoder
s = deck.slide(P3, 'Step 3: audio encoder (%)')
grid(s, 1.4, 2.5, [4.4, 1.6, 1.6, 1.6, 1.6], [
    ['Audio encoder (frozen)', 'UA', 'WA', 'wF1', 'Δ UA'],
    ['WavLM-large', '80.24', '78.88', '78.74', '−1.56'],
    ['Whisper large-v3-turbo', '**81.80**', '**81.06**', '**80.96**', '']],
    row_h=1.0, size=24, head_size=16, aligns=['l', 'c', 'c', 'c', 'c'])
bar(s, 1.4, 5.85, 10.4, 0.6, 'One Whisper checkpoint for both roles: fine-tuned for ASR, frozen encoder for audio', TEAL2, 17)
notes(s)

# 10. Ablation
s = deck.slide(P3, 'Step 4: ablation (%)')
grid(s, 1.4, 2.1, [4.4, 1.6, 1.6, 1.6, 1.6], [
    ['System', 'UA', 'WA', 'wF1', 'Δ UA'],
    ['Full', '**81.80**', '**81.06**', '**80.96**', ''],
    ['w/o audio', '77.54', '76.53', '76.33', '−4.26'],
    ['w/o context', '76.83', '75.24', '74.96', '−4.97'],
    ['w/o text', '72.60', '71.10', '70.50', '[[mauve|**−9.20**]]']],
    row_h=0.72, size=22, head_size=16, aligns=['l', 'c', 'c', 'c', 'c'])
for k, (t, col) in enumerate([('text  −9.20', MAUVE), ('context  −4.97', TEAL), ('audio  −4.26', TEAL2)]):
    pill(s, 1.4 + k * 3.55, 6.15, 3.25, 0.55, t, col, 17)
notes(s)

# 11. ASR quality
s = deck.slide(P4, 'ASR quality and emotion')
for i, (head, rows_, tag) in enumerate([
        ('Text only: ceiling of a better ASR',
         [['Transcript', 'WER', 'UA'], ['Ground truth', '0', '78.57'], ['Whisper-FT', '8.32', '77.54']],
         'perfect ASR adds **1.03** UA'),
        ('Full system: two ASR models',
         [['Transcript', 'WER', 'UA'], ['SenseVoice-FT', '9.73', '81.29'], ['Whisper-FT', '8.32', '**81.80**']],
         'gap **0.51** UA, within fold spread (std ≈ 2.3)')]):
    x = 0.667 + i * 6.1
    card(s, x, 2.0, 5.85, 4.6)
    text(s, x + 0.3, 2.2, 5.3, 0.45, head, size=18, bold=True, color=TEAL if i == 0 else TEAL2)
    grid(s, x + 0.3, 3.0, [2.6, 1.2, 1.4], rows_, row_h=0.8, size=20, head_size=14, aligns=['l', 'c', 'c'])
    bar(s, x + 0.3, 5.65, 5.25, 0.65, tag, TEAL if i == 0 else TEAL2, 15)
notes(s)

# 12. Per-class
s = deck.slide(P4, 'Contribution of each module')
grid(s, 0.667, 2.1, [2.6, 1.5, 1.4, 1.4, 1.3], [
    ['Recall (%)', 'Neutral', 'Happy', 'Angry', 'Sad'],
    ['Full', '**72.1**', '**85.6**', '**87.5**', '81.9'],
    ['w/o audio', '64.1', '82.7', '80.9', '**82.5**'],
    ['w/o context', '63.9', '78.5', '82.5', '82.4'],
    ['w/o text', '62.1', '67.1', '80.7', '80.4']],
    row_h=0.78, size=19, head_size=15, aligns=['l', 'c', 'c', 'c', 'c'])
card(s, 9.0, 2.0, 3.62, 4.6)
text(s, 9.25, 2.15, 3.2, 0.4, 'Audio weight of the gate', size=15, bold=True, color=GREY)
for k, (nm, v) in enumerate([('Full', '0.53'), ('w/o context', '0.55'), ('WavLM encoder', '0.31')]):
    y = 2.75 + k * 0.75
    text(s, 9.25, y, 2.2, 0.5, nm, size=16, color=DK, anchor='m')
    text(s, 11.3, y, 1.1, 0.5, f'**{v}**', size=18, color=TEAL, anchor='m', align='r')
pill(s, 9.25, 5.1, 3.12, 0.5, 'audio: neutral +8.0', TEAL, 15)
pill(s, 9.25, 5.8, 3.12, 0.5, 'context: happy +7.1', TEAL2, 15)
notes(s)

# 13. Related
s = deck.slide(P4, 'Published systems on IEMOCAP')
Y, N = 'yes', '–'
grid(s, 0.667, 1.95, [3.3, 3.2, 1.25, 1.0, 1.15, 1.0, 1.0], [
    ['System', 'Type', 'ASR text', 'Audio', 'Context', 'WER', 'UA'],
    ['Gao et al. 2023', 'two-stage multi-task', N, Y, N, N, '76.10'],
    ['ENT, ICASSP 2024', 'joint ASR + SER', N, Y, N, '26.47', '73.88'],
    ['VIB-Emo, IS 2025', 'joint ASR + SER (LLM)', N, Y, N, '11.1', '77.6'],
    ['MF-AED-AEC, ICASSP 2024', 'ASR then SER', Y, Y, N, '20.48', '79.3'],
    ['EmoBox Whisper, IS 2024', 'audio only', N, Y, N, N, '73.54'],
    ['PL-Distill, ICASSP 2026', 'audio only (distilled LLM)', N, Y, N, N, '83.91'],
    ['Crab, arXiv 2026', 'ground-truth text + audio', N, Y, N, N, '75.85'],
    ['**Ours**', '**ASR then SER**', Y, Y, Y, '**8.32**', '**81.80**']],
    row_h=0.46, size=14, head_size=13, aligns=['l', 'l', 'c', 'c', 'c', 'c', 'c'])
bar(s, 0.667, 6.4, 11.95, 0.42, 'Protocols differ (split, dev set, metric): not directly comparable', MAUVE, 13)
notes(s)

# 14. Conclusion
s = deck.slide(P5, 'Conclusion')
for k, (big, small, col) in enumerate([('8.32% WER', 'Whisper large-v3-turbo, fine-tuned', TEAL),
                                       ('81.80% UA', 'one Whisper for ASR and audio, gated fusion', TEAL2),
                                       ('text > context > audio', 'ASR errors cost about 1 UA', MAUVE)]):
    x = 0.667 + k * 4.05
    card(s, x, 2.1, 3.85, 3.5)
    text(s, x + 0.25, 2.6, 3.35, 1.4, big, size=30 if k < 2 else 26, bold=True, color=col, align='c', anchor='m')
    text(s, x + 0.25, 4.25, 3.35, 1.0, small, size=17, color=DK, align='c')
bar(s, 0.667, 5.95, 11.95, 0.75, '**Future work:** spontaneous speech, e.g. MSP-Podcast', TEAL, 17)
notes(s)

# 15. Backup
s = deck.slide('Backup', 'UA per test session (%)')
grid(s, 0.667, 2.1, [3.2, 1.75, 1.75, 1.75, 1.75, 1.75], [
    ['System', 'S1', 'S2', 'S3', 'S4', 'S5'],
    ['Full', '**79.26**', '84.28', '**78.99**', '**82.45**', '**84.03**'],
    ['WavLM encoder', '75.96', '**85.15**', '76.87', '80.32', '82.88'],
    ['w/o audio', '73.38', '82.14', '74.43', '76.77', '81.01'],
    ['w/o context', '74.43', '77.20', '75.93', '79.20', '77.37'],
    ['w/o text', '70.05', '75.49', '71.06', '70.76', '75.63']],
    row_h=0.75, size=20, head_size=15, aligns=['l', 'c', 'c', 'c', 'c', 'c'])
notes(s)

deck.save(OUT)
