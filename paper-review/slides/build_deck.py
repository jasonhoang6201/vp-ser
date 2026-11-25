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
PB, P1, P2, P3, P4 = ('Part 1. Background', 'Part 2. Problem', 'Part 3. Method', 'Part 4. Experiments',
                      'Part 5. Analysis')

# Upper bound of the full system on ground-truth transcripts (combined_gt_ctx_whisper_turbo); '' until the run is in.
GT_FULL_UA = '81.89'


def notes(s):
    # flow.md numbers its slides in deck order, so the slide just added has the matching speech block.
    Deck.notes(s, SPEECH[deck.n])


def bar(s, x, y, w, h, spec, color=TEAL, size=15):
    rect(s, x, y, w, h, TINT[color], None, r=0.3)
    text(s, x + 0.2, y, w - 0.4, h, spec, size=size, color=DK, anchor='m')


def fig(name):
    return os.path.join(FIG, name + '.svg'), os.path.join(FIG, name + '_hi.png')


# 1. Title
# The kit's cover title box fits one line at 54pt; this title needs two, so the cover is laid out here.
s = deck._new(deck.blob_big + deck.blob_br)
text(s, 0.68, 1.2, 9.0, 1.9, 'Emotion-Aware Speech Recognition', size=48, bold=True, color=DK, line=0.95)
text(s, 0.70, 3.15, 8.6, 0.5, 'with Separate ASR and Emotion Models', size=24, color=GREY)
rect(s, 0.70, 3.9, 1.2, 0.06, TEAL, None)
text(s, 0.70, 4.2, 9.0, 0.4, 'Voice Processing, final project', size=17, color=DK)
text(s, 5.0, 5.75, 6.2, 0.9, ['Name  [[grey|Student ID]]'], size=18, color=DK, align='r', space=6)
notes(s)

# 2. Outline
s = deck.toc('Emotion-Aware Speech Recognition', ['Background', 'Problem', 'Method', 'Experiments', 'Analysis'])
for sh in s.shapes:
    if sh.has_text_frame and sh.text_frame.text == 'Mục lục':
        sh.text_frame.paragraphs[0].runs[0].text = 'Outline'
notes(s)

# 3. SER
s = deck.slide(PB, 'Speech emotion recognition')
card(s, 0.667, 2.0, 6.6, 4.75)
text(s, 0.92, 2.2, 6.1, 0.4, 'Two kinds of labels', size=18, bold=True, color=GREY)
for k, (kind, items, col) in enumerate([
        ('Categorical', ['a few basic emotions, such as angry, happy, sad', 'IEMOCAP uses 4 classes'], TEAL),
        ('Dimensional', ['valence: how negative or positive', 'arousal: how calm or excited'], TEAL2)]):
    y = 2.9 + k * 1.85
    pill(s, 0.92, y, 2.2, 0.5, kind, col, 16)
    bullets(s, 0.92, y + 0.7, 6.1, 1.0, items, size=17, space=6)
card(s, 7.5, 2.0, 5.12, 4.75)
text(s, 7.75, 2.2, 4.6, 0.4, 'Applications', size=18, bold=True, color=GREY)
for k, a in enumerate(['Call centers and customer service', 'Mental health monitoring', 'Voice assistants and robots',
                       'Education and e-learning', 'Driver monitoring']):
    y = 2.9 + k * 0.72
    rect(s, 7.75, y, 4.62, 0.56, WHITE, LIGHT, 0.5, r=0.3)
    text(s, 7.95, y, 4.3, 0.56, a, size=16, color=DK, anchor='m')
notes(s)

# 4. History
s = deck.slide(PB, 'History of SER')
eras = [('2000s', 'Hand-crafted features', 'MFCC, pitch and energy, classified by GMM, HMM or SVM',
         'GeMAPS feature set', TEAL),
        ('2014-2019', 'Deep learning', 'CNN and LSTM learn features from spectrograms or raw audio',
         'CNN, LSTM, attention', TEAL2),
        ('2020-2023', 'Self-supervised models', 'pre-trained on large amounts of unlabeled speech, then reused for SER',
         'wav2vec 2.0, WavLM', TEAL),
        ('2024-', 'Audio LLMs', 'one large model answers questions about speech, including emotion',
         'SALMONN, Qwen2-Audio', MAUVE)]
for k, (yrs, head, desc, ex, col) in enumerate(eras):
    x = 0.667 + k * 3.03
    card(s, x, 2.0, 2.88, 4.75)
    pill(s, x + 0.2, 2.25, 1.6, 0.48, yrs, col, 15)
    text(s, x + 0.2, 3.0, 2.5, 0.8, head, size=19, bold=True, color=col)
    text(s, x + 0.2, 3.85, 2.5, 1.6, desc, size=16, color=DK)
    rect(s, x + 0.2, 5.75, 2.48, 0.7, WHITE, LIGHT, 0.5, r=0.25)
    text(s, x + 0.3, 5.75, 2.28, 0.7, ex, size=13, color=DK, anchor='m', align='c')
    if k < 3:
        arrow(s, x + 2.9, 2.49, x + 3.02, 2.49, INK, 1.25)
notes(s)

# 5. Cues
s = deck.slide(PB, 'Three sources of emotion')
cues = [('Acoustic', 'How it is said', ['pitch, loudness and speaking speed', 'voice quality', 'spectrum, such as MFCC'],
         'audio-only SER', TEAL),
        ('Lexical', 'What is said', ['the words of the sentence', 'needs ASR or a manual transcript',
                                     'example: "I am so happy"'], 'text-only SER', TEAL2),
        ('Context', 'What came before', ['earlier turns of the conversation', 'the other speaker\'s reaction',
                                         'the same words can carry different emotions'], 'conversational SER', MAUVE)]
for k, (h, sub, items, used, col) in enumerate(cues):
    x = 0.667 + k * 4.05
    card(s, x, 2.0, 3.85, 4.75)
    text(s, x + 0.3, 2.2, 3.3, 0.5, h, size=24, bold=True, color=col)
    text(s, x + 0.3, 2.78, 3.3, 0.4, sub, size=17, bold=True, color=DK)
    bullets(s, x + 0.3, 3.4, 3.3, 2.4, items, size=16, space=10)
    pill(s, x + 0.3, 5.95, 3.25, 0.5, f'used in {used}', col, 14)
notes(s)

# 6. Datasets
s = deck.slide(PB, 'Datasets and metrics')
grid(s, 0.667, 1.95, [1.6, 1.15, 2.75, 2.3, 1.85, 2.3], [
    ['Dataset', 'Language', 'Speech', 'Size', 'Labels', 'Source'],
    ['**IEMOCAP**', 'English', 'acted dialogues, scripted and improvised', 'about 12 hours, 10 actors',
     'categories and dimensions', 'Busso et al., LRE 2008'],
    ['MSP-Podcast', 'English', 'natural speech from podcasts', '409 hours', 'categories and dimensions',
     'Lotfian and Busso, IEEE TAC 2019'],
    ['MELD', 'English', 'dialogues from the TV series Friends', 'multi-party dialogues', '7 emotions',
     'Poria et al., ACL 2019']],
    row_h=0.74, size=15, head_size=13, aligns=['l', 'l', 'l', 'l', 'l', 'l'])
for k, (m, d, col) in enumerate([('WER', 'share of words the ASR gets wrong', TEAL),
                                 ('UA', 'recall averaged over the emotion classes', TEAL2),
                                 ('WA', 'share of sentences classified correctly', TEAL2),
                                 ('F1', 'balance of precision and recall', TEAL)]):
    x = 0.667 + (k % 2) * 6.05
    y = 5.55 + (k // 2) * 0.68
    rect(s, x, y, 5.9, 0.56, TINT[col], None, r=0.3)
    text(s, x + 0.25, y, 5.4, 0.56, f'[[{"teal" if col == TEAL else "teal2"}|**{m}**]]   {d}', size=15, color=DK,
         anchor='m')
notes(s)

# 7. Directions
s = deck.slide(PB, 'Four approaches to emotion-aware ASR')
dirs = [('Robust ASR on emotional speech', 'make ASR more accurate on emotional speech, where WER is higher'),
        ('One model for ASR and SER', 'one network with two outputs: the transcript and the emotion'),
        ('ASR first, then SER', 'get the transcript first, then predict the emotion from the text and the audio'),
        ('Audio LLMs', 'a large language model listens to the speech and returns the transcript and the emotion')]
for k, (h, d) in enumerate(dirs):
    x = 0.667 + (k % 2) * 6.1
    y = 2.0 + (k // 2) * 2.45
    card(s, x, y, 5.85, 2.25)
    badge(s, x + 0.3, y + 0.3, k + 1, [TEAL, TEAL2, TEAL, MAUVE][k], size=0.5, fs=15)
    text(s, x + 1.0, y + 0.27, 3.4 if k == 2 else 4.6, 0.56, h, size=20, bold=True, color=DK, anchor='m')
    if k == 2:
        tag(s, x + 4.3, y + 0.32, 'this work', w=1.3, h=0.45, size=13)
    text(s, x + 0.3, y + 1.1, 5.3, 0.95, d, size=17, color=DK)
notes(s)

# 8. Task
s = deck.slide(P1, 'Task definition')
card(s, 0.667, 2.2, 3.4, 4.2)
text(s, 0.667, 2.4, 3.4, 0.4, 'Input', size=18, bold=True, color=GREY, align='c')
b = rect(s, 1.0, 3.75, 2.75, 1.0, WHITE, LIGHT, 0.75, r=0.2)
label(b, 'Audio of one sentence', 18, DK, True)
for i, (ttl, out, metric, col) in enumerate([
        ('Output 1', 'Transcript', 'ASR, measured by **WER**, lower is better', TEAL),
        ('Output 2', 'Emotion: neutral, happy, angry or sad', 'SER, measured by **UA**, higher is better', TEAL2)]):
    y = 2.2 + i * 2.2
    arrow(s, 4.1, 4.25, 5.2, y + 1.0, INK, 1.75)
    card(s, 5.3, y, 7.32, 2.0)
    pill(s, 5.55, y + 0.25, 1.6, 0.45, ttl, col, 14)
    text(s, 5.55, y + 0.85, 6.8, 0.5, out, size=22, bold=True, color=col)
    text(s, 5.55, y + 1.38, 6.8, 0.45, metric, size=17, color=DK)
notes(s)

# 9. Problem
s = deck.slide(P1, 'Existing approaches and their limits')
fams = [
    ('One model for both tasks',
     [('Two-stage Finetuning of Wav2vec 2.0 for SER with ASR and Gender Pretraining', 'Gao et al.  |  Interspeech 2023'),
      ('Emotion Neural Transducer for Fine-Grained Speech Emotion Recognition', 'Shen et al.  |  ICASSP 2024')],
     'the two tasks compete inside one network, and it ends up favoring ASR'),
    ('ASR first, then emotion from the text',
     [('Speech Emotion Recognition with ASR Transcripts', 'Li et al.  |  SLT 2024'),
      ('MF-AED-AEC: SER by Multimodal Fusion, ASR Error Detection and Correction', 'He et al.  |  ICASSP 2024')],
     'mistakes in the transcript lead to wrong emotions'),
]
for i, (h, papers, lim) in enumerate(fams):
    x = 0.667 + i * 6.1
    card(s, x, 2.0, 5.85, 4.6)
    text(s, x + 0.25, 2.15, 5.4, 0.8, h, size=20, bold=True, color=TEAL if i == 0 else TEAL2, anchor='m')
    for k, (t, m) in enumerate(papers):
        chip(s, x + 0.25, 3.15 + k * 0.85, 5.35, t, m, h=0.68)
    rect(s, x + 0.25, 5.0, 5.35, 1.3, TINT[MAUVE], None, r=0.2)
    text(s, x + 0.45, 5.0, 5.0, 1.3, f'[[mauve|**Limitation:**]] {lim}', size=17, color=DK, anchor='m')
notes(s)

# 10. Overview
s = deck.slide(P2, 'System overview')
pic_svg(s, *fig('fig_overview'), 0.667, 2.15, w=11.95)
notes(s)

# 11. Flow
# The figure is taller than wide, so it fills the body height and sits in the middle.
s = deck.slide(P2, 'System flow')
fh = 5.1
fw = fh * 796.08 / 675.12
pic_svg(s, *fig('fig_flow'), (13.333 - fw) / 2, 1.75, h=fh)
notes(s)

# 12. Data
s = deck.slide(P3, 'Data and evaluation')
grid(s, 0.667, 2.0, [1.4, 1.25, 1.15, 1.1, 1.0, 1.1], [
    ['Session', 'Neutral', 'Happy', 'Angry', 'Sad', 'Total'],
    ['1', '384', '278', '229', '194', '1085'],
    ['2', '362', '327', '137', '197', '1023'],
    ['3', '320', '286', '240', '305', '1151'],
    ['4', '258', '303', '327', '143', '1031'],
    ['5', '384', '442', '170', '245', '1241'],
    ['Total', '1708', '1636', '1103', '1084', '5531']],
    row_h=0.64, size=17, aligns=['l', 'r', 'r', 'r', 'r', 'r'], bold_rows=(6,))
items = [('IEMOCAP', '10 actors, 5 sessions; 5531 of 10039 sentences have one of the 4 labels, the rest serve as context'),
         ('5 folds', 'each fold tests on one session, so test speakers are never seen in training'),
         ('Validation set', '10% of the training sessions, picked at random, used to choose the best epoch')]
for k, (h, d) in enumerate(items):
    y = 2.0 + k * 1.55
    card(s, 7.95, y, 4.67, 1.4)
    text(s, 8.2, y + 0.12, 4.2, 0.4, h, size=17, bold=True, color=[TEAL, TEAL2, MAUVE][k])
    text(s, 8.2, y + 0.55, 4.2, 0.8, d, size=14, color=DK)
notes(s)

# 13. ASR
s = deck.slide(P3, 'Choosing the ASR model')
# Two cards share the heading line, the header rule and the bottom pill line; the left table has fewer rows, so its
# body rows are taller to end where the right table ends.
TOP, HEAD_Y, HEAD_H, BODY_END, PILL_Y = 2.0, 2.65, 0.55, 5.75, 6.0
card(s, 0.667, TOP, 8.47, 4.75)
text(s, 0.92, 2.2, 7.97, 0.4, 'WER before and after finetuning (%)', size=15, bold=True, color=GREY)
y0, rh = tall_table(s, 0.92, HEAD_Y, [3.75, 1.25, 1.0, 1.05, 0.92],
                    ['Model', 'Parameters', 'Original', 'Finetuned', 'Change'],
                    [['SenseVoice-Small', '234M', '**11.20**', '9.73', '−1.47'],
                     ['Whisper large-v3-turbo', '809M', '13.19', '**9.56**', '−3.63']], BODY_END, head_h=HEAD_H)
row_tag(s, 3.75, y0 + rh, rh)
pill(s, 0.92, PILL_Y, 7.97, 0.5, 'finetuning helps Whisper more: −3.63 vs −1.47', TEAL, 13)
card(s, 9.37, TOP, 3.25, 4.75)
text(s, 9.55, 2.2, 2.9, 0.4, 'WER by emotion (%)', size=15, bold=True, color=GREY)
grid(s, 9.55, HEAD_Y, [0.95, 1.15, 0.85], [
    ['Emotion', 'SenseVoice', 'Whisper'],
    ['Neutral', '10.5', '**8.4**'],
    ['Happy', '**12.5**', '16.0'],
    ['Angry', '6.8', '**4.7**'],
    ['Sad', '8.4', '**7.8**']],
    row_h=HEAD_H, size=16, head_size=12, aligns=['l', 'c', 'c'])
pill(s, 9.55, PILL_Y, 2.9, 0.5, 'happy is hardest for both', MAUVE, 13)
notes(s)

# 14. Encoder
s = deck.slide(P3, 'Choosing the audio encoder')
y0, rh = tall_table(s, 0.667, 2.55, [4.3, 1.2, 1.2, 1.6, 1.65, 2.0],
                    ['Frozen audio encoder (%)', 'UA', 'WA', 'Weighted F1', 'Change in UA', 'Audio gate weight'],
                    [['WavLM-large', '79.93', '78.89', '78.55', '−1.24', '0.28'],
                     ['Whisper large-v3-turbo', '**81.17**', '**79.98**', '**79.88**', '', '**0.51**']],
                    5.9, size=22, head_size=14)
row_tag(s, 3.9, y0 + rh, rh)
notes(s)

# 15. Ablation
s = deck.slide(P3, 'Ablation study')
grid(s, 1.0, 2.05, [4.6, 1.6, 1.6, 1.9, 1.6], [
    ['System (%)', 'UA', 'WA', 'Weighted F1', 'Change in UA'],
    ['Our system', '**81.17**', '**79.98**', '**79.88**', ''],
    ['Without audio', '77.91', '76.70', '76.67', '−3.26'],
    ['Without context', '75.89', '74.64', '74.34', '−5.28'],
    ['Without text', '72.60', '71.10', '70.50', '[[mauve|**−8.57**]]']],
    row_h=0.72, size=22, head_size=15, aligns=['l', 'c', 'c', 'c', 'c'])
for k, (t, col) in enumerate([('text  −8.57', MAUVE), ('context  −5.28', TEAL), ('audio  −3.26', TEAL2)]):
    pill(s, 1.0 + k * 3.85, 6.15, 3.5, 0.55, t, col, 17)
notes(s)

# 16. Per session
s = deck.slide(P3, 'UA per test session')
grid(s, 0.667, 2.1, [3.2, 1.75, 1.75, 1.75, 1.75, 1.75], [
    ['UA (%)', 'Session 1', 'Session 2', 'Session 3', 'Session 4', 'Session 5'],
    ['Our system', '**78.89**', '**84.79**', '**78.55**', '**82.33**', '81.31'],
    ['WavLM encoder', '76.85', '83.76', '78.00', '78.96', '**82.06**'],
    ['Without audio', '75.02', '82.88', '75.99', '75.64', '80.00'],
    ['Without context', '75.81', '73.47', '70.81', '79.63', '79.71'],
    ['Without text', '70.05', '75.49', '71.06', '70.76', '75.63']],
    row_h=0.75, size=20, head_size=15, aligns=['l', 'c', 'c', 'c', 'c', 'c'])
notes(s)

# 17. ASR quality
s = deck.slide(P4, 'How ASR errors affect emotion')
grid(s, 1.17, 2.3, [4.8, 1.9, 1.9, 2.4], [
    ['Transcript used by our system', 'WER (%)', 'UA (%)', 'Change in UA'],
    ['Ground truth', '0', f'**{GT_FULL_UA}**', ''],
    ['Whisper, finetuned', '9.56', '81.17', '−0.72'],
    ['SenseVoice, finetuned', '9.73', '81.29', '−0.60']],
    row_h=0.95, size=24, head_size=15, aligns=['l', 'c', 'c', 'c'])
notes(s)

# 18. Per-class
s = deck.slide(P4, 'Recall of each class')
grid(s, 0.667, 2.0, [2.6, 1.35, 1.3, 1.3, 1.25], [
    ['Recall (%)', 'Neutral', 'Happy', 'Angry', 'Sad'],
    ['Our system', '**69.3**', '**84.0**', '**87.5**', '**83.9**'],
    ['Without audio', '66.6', '81.6', '80.1', '83.4'],
    ['Without context', '65.8', '74.2', '81.3', '82.2'],
    ['Without text', '62.1', '67.1', '80.7', '80.4']],
    row_h=0.95, size=20, head_size=15, aligns=['l', 'c', 'c', 'c', 'c'])
card(s, 9.0, 2.0, 3.62, 4.75)
text(s, 9.25, 2.2, 3.2, 0.4, 'Gate weight on audio', size=15, bold=True, color=GREY)
for k, (nm, v) in enumerate([('Our system', '0.51'), ('Without context', '0.71'), ('WavLM encoder', '0.28')]):
    y = 2.85 + k * 0.7
    text(s, 9.25, y, 2.2, 0.5, nm, size=16, color=DK, anchor='m')
    text(s, 11.3, y, 1.1, 0.5, f'**{v}**', size=18, color=TEAL, anchor='m', align='r')
pill(s, 9.25, 5.2, 3.12, 0.5, 'audio helps angry: +7.4', TEAL, 14)
pill(s, 9.25, 5.9, 3.12, 0.5, 'context helps happy: +9.8', TEAL2, 14)
notes(s)

# 19. Related
s = deck.slide(P4, 'Comparison with published systems')
Y, N = 'yes', '–'
grid(s, 0.667, 1.95, [3.3, 3.3, 1.35, 0.95, 1.1, 0.95, 1.0], [
    ['System', 'Type', 'Transcript', 'Audio', 'Context', 'WER', 'UA'],
    ['Gao et al., Interspeech 2023', 'multi-task training', N, Y, N, N, '76.10'],
    ['ENT, ICASSP 2024', 'one model, ASR + SER', N, Y, N, '26.47', '73.88'],
    ['VIB-Emo, Interspeech 2025', 'one LLM, ASR + SER', N, Y, N, '11.1', '77.6'],
    ['MF-AED-AEC, ICASSP 2024', 'ASR first, then SER', Y, Y, N, '20.48', '79.3'],
    ['EmoBox Whisper, Interspeech 2024', 'audio only', N, Y, N, N, '73.54'],
    ['PL-Distill, ICASSP 2026', 'audio only, learned from an LLM', N, Y, N, N, '83.91'],
    ['Crab, arXiv 2026', 'ground-truth text + audio', N, Y, N, N, '75.85'],
    ['**Our system**', '**ASR first, then SER**', Y, Y, Y, '**9.56**', '**81.17**']],
    row_h=0.46, size=14, head_size=13, aligns=['l', 'l', 'c', 'c', 'c', 'c', 'c'])
bar(s, 0.667, 6.4, 11.95, 0.42, 'Data splits and metrics differ between papers, so the numbers are not directly comparable',
    MAUVE, 13)
notes(s)

deck.save(OUT)
