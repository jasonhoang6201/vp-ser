// Audio-LLM toy for the clips: eight typical free-text answers and the parsing rules that turn them into
// MELD labels, plus the input length in tokens. Same answers and rules as demo/23-audio-llm.html.
(function (root) {
  const OUTS = [
    ['joy', 'joy'], ['Happy', 'joy'], ['The speaker sounds angry.', 'anger'], ['The emotion is sadness.', 'sadness'],
    ['Neutral', 'neutral'], ['The transcription is: "I am so happy for you." The speaker sounds sad.', 'sadness'],
    ['The speaker seems frustrated.', 'anger'], ["I'm sorry, I can't determine the emotion from this audio.", 'neutral'],
  ];
  const LBL = ['neutral', 'joy', 'surprise', 'anger', 'sadness', 'disgust', 'fear'];
  const SYN = {happy: 'joy', happiness: 'joy', angry: 'anger', sad: 'sadness', surprised: 'surprise', scared: 'fear', afraid: 'fear', disgusted: 'disgust'};

  // p1: case-insensitive keyword search, p2: synonym table, p3: drop the "The transcription is: ..." preamble.
  function parse(s, p1, p2, p3) {
    if (!p1) return LBL.includes(s) ? s : null;
    let t = s.toLowerCase();
    if (p3) t = t.replace(/the transcription is:\s*"[^"]*"\s*/, '');
    for (const w of t.replace(/[^a-z ]/g, ' ').split(/\s+/)) {
      if (LBL.includes(w)) return w;
      if (p2 && SYN[w]) return SYN[w];
    }
    return null;
  }
  const score = (p1, p2, p3) => OUTS.map(([s, truth]) => { const got = parse(s, p1, p2, p3); return {s, truth, got, ok: got === truth}; });

  // Prompt tokens are an example; audio is 25 tokens per second (one per ~40 ms).
  const tokens = sec => ({sys: 12, audio: sec * 25, prompt: 28});

  root.LLM = {OUTS, LBL, parse, score, tokens};
})(typeof window !== 'undefined' ? window : globalThis);
