# Transcribe the voiced segments on Colab with Whisper so mispronounced or garbled segments show up without listening.
# Usage on the session: python3 asr_check.py 02-log 03-dct ...
import json
import subprocess
import sys

subprocess.run('pip -q install openai-whisper', shell=True, check=True)
import whisper

model = whisper.load_model('small')
for clip in sys.argv[1:]:
    spec = json.load(open(f'/content/tts/{clip}.json'))
    print(f'== {clip}', flush=True)
    for seg in spec['segments']:
        r = model.transcribe(f"/content/tts/out/{clip}/{seg['id']}.wav", language='vi', fp16=False)
        print(f"{seg['id']} | {r['text'].strip()}", flush=True)
