import soundfile as sf
from teeteeass.infer import TTS

text = 'Bagaimana saya bisa membantu anda hari ini?'
language = "EN"

# speakers = TTS.list_speakers()
# speaker = speakers[0]
speaker= None
print(speaker)

tts = TTS(speaker=speaker)
audio = tts.generate(text,language)
sf.write('test.wav', audio, 44100)
