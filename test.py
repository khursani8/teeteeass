import soundfile as sf
from teeteeass.infer import TTS

text = 'Bagaimana saya bisa membantu anda hari ini?'
language = "EN"
tts = TTS()
audio = tts.generate(text,language)
sf.write('test.wav', audio, 44100)
