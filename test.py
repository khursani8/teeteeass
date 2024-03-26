import soundfile as sf
from teeteeass.infer import TTS

text = 'Kerjasama yang baik dapat menghasilkan keberhasilan bersama.'
language = "EN"

text = 'Good cooperation can produce mutual success.'
language = "EN"

text = '良好的合作可以带来共同的成功。'
language = "ZH"

# text = '良好な協力は相互の成功を生み出す可能性があります。'
# language = "JP"

speakers = TTS.list_speakers()
print(speakers)
speaker = speakers[-2]
# speaker= None
print(speaker)

tts = TTS(speaker=speaker)
audio = tts.generate(text,language)
sf.write('test.wav', audio, 44100)
