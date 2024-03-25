import time
from onnx_modules.V220_OnnxInference import OnnxInferenceSession
import numpy as np
from teeteeass.common import get_text
import soundfile as sf

Session = OnnxInferenceSession(
    {
        "enc": "onnx/BertVits2.2PT/BertVits2.2PT_enc_p.onnx",
        "emb_g": "onnx/BertVits2.2PT/BertVits2.2PT_emb.onnx",
        "dp": "onnx/BertVits2.2PT/BertVits2.2PT_dp.onnx",
        "sdp": "onnx/BertVits2.2PT/BertVits2.2PT_sdp.onnx",
        "flow": "onnx/BertVits2.2PT/BertVits2.2PT_flow.onnx",
        "dec": "onnx/BertVits2.2PT/BertVits2.2PT_dec.onnx",
    },
    Providers=["CPUExecutionProvider"],
)

class hps:
    version = "2.2.0"
    data = type('Data', (), {'add_blank': True})

text = 'Bagaimana saya bisa membantu anda hari ini?'
language = "EN"

def tts(text,language):
    start = time.monotonic()

    bert, ja_bert, en_bert, phones, tones, lang_ids = get_text(
        text,
        language,
        hps,
        'cpu'
    )
    bert = bert.T
    ja_bert = ja_bert.T
    en_bert = en_bert.T
    phones = phones
    tones = tones
    lang_ids = lang_ids

    emo = np.random.randn(512, 1)
    sid = np.array([0])

    audio = Session(phones, tones, lang_ids, bert, ja_bert, en_bert, emo, sid).squeeze()
    print(audio.shape)
    sf.write('test.wav', audio, 44100)

    print(time.monotonic() - start)

tts(text,language)
tts(text + " lalala",language)