from onnx_modules.V220_OnnxInference import OnnxInferenceSession
import time
import numpy as np
from teeteeass.common import get_text

class hps:
    version = "2.2.0"
    data = type('Data', (), {'add_blank': True})

class TTS:
    def __init__(self,speaker):
        if speaker is None:
            pass
        self.session = OnnxInferenceSession(
            {
                "enc": "onnx/BertVits2.2PT/BertVits2.2PT_enc_p.onnx",
                "emb_g": "onnx/BertVits2.2PT/BertVits2.2PT_emb.onnx",
                "dp": "onnx/BertVits2.2PT/BertVits2.2PT_dp.onnx",
                "sdp": "onnx/BertVits2.2PT/BertVits2.2PT_sdp.onnx",
                "flow": "onnx/BertVits2.2PT/BertVits2.2PT_flow.onnx",
                "dec": "onnx/BertVits2.2PT/BertVits2.2PT_dec.onnx",
            },
            Providers=['CPUExecutionProvider'],
        )
    def generate(self,text,language):
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

        audio = self.session(phones, tones, lang_ids, bert, ja_bert, en_bert, emo, sid).squeeze()
        print(time.monotonic() - start)
        return audio
    def set_providers(self,providers):
        self.session.set_providers(providers)
    def get_providers(self):
        import onnxruntime as ort
        return ort.get_available_providers()