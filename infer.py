from onnx_modules.V220_OnnxInference import OnnxInferenceSession
import numpy as np
from style_bert_vits2.nlp.english.g2p import g2p
from style_bert_vits2.nlp.english.normalizer import normalize_text

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

# 这里的输入和原版是一样的，只需要在原版预处理结果出来之后加上.numpy()即可

text = 'selamat pagi semua'
norm_text = normalize_text(text)
x, tone, word2ph = g2p(norm_text)
tone = np.zeros_like(x)
language = np.zeros_like(x)
sid = np.array([0])
bert = np.random.randn(x.shape[0], 1024)
ja_bert = np.random.randn(x.shape[0], 1024)
en_bert = np.random.randn(x.shape[0], 1024)
emo = np.random.randn(512, 1)

audio = Session(x, tone, language, bert, ja_bert, en_bert, emo, sid).squeeze()
print(audio.shape)
import soundfile as sf
sf.write('test.wav', audio, 44100)

