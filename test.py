from onnx_modules.V220_OnnxInference import OnnxInferenceSession
import numpy as np
from teeteeass.common import intersperse
from teeteeass.nlp import (
    clean_text,
    cleaned_text_to_sequence,
    extract_bert_feature
)

def get_text(
    text: str,
    language_str,
    hps,
    device: str,
    assist_text = None,
    assist_text_weight = 0.7,
    given_tone = None,
):
    use_jp_extra = hps.version.endswith("JP-Extra")
    # 推論時のみ呼び出されるので、raise_yomi_error は False に設定
    norm_text, phone, tone, word2ph = clean_text(
        text,
        language_str,
        use_jp_extra=use_jp_extra,
        raise_yomi_error=False,
    )
    if given_tone is not None:
        if len(given_tone) != len(phone):
            raise Exception(
                f"Length of given_tone ({len(given_tone)}) != length of phone ({len(phone)})"
            )
        tone = given_tone
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if hps.data.add_blank:
        phone = intersperse(phone, 0)
        tone = intersperse(tone, 0)
        language = intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    bert_ori = extract_bert_feature(
        norm_text,
        word2ph,
        language_str,
        device,
        assist_text,
        assist_text_weight,
    )
    del word2ph
    assert bert_ori.shape[-1] == len(phone), phone

    if language_str == "ZH":
        bert = bert_ori
        ja_bert = np.zeros((1024, len(phone)))
        en_bert = np.zeros((1024, len(phone)))
    elif language_str == "JP":
        bert = np.zeros((1024, len(phone)))
        ja_bert = bert_ori
        en_bert = np.zeros((1024, len(phone)))
    elif language_str == "EN":
        print(phone)
        bert = np.zeros((1024, len(phone)))
        ja_bert = np.zeros((1024, len(phone)))
        en_bert = bert_ori
    else:
        raise ValueError("language_str should be ZH, JP or EN")

    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    phone = np.longlong(phone)
    tone = np.longlong(tone)
    language = np.longlong(language)
    return bert, ja_bert, en_bert, phone, tone, language


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

# 这里的输入和原版是一样的，只需要在原版预处理结果出来之后加上即可

text = 'Saya suka makan nasi lemak di pagi hari.'
language = "EN"

class hps:
    version = "2.2.0"
    data = type('Data', (), {'add_blank': True})

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
import soundfile as sf
sf.write('test.wav', audio, 44100)

