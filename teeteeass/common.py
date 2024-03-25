import json
import numpy as np
from teeteeass.nlp import (
    clean_text,
    cleaned_text_to_sequence,
    extract_bert_feature
)

def get_hparams_from_file(config_path):
    # print("config_path: ", config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams

class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()

def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


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

