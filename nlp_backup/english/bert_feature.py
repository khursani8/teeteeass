from typing import Optional

from teeteeass.constants import Languages
from teeteeass.nlp import bert_models
import numpy as np


def extract_bert_feature(
    text: str,
    word2ph: list[int],
    device: str,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
):
    """
    英語のテキストから BERT の特徴量を抽出する

    Args:
        text (str): 英語のテキスト
        word2ph (list[int]): 元のテキストの各文字に音素が何個割り当てられるかを表すリスト
        device (str): 推論に利用するデバイス
        assist_text (Optional[str], optional): 補助テキスト (デフォルト: None)
        assist_text_weight (float, optional): 補助テキストの重み (デフォルト: 0.7)

    Returns:
        torch.Tensor: BERT の特徴量
    """

    device = "cpu"
    model = bert_models.load_model(Languages.EN)  # type: ignore

    style_res_mean = None
    tokenizer = bert_models.load_tokenizer(Languages.EN)
    inputs = tokenizer(text, return_tensors="np")
    inputs = {**inputs}
    # res = model(**inputs, output_hidden_states=True)
    res = model.run(['last_hidden_state'], inputs)[0][0]
    # res = np.concatenate(res[-3:-2], -1)[0]
    if assist_text:
        style_inputs = tokenizer(assist_text, return_tensors="np")
        for i in style_inputs:
            style_inputs[i] = style_inputs[i]  # type: ignore
        style_res = model(**style_inputs, output_hidden_states=True)
        style_res = np.concatenate(style_res["hidden_states"][-3:-2], -1)[0]
        style_res_mean = style_res.mean(0)

    assert len(word2ph) == res.shape[0], (text, res.shape[0], len(word2ph))
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        if assist_text:
            assert style_res_mean is not None
            repeat_feature = (
                np.tile(res[i],(word2phone[i],1)) * (1 - assist_text_weight)
                + style_res_mean.repeat(word2phone[i], 1) * assist_text_weight
            )
        else:
            repeat_feature = np.tile(res[i],(word2phone[i],1))
        phone_level_feature.append(repeat_feature)

    phone_level_feature = np.concatenate(phone_level_feature, axis=0)

    return phone_level_feature.T
