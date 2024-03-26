from typing import Optional

from teeteeass.constants import Languages
from teeteeass.nlp import bert_models
from teeteeass.nlp.japanese.g2p import text_to_sep_kata

import numpy as np
def extract_bert_feature(
    text: str,
    word2ph: list[int],
    device: str,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
):
    """
    日本語のテキストから BERT の特徴量を抽出する

    Args:
        text (str): 日本語のテキスト
        word2ph (list[int]): 元のテキストの各文字に音素が何個割り当てられるかを表すリスト
        device (str): 推論に利用するデバイス
        assist_text (Optional[str], optional): 補助テキスト (デフォルト: None)
        assist_text_weight (float, optional): 補助テキストの重み (デフォルト: 0.7)

    Returns:
        torch.Tensor: BERT の特徴量
    """

    # 各単語が何文字かを作る `word2ph` を使う必要があるので、読めない文字は必ず無視する
    # でないと `word2ph` の結果とテキストの文字数結果が整合性が取れない
    text = "".join(text_to_sep_kata(text, raise_yomi_error=False)[0])
    if assist_text:
        assist_text = "".join(text_to_sep_kata(assist_text, raise_yomi_error=False)[0])

    model = bert_models.load_model(Languages.JP)

    style_res_mean = None
    tokenizer = bert_models.load_tokenizer(Languages.JP)
    inputs = tokenizer(text, return_tensors="np")
    inputs = {**inputs}
    # for i in inputs:
    #     inputs[i] = inputs[i].to(device)  # type: ignore
    res = model.run(['last_hidden_state'], inputs)[0][0]
    if assist_text:
        style_inputs = tokenizer(assist_text, return_tensors="pt")
        for i in style_inputs:
            style_inputs[i] = style_inputs[i].to(device)  # type: ignore
        style_res = model(**style_inputs, output_hidden_states=True)
        style_res = np.concatenate(style_res["hidden_states"][-3:-2], -1)[0]
        style_res_mean = style_res.mean(0)

    assert len(word2ph) == len(text) + 2, text
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        if assist_text:
            assert style_res_mean is not None
            repeat_feature = (
                res[i].repeat(word2phone[i], 1) * (1 - assist_text_weight)
                + style_res_mean.repeat(word2phone[i], 1) * assist_text_weight
            )
        else:
            repeat_feature = np.tile(res[i],(word2phone[i],1))
        phone_level_feature.append(repeat_feature)

    phone_level_feature = np.concatenate(phone_level_feature, axis=0)

    return phone_level_feature.T
