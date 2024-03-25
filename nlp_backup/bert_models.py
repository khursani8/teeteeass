import gc
from typing import Optional, Union, cast

from transformers import AutoTokenizer
import onnxruntime as ort

from teeteeass.constants import DEFAULT_BERT_TOKENIZER_PATHS, Languages

__loaded_models = {}

__loaded_tokenizers = {}

def load_model(
    language: Languages,
    pretrained_model_name_or_path: Optional[str] = None
):
    # すでにロード済みの場合はそのまま返す
    if language in __loaded_models:
        return __loaded_models[language]
    
    if pretrained_model_name_or_path is None:
        assert DEFAULT_BERT_TOKENIZER_PATHS[
            language
        ].exists(), f"The default {language} BERT tokenizer does not exist on the file system. Please specify the path to the pre-trained model."
        pretrained_model_name_or_path = str(DEFAULT_BERT_TOKENIZER_PATHS[language])

    Providers=["CPUExecutionProvider"]
    sess = ort.InferenceSession(f"{pretrained_model_name_or_path}/model.onnx", providers=Providers)
    __loaded_models[language] = sess

    return sess

def load_tokenizer(
    language: Languages,
    pretrained_model_name_or_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    revision: str = "main",
):

    # すでにロード済みの場合はそのまま返す
    if language in __loaded_tokenizers:
        return __loaded_tokenizers[language]

    # pretrained_model_name_or_path が指定されていない場合はデフォルトのパスを利用
    if pretrained_model_name_or_path is None:
        assert DEFAULT_BERT_TOKENIZER_PATHS[
            language
        ].exists(), f"The default {language} BERT tokenizer does not exist on the file system. Please specify the path to the pre-trained model."
        pretrained_model_name_or_path = str(DEFAULT_BERT_TOKENIZER_PATHS[language])

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        cache_dir=cache_dir,
        revision=revision,
    )
    tokenizer.model_input_names = ['input_ids', 'attention_mask']
    __loaded_tokenizers[language] = tokenizer
    return tokenizer
