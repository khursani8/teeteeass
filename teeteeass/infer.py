from teeteeass.onnx_modules.V220_OnnxInference import OnnxInferenceSession
import time
import numpy as np
from teeteeass.common import (
    intersperse,
    clean_text,
    cleaned_text_to_sequence,
    extract_bert_feature
)
from pathlib import Path
from huggingface_hub import snapshot_download,HfApi
class hps:
    version = "2.2.0"
    data = type('Data', (), {'add_blank': True})

repo_id = "khursani8/onnx"
class TTS:
    def __init__(self,speaker):
        self.speaker = speaker
        if not Path(f"onnx/{self.speaker}").is_dir():
            snapshot_download(repo_id=repo_id, allow_patterns=f"{speaker}/*",local_dir="onnx",local_dir_use_symlinks="auto")
        self.session = OnnxInferenceSession(
            {
                "enc": f"onnx/{self.speaker}/{self.speaker}_enc_p.onnx",
                "emb_g": f"onnx/{self.speaker}/{self.speaker}_emb.onnx",
                "dp": f"onnx/{self.speaker}/{self.speaker}_dp.onnx",
                "sdp": f"onnx/{self.speaker}/{self.speaker}_sdp.onnx",
                "flow": f"onnx/{self.speaker}/{self.speaker}_flow.onnx",
                "dec": f"onnx/{self.speaker}/{self.speaker}_dec.onnx",
            },
            Providers=['CPUExecutionProvider'],
        )
    def clean_text(self,text,language_str,use_jp_extra=False,raise_yomi_error=False):
        norm_text, phone, tone, word2ph = clean_text(
            text,
            language_str,
            use_jp_extra=use_jp_extra,
            raise_yomi_error=raise_yomi_error,
        )
        return norm_text, phone, tone, word2ph
    def clean_text_to_sequence(self,phone,tone,language):
        phone, tone, lang_ids = cleaned_text_to_sequence(
            phone,
            tone,
            language
        )
        return phone, tone, lang_ids
    def get_bert_feature(self,norm_text,word2ph,language,device):
        bert_ori = extract_bert_feature(
            norm_text,
            word2ph,
            language,
            device,
            # assist_text,
            # assist_text_weight,
        )
        return bert_ori
    def get_text(self,text,language,hps=hps,device='cpu',given_tone=None):
        norm_text, phone, tone, word2ph  = self.clean_text(
            text,
            language,

        )
        if given_tone is not None:
            if len(given_tone) != len(phone):
                raise Exception(
                    f"Length of given_tone ({len(given_tone)}) != length of phone ({len(phone)})"
                )
            tone = given_tone
        phone, tone, lang_ids = self.clean_text_to_sequence(
            phone,
            tone,
            language
        )
        if hps.data.add_blank:
            phone = intersperse(phone, 0)
            tone = intersperse(tone, 0)
            lang_ids = intersperse(lang_ids, 0)
            for i in range(len(word2ph)):
                word2ph[i] = word2ph[i] * 2
            word2ph[0] += 1
        
        bert_ori = self.get_bert_feature(norm_text,word2ph,language,'cpu')
        del word2ph
        assert bert_ori.shape[-1] == len(phone), phone

        ja_bert = np.zeros((1024, len(phone)))
        en_bert = np.zeros((1024, len(phone)))
        bert    = np.zeros((1024, len(phone)))

        if language == "ZH":
            bert = bert_ori
        elif language == "JP":
            ja_bert = bert_ori
        elif language == "EN":
            en_bert = bert_ori
        else:
            raise ValueError("language should be ZH, JP or EN")

        assert bert.shape[-1] == len(
            phone
        ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

        phone = np.longlong(phone)
        tone = np.longlong(tone)
        lang_ids = np.longlong(lang_ids)
        
        return bert.T,ja_bert.T,en_bert.T,phone,tone,lang_ids
    def generate(self,text,language,emo=None):
        start = time.monotonic()
        bert, ja_bert, en_bert, phones, tones, lang_ids = self.get_text(text,language,hps,'cpu')

        audio = self.session(phones, tones, lang_ids, bert, ja_bert, en_bert, emo).squeeze()
        print(time.monotonic() - start)
        return audio
    def set_providers(self,providers):
        self.session.set_providers(providers)
    def get_providers(self):
        import onnxruntime as ort
        return ort.get_available_providers()
    def list_speakers():
        api = HfApi()
        repo_files = api.list_repo_files(repo_id=repo_id)
        folders = [file.split('/')[0] for file in repo_files if '/' in file]
        return sorted(set(folders))