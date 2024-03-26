from teeteeass.onnx_modules.V220_OnnxInference import OnnxInferenceSession
import time
import numpy as np
from teeteeass.common import get_text
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
    def list_speakers():
        api = HfApi()
        repo_files = api.list_repo_files(repo_id=repo_id)
        folders = [file.split('/')[0] for file in repo_files if '/' in file]
        return sorted(set(folders))