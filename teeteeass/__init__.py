from huggingface_hub import snapshot_download
repos = [
    "khursani8/microsoft_deberta-v3-large_onnx",
    "khursani8/hfl_chinese-roberta-wwm-ext-large_onnx",
    "khursani8/ku-nlp_deberta-v2-large-japanese-char-wwm_onnx"
]
for repo_id in repos:
    snapshot_download(
        repo_id=repo_id,
        local_dir=f"onnx/{repo_id.split('/')[-1]}",
        local_dir_use_symlinks="auto"
    )