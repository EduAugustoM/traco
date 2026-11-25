"""Publica a pasta space/ em um Hugging Face Space estático (sdk: static).

Uso:
  python training/deploy_hf.py --repo SEU_USUARIO/traco --token hf_...
  # ou definindo a variável de ambiente HF_TOKEN
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
SPACE_DIR = ROOT / "space"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="ID do Space no formato usuario/nome-do-space")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN", ""), help="Token do Hugging Face com permissão write")
    a = p.parse_args()

    api = HfApi(token=a.token or None)
    # Atualiza o link do notebook no index.html com o repo real
    idx_path = SPACE_DIR / "index.html"
    html = idx_path.read_text(encoding="utf-8")
    if "__SPACE_ID__" in html:
        idx_path.write_text(html.replace("__SPACE_ID__", a.repo), encoding="utf-8")

    api.create_repo(repo_id=a.repo, repo_type="space", space_sdk="static", exist_ok=True)
    url = api.upload_folder(
        repo_id=a.repo,
        repo_type="space",
        folder_path=str(SPACE_DIR),
        commit_message="Deploy Traço (static HTML + JS + onnxruntime-web + OpenCV.js)",
    )
    print(f"Publicado com sucesso: https://huggingface.co/spaces/{a.repo}")
    print(f"Commit: {url}")


if __name__ == "__main__":
    main()
