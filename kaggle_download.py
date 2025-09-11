from __future__ import annotations
import os, sys, pathlib, argparse
from typing import Optional

def kaggle_setup_and_download(
    config_dir: str = ".",
    dataset: str = "zygmunt/goodbooks-10k",
    out_dir: str = "data/raw",
) -> pathlib.Path:
    """
    Configure le client Kaggle pour utiliser un kaggle.json local, s'authentifie,
    puis télécharge + dézippe le dataset vers out_dir.
    """
    config_dir = str(pathlib.Path(config_dir).resolve())
    print("CWD:", pathlib.Path.cwd())
    print("Config dir:", config_dir)
    try:
        print("Contenu (fichiers commençant par 'kaggle'):",
              [p.name for p in pathlib.Path(config_dir).iterdir() if p.name.lower().startswith("kaggle")])
    except FileNotFoundError:
        pass

    
    cfg_path = pathlib.Path(config_dir) / "kaggle.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"kaggle.json introuvable dans {config_dir}. "
            "Place ton fichier kaggle.json ici ou passe --config-dir correctement."
        )

    
    for m in list(sys.modules):
        if m.startswith("kaggle"):
            del sys.modules[m]

    
    os.environ["KAGGLE_CONFIG_DIR"] = config_dir

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    print("OK Kaggle")

    from pathlib import Path
    raw = Path(out_dir); raw.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(dataset, path=str(raw), unzip=True)
    print("Téléchargé dans:", raw.resolve())
    return raw.resolve()

def main():
    parser = argparse.ArgumentParser(description="Télécharger un dataset Kaggle (avec kaggle.json)")
    parser.add_argument("--config-dir", default=".", help="Dossier contenant kaggle.json (par défaut: .)")
    parser.add_argument("--dataset", default="zygmunt/goodbooks-10k", help="Slug Kaggle du dataset")
    parser.add_argument("--out-dir", default="data/raw", help="Dossier de sortie pour les fichiers dézippés")
    args = parser.parse_args()
    kaggle_setup_and_download(config_dir=args.config_dir, dataset=args.dataset, out_dir=args.out_dir)

if __name__ == "__main__":
    main()
