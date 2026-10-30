"""Chargement des données Sparkov (Kaggle : kartik2112/fraud-detection)."""
import os
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("FRAUD_DATA_DIR", ROOT / "data"))
ARTIFACTS_DIR = Path(os.environ.get("FRAUD_ARTIFACTS", ROOT / "artifacts"))
KAGGLE_ID = "kartik2112/fraud-detection"


def download_kaggle(dest: Path = DATA_DIR) -> None:
    """Téléchargement automatique (nécessite ~/.kaggle/kaggle.json)."""
    import kagglehub

    src = Path(kagglehub.dataset_download(KAGGLE_ID))
    dest.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*.csv"):
        shutil.copy(f, dest / f.name)


def load_raw(source=None) -> pd.DataFrame:
    """Charge fraudTrain.csv + fraudTest.csv (ou un fichier/buffer fourni),
    puis trie par date : le découpage officiel Kaggle est ignoré, on refait
    notre propre séparation chronologique 60/20/20."""
    if source is None:
        files = [DATA_DIR / "fraudTrain.csv", DATA_DIR / "fraudTest.csv"]
        files = [f for f in files if f.exists()]
        if not files:
            raise FileNotFoundError(
                f"Aucun CSV dans {DATA_DIR}. Lancez `python -m src.data` "
                "ou déposez fraudTrain.csv / fraudTest.csv dans data/.")
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    else:
        df = pd.read_csv(source)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    return df.sort_values("trans_date_trans_time", kind="stable").reset_index(drop=True)


def make_eda(df: pd.DataFrame) -> pd.DataFrame:
    """Version légère pour l'exploration (module 1)."""
    ts = df["trans_date_trans_time"]
    return pd.DataFrame({
        "ts": ts, "amt": df["amt"], "is_fraud": df["is_fraud"].astype("int8"),
        "category": df["category"].astype("category"),
        "gender": df["gender"].astype("category"),
        "hour": ts.dt.hour.astype("int8"), "dow": ts.dt.dayofweek.astype("int8"),
    })


if __name__ == "__main__":
    download_kaggle()
    print("Données téléchargées dans", DATA_DIR)
