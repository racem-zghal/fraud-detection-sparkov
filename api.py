"""API FastAPI : score + codes motifs pour une transaction brute.
Lancer : uvicorn api:app --reload   (docs : http://127.0.0.1:8000/docs)"""
import json

import joblib
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel

from src.data import ARTIFACTS_DIR
from src.explain import reason_codes
from src.features import build_features

meta = json.load(open(ARTIFACTS_DIR / "meta.json"))
model = joblib.load(ARTIFACTS_DIR / "model.joblib")
explainer = shap.TreeExplainer(model)
FEATURES = meta["features"]
THR = meta["threshold"] if meta["threshold"] is not None else 1.0

app = FastAPI(title="Détection de fraude explicable", version="1.0")


class Tx(BaseModel):
    trans_date_trans_time: str
    cc_num: int
    amt: float
    category: str
    gender: str
    lat: float
    long: float
    merch_lat: float
    merch_long: float
    city_pop: int
    dob: str


class Past(BaseModel):
    trans_date_trans_time: str
    amt: float


class Request(BaseModel):
    transaction: Tx
    history: list[Past] = []  # transactions passées de la même carte (30 derniers jours)


def to_frame(req: Request) -> pd.DataFrame:
    tx = req.transaction.model_dump()
    rows = [{**tx, **h.model_dump()} for h in req.history] + [tx]
    df = pd.DataFrame(rows)
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["unix_time"] = df["trans_date_trans_time"].astype("int64") // 10**9
    df["is_fraud"] = 0
    cur = df.iloc[[-1]]
    past = df.iloc[:-1]
    past = past[past["unix_time"] <= cur["unix_time"].iloc[0]]
    return pd.concat([past, cur]).sort_values("unix_time", kind="stable").reset_index(drop=True)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def info():
    return {"modele": meta["champion"], "strategie_desequilibre": meta["strategy"],
            "seuil": THR, "cout_alerte": meta["c_alert"], "variables": FEATURES, "split": meta["split"]}


@app.post("/predict")
def predict(req: Request):
    feat = build_features(to_frame(req))
    row = feat[FEATURES].iloc[[-1]]  # la transaction courante est la dernière
    score = float(model.predict_proba(row)[:, 1][0])
    sv = explainer.shap_values(row)[0]
    return {"score": score, "alerte": score >= THR, "seuil": THR,
            "niveau_risque": "élevé" if score >= THR else ("moyen" if score >= THR / 2 else "faible"),
            "codes_motifs": reason_codes(sv, row.iloc[0].to_numpy(), FEATURES, top=5)}
