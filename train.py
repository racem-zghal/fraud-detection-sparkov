"""Pipeline complet : données -> variables -> split chrono -> modèles -> évaluation -> artefacts.
Usage : python train.py [--c-alert 5] [--k-rate 0.01]"""
import argparse
import json
import math

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data import ARTIFACTS_DIR, load_raw, make_eda
from src.evaluate import (best_threshold, metrics_at_threshold, precision_recall_at_k,
                          ranking_metrics)
from src.features import FEATURES, NUMERIC_UNSUP, build_features
from src.split import chrono_split


def xgb(spw=1.0, seed=42):
    return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                         colsample_bytree=0.8, scale_pos_weight=spw, tree_method="hist",
                         eval_metric="aucpr", n_jobs=-1, random_state=seed)


def main(a):
    out = ARTIFACTS_DIR; out.mkdir(parents=True, exist_ok=True)
    print("Chargement…"); df = load_raw()
    make_eda(df).to_parquet(out / "eda.parquet")
    print("Variables…"); feat = build_features(df)
    tr, va, te = chrono_split(feat)
    X = {k: d[FEATURES] for k, d in zip(["train", "val", "test"], [tr, va, te])}
    y = {k: d["is_fraud"].to_numpy() for k, d in zip(["train", "val", "test"], [tr, va, te])}
    split_info = {k: {"n": int(len(d)), "taux_fraude": float(d["is_fraud"].mean()),
                      "debut": str(d["ts"].min()), "fin": str(d["ts"].max())}
                  for k, d in zip(["train", "val", "test"], [tr, va, te])}
    print(json.dumps(split_info, indent=1))

    # ---- 1) Stratégie contre le déséquilibre (appliquée au TRAIN seulement, jugée sur VAL)
    spw = float((y["train"] == 0).sum() / max((y["train"] == 1).sum(), 1))
    rng = np.random.default_rng(a.seed)
    neg = np.where(y["train"] == 0)[0]; pos = np.where(y["train"] == 1)[0]
    sub = np.sort(np.r_[pos, rng.choice(neg, size=min(len(neg), 10 * len(pos)), replace=False)])
    strategies = {
        "Aucune": (X["train"], y["train"], 1.0),
        "Pondération (scale_pos_weight)": (X["train"], y["train"], spw),
        "Sous-échantillonnage 1:10": (X["train"].iloc[sub], y["train"][sub], 1.0),
    }
    imb, fitted = [], {}
    for name, (Xs, ys, w) in strategies.items():
        m = xgb(w, a.seed).fit(Xs, ys)
        s = m.predict_proba(X["val"])[:, 1]
        p, r = precision_recall_at_k(y["val"], s, math.ceil(a.k_rate * len(s)))
        imb.append({"strategie": name, **ranking_metrics(y["val"], s), "precision@k": p, "rappel@k": r})
        fitted[name] = m
    imb = pd.DataFrame(imb); best = imb.sort_values("PR-AUC").iloc[-1]["strategie"]
    print(imb.round(4).to_string()); print("Stratégie retenue :", best)
    xgb_model = fitted[best]

    # ---- 2) Modèles
    lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=500)
                       ).fit(X["train"], y["train"])
    iso = make_pipeline(StandardScaler(), IsolationForest(n_estimators=200, random_state=a.seed, n_jobs=-1)
                        ).fit(X["train"])
    lof_idx = rng.choice(len(X["train"]), size=min(20000, len(X["train"])), replace=False)
    lof = make_pipeline(StandardScaler(), LocalOutlierFactor(n_neighbors=35, novelty=True, n_jobs=-1)
                        ).fit(X["train"][NUMERIC_UNSUP].iloc[lof_idx])
    models = {  # nom -> (type, fonction de score ; plus grand = plus suspect)
        "XGBoost": ("supervisé", lambda d: xgb_model.predict_proba(d[FEATURES])[:, 1]),
        "Régression logistique pondérée": ("supervisé", lambda d: lr.predict_proba(d[FEATURES])[:, 1]),
        "Isolation Forest": ("non supervisé", lambda d: -iso.score_samples(d[FEATURES])),
        "LOF": ("non supervisé", lambda d: -lof.score_samples(d[NUMERIC_UNSUP])),
    }
    raw = {n: {k: f(d) for k, d in zip(["val", "test"], [va, te])} for n, (_, f) in models.items()}
    scaling = {}
    for n, (typ, _) in models.items():  # scores non supervisés ramenés dans [0,1] (min/max de la VAL)
        if typ == "non supervisé":
            lo, hi = float(raw[n]["val"].min()), float(raw[n]["val"].max())
            scaling[n] = (lo, hi)
            for k in raw[n]:
                raw[n][k] = np.clip((raw[n][k] - lo) / (hi - lo + 1e-12), 0, 1)

    # ---- 3) Évaluation : seuil de coût choisi sur VAL, confirmé sur TEST
    rows = []
    for n, (typ, _) in models.items():
        sv, st = raw[n]["val"], raw[n]["test"]
        thr, kb, cv = best_threshold(y["val"], sv, va["amt"].to_numpy(), a.c_alert)
        mt = metrics_at_threshold(y["test"], st, te["amt"].to_numpy(), thr, a.c_alert)
        kt = math.ceil(a.k_rate * len(st)); p, r = precision_recall_at_k(y["test"], st, kt)
        rows.append({"modèle": n, "type": typ, **{f"{k} (val)": v for k, v in ranking_metrics(y["val"], sv).items()},
                     **{f"{k} (test)": v for k, v in ranking_metrics(y["test"], st).items()},
                     f"précision@{kt}": p, f"rappel@{kt}": r, "seuil (val)": thr,
                     "alertes (test)": mt["alertes"], "F2 (test)": mt["F2"], "MCC (test)": mt["MCC"],
                     "coût (test)": mt["cout"]})
    comp = pd.DataFrame(rows); comp.to_csv(out / "comparison.csv", index=False)
    imb.to_csv(out / "imbalance.csv", index=False)
    print(comp.round(4).T.to_string())

    # ---- 4) Artefacts
    sc = pd.concat([pd.DataFrame({"split": k, "y": y[k], "amt": d["amt"].to_numpy(), "ts": d["ts"].to_numpy(),
                                  **{n: raw[n][k] for n in models}})
                    for k, d in zip(["val", "test"], [va, te])], ignore_index=True)
    sc.to_parquet(out / "scores.parquet")
    X["test"].to_parquet(out / "X_test.parquet")
    joblib.dump(xgb_model, out / "model.joblib")
    thr_x, kb, _ = best_threshold(y["val"], raw["XGBoost"]["val"], va["amt"].to_numpy(), a.c_alert)
    total_test = float(te["amt"][te["is_fraud"] == 1].sum())
    json.dump({"features": FEATURES, "models": list(models), "champion": "XGBoost",
               "strategy": best, "c_alert": a.c_alert, "k_rate": a.k_rate,
               "threshold": None if math.isinf(thr_x) else thr_x, "split": split_info,
               "cout_sans_modele_test": total_test}, open(out / "meta.json", "w"), indent=1)
    print("Artefacts écrits dans", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--c-alert", type=float, default=5.0, help="coût d'analyse d'une alerte (même unité que amt)")
    p.add_argument("--k-rate", type=float, default=0.01, help="budget d'alertes en fraction des transactions")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
