"""Application Streamlit : lancer avec  streamlit run app.py"""
import json
import math

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.metrics import precision_recall_curve

from src.data import ARTIFACTS_DIR as ART, download_kaggle, load_raw, make_eda
from src.evaluate import best_threshold, cost_curve, metrics_at_threshold, threshold_for_k
from src.explain import label, reason_codes

st.set_page_config(page_title="Détection de fraude explicable", layout="wide")
st.title("🛡️ Détection de fraude explicable — Sparkov")

if not (ART / "meta.json").exists():
    st.error("Artefacts introuvables. Lancez d'abord `python train.py`.")
    st.stop()


@st.cache_data
def load_artifacts():
    return (json.load(open(ART / "meta.json")), pd.read_parquet(ART / "scores.parquet"),
            pd.read_parquet(ART / "eda.parquet"), pd.read_csv(ART / "comparison.csv"),
            pd.read_csv(ART / "imbalance.csv"))


@st.cache_resource
def load_model():
    model = joblib.load(ART / "model.joblib")
    return model, shap.TreeExplainer(model), pd.read_parquet(ART / "X_test.parquet")


meta, scores, eda, comp, imb = load_artifacts()
t1, t2, t3, t4, t5 = st.tabs(["1 · Données", "2 · Préparation", "3 · Modèles", "4 · Évaluation & seuil",
                              "5 · Explications"])

# ---------------------------------------------------------------- 1. Données
with t1:
    with st.expander("Charger / recharger les données"):
        c1, c2 = st.columns(2)
        if c1.button("Télécharger depuis Kaggle"):
            download_kaggle(); st.success("Téléchargé dans data/ — relancez train.py")
        up = c2.file_uploader("…ou importer un CSV au format Sparkov", type="csv")
        if up is not None:
            eda = make_eda(load_raw(up)); st.info("Exploration sur le fichier importé (le modèle reste inchangé).")
    a, b, c, d = st.columns(4)
    a.metric("Transactions", f"{len(eda):,}"); b.metric("Fraudes", f"{eda.is_fraud.sum():,}")
    c.metric("Taux de fraude", f"{eda.is_fraud.mean():.3%}")
    d.metric("Période", f"{eda.ts.min():%Y-%m} → {eda.ts.max():%Y-%m}")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Taux de fraude par mois")
        st.line_chart(eda.groupby(eda.ts.dt.to_period("M").astype(str)).is_fraud.mean())
        st.subheader("Taux de fraude par heure")
        st.bar_chart(eda.groupby("hour").is_fraud.mean())
    with col2:
        st.subheader("Distribution des montants par classe")
        fig, ax = plt.subplots()
        bins = np.logspace(0, np.log10(max(eda.amt.max(), 10)), 50)
        for v, nm in [(0, "légitime"), (1, "fraude")]:
            ax.hist(eda.amt[eda.is_fraud == v].clip(lower=1), bins=bins, alpha=.6, density=True, label=nm)
        ax.set_xscale("log"); ax.set_xlabel("montant"); ax.legend(); st.pyplot(fig); plt.close(fig)
        st.subheader("Taux de fraude par catégorie")
        st.bar_chart(eda.groupby("category", observed=True).is_fraud.mean().sort_values())

# ---------------------------------------------------------------- 2. Préparation
with t2:
    st.subheader("Séparation chronologique 60 / 20 / 20")
    st.dataframe(pd.DataFrame(meta["split"]).T.rename(index={"train": "Entraînement", "val": "Validation", "test": "Test"}))
    st.markdown("""**Variables** : heure en codage cyclique (sin/cos), jour de semaine, nuit/week-end, distance
client–marchand, âge, et variables **comportementales par carte** (nb de transactions sur 1 h / 24 h,
temps depuis la précédente, montant ÷ moyenne des 30 derniers jours) calculées **sur le passé strict**
(la transaction courante est exclue) : pas de fuite de données.""")
    st.subheader("Stratégie contre le déséquilibre (train seul, comparée sur la validation)")
    st.dataframe(imb.style.format({c: "{:.4f}" for c in imb.columns if c != "strategie"}))
    st.success(f"Stratégie retenue : **{meta['strategy']}** (meilleure PR-AUC en validation)")

# ---------------------------------------------------------------- 3. Modèles
with t3:
    st.subheader("Tableau comparatif")
    st.dataframe(comp.style.format({c: "{:.4f}" for c in comp.columns if c not in ("modèle", "type", "alertes (test)")}))
    split_pr = st.radio("Courbes précision-rappel sur", ["test", "val"], horizontal=True)
    d = scores[scores.split == split_pr]
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in meta["models"]:
        p, r, _ = precision_recall_curve(d.y, d[m]); ax.plot(r, p, label=m)
    ax.axhline(d.y.mean(), ls="--", c="gray", label="hasard"); ax.set_xlabel("rappel"); ax.set_ylabel("précision")
    ax.legend(); st.pyplot(fig); plt.close(fig)

# ---------------------------------------------------------------- 4. Évaluation & seuil
with t4:
    c1, c2, c3 = st.columns(3)
    mdl = c1.selectbox("Modèle", meta["models"], index=0)
    split = c2.radio("Jeu évalué", ["test", "val"], horizontal=True)
    c_alert = c3.number_input("Coût d'analyse par alerte", 0.0, 10000.0, float(meta["c_alert"]), 1.0)
    v, d = scores[scores.split == "val"], scores[scores.split == split]
    thr_opt, k_opt, cost_opt = best_threshold(v.y.to_numpy(), v[mdl].to_numpy(), v.amt.to_numpy(), c_alert)
    thr_opt = min(thr_opt, 1.0)
    mode = st.radio("Piloter par", ["Seuil", "Budget d'alertes k"], horizontal=True)
    s, y, amt = d[mdl].to_numpy(), d.y.to_numpy(), d.amt.to_numpy()
    if mode == "Seuil":
        thr = st.slider("Seuil d'alerte", 0.0, 1.0, float(thr_opt), 0.001, key=f"thr_{mdl}")
    else:
        pct = st.slider("Budget k (% des transactions analysées)", 0.01, 5.0, meta["k_rate"] * 100, 0.01, key=f"k_{mdl}")
        thr = threshold_for_k(s, math.ceil(pct / 100 * len(s)))
    m = metrics_at_threshold(y, s, amt, thr, c_alert)
    base = float(amt[y == 1].sum())
    r1 = st.columns(4)
    r1[0].metric("Alertes", f"{m['alertes']:,}"); r1[1].metric("Précision", f"{m['precision']:.1%}")
    r1[2].metric("Rappel", f"{m['rappel']:.1%}"); r1[3].metric("Fraudes détectées", f"{m['tp']} / {int(y.sum())}")
    r2 = st.columns(4)
    r2[0].metric("F2", f"{m['F2']:.3f}"); r2[1].metric("MCC", f"{m['MCC']:.3f}")
    r2[2].metric("Coût total", f"{m['cout']:,.0f}", f"{m['cout'] - base:,.0f} vs sans modèle", delta_color="inverse")
    r2[3].metric("Montant des fraudes manquées", f"{m['montant_manque']:,.0f}")
    st.caption(f"Seuil optimal choisi sur la **validation** : {thr_opt:.4f} "
               f"(coût validation {cost_opt:,.0f}, {k_opt} alertes). Sur {split} avec ce seuil : "
               f"coût {metrics_at_threshold(y, s, amt, thr_opt, c_alert)['cout']:,.0f}.")
    cc = cost_curve(y, s, amt, c_alert); ks = np.unique(np.r_[0, np.geomspace(1, len(s), 600).astype(int)])
    fig, ax = plt.subplots(figsize=(7, 3.5)); ax.plot(ks / len(s) * 100, cc[ks])
    ax.axvline(m["alertes"] / len(s) * 100, c="r", ls="--", label="réglage courant")
    ax.set_xscale("symlog", linthresh=0.01); ax.set_xlim(0, 100); ax.set_xlabel("% de transactions alertées"); ax.set_ylabel("coût")
    ax.legend(); st.pyplot(fig); plt.close(fig)

# ---------------------------------------------------------------- 5. Explications
with t5:
    model, explainer, Xt = load_model()
    feats = meta["features"]
    st.subheader("Explication globale (SHAP beeswarm)")
    @st.cache_data
    def global_shap():
        yt = scores[scores.split == "test"].y.to_numpy()
        fr = np.where(yt == 1)[0]; ok = np.where(yt == 0)[0]
        rng = np.random.default_rng(0)
        idx = np.r_[rng.choice(fr, min(500, len(fr)), replace=False),
                    rng.choice(ok, min(2000, len(ok)), replace=False)]
        samp = Xt.iloc[idx]
        return samp, explainer.shap_values(samp)
    samp, sv = global_shap()
    st.caption("Échantillon du test : 500 fraudes + 2 000 transactions légitimes (les fraudes sont "
               "sur-représentées pour que le graphique soit lisible ; les fraudes ne font que ~0,4 % du test).")
    fig = plt.figure(); shap.summary_plot(sv, samp, show=False, max_display=12)
    st.pyplot(plt.gcf()); plt.close("all")

    st.subheader("Explication d'une alerte")
    te = scores[scores.split == "test"].reset_index(drop=True)
    top = te["XGBoost"].sort_values(ascending=False).head(100)
    pick = st.selectbox("Alerte (classées par score)", top.index, format_func=lambda i: (
        f"#{list(top.index).index(i) + 1} · score {te.XGBoost[i]:.3f} · montant {te.amt[i]:.0f} · "
        f"{'FRAUDE' if te.y[i] else 'légitime'}"))
    row = Xt.iloc[[pick]]
    ex = explainer(row)[0]
    ex.feature_names = [label(f) for f in feats]
    cl, cr = st.columns([3, 2])
    with cl:
        shap.plots.waterfall(ex, max_display=10, show=False); st.pyplot(plt.gcf()); plt.close("all")
    with cr:
        st.markdown("**Codes motifs** (3 à 5 variables qui poussent vers « fraude »)")
        for r in reason_codes(ex.values, row.iloc[0].to_numpy(), feats, top=5):
            st.write(f"• **{r['motif']}** = {r['valeur']:.2f}  (SHAP +{r['contribution_shap']:.2f})")
