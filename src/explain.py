"""Codes motifs : variables qui contribuent le plus à pousser le score vers 'fraude'."""
import numpy as np

LABELS = {
    "amt": "Montant de la transaction", "log_amt": "Montant (échelle log)",
    "hour_sin": "Heure (composante sin)", "hour_cos": "Heure (composante cos)",
    "dow": "Jour de la semaine", "is_weekend": "Week-end", "is_night": "Transaction nocturne",
    "dist_km": "Distance client–marchand", "age": "Âge du porteur",
    "log_city_pop": "Taille de la ville", "is_male": "Genre",
    "n_tx_1h": "Nb de transactions sur 1 h", "n_tx_24h": "Nb de transactions sur 24 h",
    "time_since_prev_h": "Temps depuis la transaction précédente (h)",
    "amt_ratio_30d": "Montant / moyenne des 30 derniers jours",
}


def label(feature: str) -> str:
    if feature.startswith("cat_"):
        return "Catégorie marchand : " + feature[4:]
    return LABELS.get(feature, feature)


GROUPS = {"amt": ["amt", "log_amt"], "hour": ["hour_sin", "hour_cos"]}  # variables redondantes -> un seul motif


def reason_codes(shap_row, x_row, features, top=5):
    """Contributions positives les plus fortes ; amt/log_amt et hour_sin/hour_cos sont fusionnés
    (SHAP répartit le crédit entre variables corrélées, ce qui dupliquerait le motif)."""
    shap_row = np.asarray(shap_row, float); idx = {f: i for i, f in enumerate(features)}
    items = []
    for f, i in idx.items():
        if any(f in g for g in GROUPS.values()):
            continue
        items.append((f, label(f), float(x_row[i]), float(shap_row[i])))
    for name, members in GROUPS.items():
        mem = [m for m in members if m in idx]
        if not mem:
            continue
        c = sum(shap_row[idx[m]] for m in mem)
        if name == "amt":
            items.append(("amt", "Montant de la transaction", float(x_row[idx["amt"]]), c))
        else:
            h = (np.arctan2(x_row[idx["hour_sin"]], x_row[idx["hour_cos"]]) * 24 / (2 * np.pi)) % 24
            items.append(("hour", "Heure de la transaction", float(h), c))
    items = sorted(items, key=lambda t: -t[3])[:top]
    return [{"code": f, "motif": m, "valeur": v, "contribution_shap": c} for f, m, v, c in items if c > 0]
