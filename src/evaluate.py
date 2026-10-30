"""Métriques pour événements rares, budget d'alertes et coût métier.
Coût = montants des fraudes manquées + c_alert × nombre d'alertes analysées."""
import numpy as np
from sklearn.metrics import average_precision_score, matthews_corrcoef, fbeta_score, roc_auc_score


def precision_recall_at_k(y, s, k):
    k = int(min(max(k, 1), len(y)))
    idx = np.argpartition(-s, k - 1)[:k]
    tp = y[idx].sum()
    return tp / k, tp / max(y.sum(), 1)


def metrics_at_threshold(y, s, amt, thr, c_alert):
    pred = s >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    missed = float(amt[(y == 1) & ~pred].sum())
    n_alert = int(pred.sum())
    return {
        "alertes": n_alert, "tp": tp, "fp": fp,
        "precision": tp / n_alert if n_alert else 0.0,
        "rappel": tp / max(int(y.sum()), 1),
        "F2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "MCC": float(matthews_corrcoef(y, pred)) if 0 < n_alert < len(y) else 0.0,
        "montant_manque": missed,
        "cout": missed + c_alert * n_alert,
    }


def cost_curve(y, s, amt, c_alert):
    """Coût en fonction du nombre d'alertes k = 0..n (top-k par score)."""
    order = np.argsort(-s, kind="stable")
    fa = np.where(y == 1, amt, 0.0)[order]
    total = fa.sum()
    k = np.arange(1, len(y) + 1)
    return np.r_[total, (total - np.cumsum(fa)) + c_alert * k]


def best_threshold(y, s, amt, c_alert):
    """Seuil minimisant le coût (à appeler sur la VALIDATION)."""
    cc = cost_curve(y, s, amt, c_alert)
    kb = int(np.argmin(cc))
    thr = np.inf if kb == 0 else float(np.sort(s)[::-1][kb - 1])
    return thr, kb, float(cc[kb])


def threshold_for_k(s, k):
    k = int(min(max(k, 1), len(s)))
    return float(np.sort(s)[::-1][k - 1])


def ranking_metrics(y, s):
    return {"PR-AUC": float(average_precision_score(y, s)), "ROC-AUC": float(roc_auc_score(y, s))}
