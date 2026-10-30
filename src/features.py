"""Variables temporelles, géographiques et comportementales.
Toutes les variables comportementales n'utilisent QUE le passé strict de
chaque carte (aucune fuite) : la transaction courante est exclue des fenêtres."""
import numpy as np
import pandas as pd

CATEGORIES = ["entertainment", "food_dining", "gas_transport", "grocery_net",
              "grocery_pos", "health_fitness", "home", "kids_pets", "misc_net",
              "misc_pos", "personal_care", "shopping_net", "shopping_pos", "travel"]
BASE = ["amt", "log_amt", "hour_sin", "hour_cos", "dow", "is_weekend", "is_night",
        "dist_km", "age", "log_city_pop", "is_male",
        "n_tx_1h", "n_tx_24h", "time_since_prev_h", "amt_ratio_30d"]
FEATURES = BASE + [f"cat_{c}" for c in CATEGORIES]
# sous-ensemble continu pour LOF (pas de one-hot : évite la malédiction de la dimension)
NUMERIC_UNSUP = ["log_amt", "hour_sin", "hour_cos", "dist_km", "age",
                 "n_tx_1h", "n_tx_24h", "time_since_prev_h", "amt_ratio_30d"]
GAP_CAP_H = 24 * 30


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def behavioral(df: pd.DataFrame) -> pd.DataFrame:
    """df trié par temps. Fenêtres glissantes par carte, passé strict."""
    n = len(df)
    t = df["unix_time"].to_numpy(np.int64)
    amt = df["amt"].to_numpy(float)
    card = df["cc_num"].to_numpy()
    n1, n24 = np.zeros(n), np.zeros(n)
    gap, ratio = np.full(n, GAP_CAP_H, float), np.ones(n)
    order = np.argsort(card, kind="stable")  # stable : l'ordre temporel est conservé
    _, starts = np.unique(card[order], return_index=True)
    for s, e in zip(starts, list(starts[1:]) + [n]):
        idx = order[s:e]
        tt, aa = t[idx], amt[idx]
        pos = np.arange(len(idx))
        n1[idx] = pos - np.searchsorted(tt, tt - 3600, side="left")
        n24[idx] = pos - np.searchsorted(tt, tt - 86400, side="left")
        if len(idx) > 1:
            gap[idx[1:]] = np.minimum(np.diff(tt) / 3600, GAP_CAP_H)
        cs = np.r_[0.0, np.cumsum(aa)]
        lo = np.searchsorted(tt, tt - 30 * 86400, side="left")
        cnt, tot = pos - lo, cs[pos] - cs[lo]
        mean30 = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
        ratio[idx] = np.where(cnt > 0, aa / np.where(mean30 > 0, mean30, 1), 1.0)
    return pd.DataFrame({"n_tx_1h": n1, "n_tx_24h": n24,
                         "time_since_prev_h": gap, "amt_ratio_30d": ratio}, index=df.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df doit être trié par date. Retourne FEATURES + is_fraud, ts."""
    ts = pd.to_datetime(df["trans_date_trans_time"])
    hour = ts.dt.hour + ts.dt.minute / 60
    out = pd.DataFrame(index=df.index)
    out["amt"] = df["amt"].astype(float)
    out["log_amt"] = np.log1p(out["amt"])
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow"] = ts.dt.dayofweek
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["is_night"] = ((ts.dt.hour >= 22) | (ts.dt.hour <= 3)).astype(int)
    out["dist_km"] = haversine(df["lat"], df["long"], df["merch_lat"], df["merch_long"])
    out["age"] = (ts - pd.to_datetime(df["dob"])).dt.days / 365.25
    out["log_city_pop"] = np.log1p(df["city_pop"])
    out["is_male"] = (df["gender"] == "M").astype(int)
    out = out.join(behavioral(df))
    for c in CATEGORIES:
        out[f"cat_{c}"] = (df["category"] == c).astype(int)
    out["is_fraud"] = df["is_fraud"].astype(int) if "is_fraud" in df else 0
    out["ts"] = ts
    return out
