"""Génère un jeu factice au format Sparkov pour tester le pipeline hors-ligne."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import CATEGORIES


def make(out_dir, n=80000, n_cards=300, seed=0):
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cards = rng.integers(10**15, 4 * 10**18, n_cards)
    home = np.c_[rng.uniform(30, 47, n_cards), rng.uniform(-120, -75, n_cards)]
    dob = pd.to_datetime("1950-01-01") + pd.to_timedelta(rng.integers(0, 365 * 45, n_cards), "D")
    gender = rng.choice(["M", "F"], n_cards)
    pop = rng.integers(500, 2_000_000, n_cards)
    t0 = pd.Timestamp("2019-01-01").value // 10**9
    ts = np.sort(rng.integers(t0, t0 + 700 * 86400, n))
    c = rng.integers(0, n_cards, n)
    fraud = rng.random(n) < 0.012
    ts = np.where(fraud, ts - ts % 86400 + rng.integers(22, 27, n) * 3600 % 86400 + rng.integers(0, 3600, n), ts)
    amt = np.where(fraud, rng.lognormal(6, 0.5, n), rng.lognormal(3.5, 1.0, n)).round(2)
    cat = np.where(fraud & (rng.random(n) < 0.7),
                   rng.choice(["shopping_net", "misc_net", "grocery_pos"], n), rng.choice(CATEGORIES, n))
    spread = np.where(fraud, 6.0, 0.4)
    df = pd.DataFrame({
        "trans_date_trans_time": pd.to_datetime(ts, unit="s").strftime("%Y-%m-%d %H:%M:%S"),
        "cc_num": cards[c], "merchant": "fraud_Shop", "category": cat, "amt": amt,
        "gender": gender[c], "city_pop": pop[c], "dob": dob[c].strftime("%Y-%m-%d"),
        "unix_time": ts, "lat": home[c, 0], "long": home[c, 1],
        "merch_lat": home[c, 0] + rng.normal(0, 1, n) * spread,
        "merch_long": home[c, 1] + rng.normal(0, 1, n) * spread,
        "is_fraud": fraud.astype(int)})
    df = df.sort_values("unix_time").reset_index(drop=True)
    cut = int(n * 0.75)
    df.iloc[:cut].to_csv(out_dir / "fraudTrain.csv")
    df.iloc[cut:].to_csv(out_dir / "fraudTest.csv")
    print(f"{n} transactions synthétiques -> {out_dir}  (taux fraude {fraud.mean():.2%})")


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "data")
