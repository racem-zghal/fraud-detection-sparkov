import pandas as pd


def chrono_split(feat: pd.DataFrame, train=0.6, val=0.2):
    """60/20/20 chronologique : le futur n'est jamais vu à l'entraînement."""
    feat = feat.sort_values("ts", kind="stable")
    n = len(feat)
    a, b = int(n * train), int(n * (train + val))
    return feat.iloc[:a], feat.iloc[a:b], feat.iloc[b:]
