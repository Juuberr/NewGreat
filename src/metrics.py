from narwhals import col
import numpy as np
import pandas as pd

from scipy.stats import ks_2samp
from sklearn.metrics import f1_score, recall_score
from sklearn.neighbors import NearestNeighbors


def utility_metrics(y_true, y_pred, label=""):
    """F1 (macro) und Recall der Minderheitsklasse (class=1, d.h. >50K)."""
    f1  = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1,  zero_division=0)
    return {"F1 (macro)": round(f1, 4), "Recall >50K": round(rec, 4)}


def fidelity_metrics(real, synth, cols):

    ks_scores = []

    #print(type(real))
    #print(type(synth))
    #print(real.shape)
    #print(synth.shape)

    for col in cols:

        #print("=" * 40)
        #print(col)

        #print(real[col].dtype)
        #print(synth[col].dtype)

        r = real[col].dropna().astype(float).to_numpy()
        s = synth[col].dropna().astype(float).to_numpy()

        #print(type(r), r.dtype, r.shape)
        #print(type(s), s.dtype, s.shape)

        #rint(r[:5])
        #rint(s[:5])

        stat, _ = ks_2samp(r, s)

        ks_scores.append(stat)

    return {
        "Ø KS-Statistik": round(float(np.mean(ks_scores)),4)
    }

"""
def fidelity_metrics(real: pd.DataFrame, synth: pd.DataFrame, cols: list):
    
    # Fidelity: mittlere KS-Statistik über numerische Spalten.
    

    ks_scores = []

    for col in cols:

        if col not in real.columns or col not in synth.columns:
            continue

        real_vals = (
            pd.to_numeric(real[col], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )

        synth_vals = (
            pd.to_numeric(synth[col], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )

        if len(real_vals) == 0 or len(synth_vals) == 0:
            continue

        print(f"{col=}")
        print(type(np.real[col]))
        print(type(synth[col]))
        print(np.realeal[col].dtype)
        print(synth[col].dtype)

        real_vals = pd.to_numeric(np.real[col], errors="coerce").dropna().to_numpy(dtype=float)
        synth_vals = pd.to_numeric(synth[col], errors="coerce").dropna().to_numpy(dtype=float)

        print(real_vals.dtype, synth_vals.dtype)
        print(real_vals.shape, synth_vals.shape)

        try:
            stat, _ = ks_2samp(real_vals, synth_vals)
            ks_scores.append(stat)
        except Exception as e:
            print(f"KS failed for column {col}: {e}")

    mean_ks = round(float(np.mean(ks_scores)), 4) if ks_scores else float("nan")

    return {"Ø KS-Statistik": mean_ks}
"""

def privacy_nndr(real_train: np.ndarray, synth: np.ndarray, sample_n=2000, seed=42):
    """
    Nearest-Neighbor Distance Ratio (NNDR).
    Für jeden synthetischen Punkt: dist(1-NN_real) / dist(2-NN_real).

    Interpretation
    ──────────────
    NNDR ≈ 1  → 1st und 2nd Neighbor gleich weit → Punkt liegt nicht direkt
                auf einem Trainingspunkt → kein Memorization-Risiko.
    NNDR → 0  → Punkt ist fast identisch mit einem echten Trainingspunkt.

    Hinweis SMOTE: SMOTE interpoliert *zwischen* echten Punkten,
    daher liegt der nächste Nachbar typischerweise auf 0 Distanz.
    Das ist kein Bug, sondern zeigt, dass SMOTE keine echte Privatheit bietet.
    """
    rng = np.random.default_rng(seed)
    if len(synth) > sample_n:
        idx = rng.choice(len(synth), sample_n, replace=False)
        synth = synth[idx]

    nn = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(real_train)
    dists, _ = nn.kneighbors(synth)          # shape (n, 2)
    eps = 1e-9
    nndr = dists[:, 0] / (dists[:, 1] + eps)
    median_nndr = round(float(np.median(nndr)), 4)

    pct_copies = round(float(np.mean(nndr < 0.05)) * 100, 1)
    return {"Median NNDR": median_nndr, "% quasi-Kopien": pct_copies}
