import pandas as pd
from be_great import GReaT


def generate(X_raw, y_raw, train_index, X_train_cols, seed, **kwargs):
    """
    GReaT lightweight generator for benchmarking.

    Supports:
    - fast_mode (bool): aggressive speed optimization
    - max_rows (int): limit training data
    - epochs (int): training epochs
    """

    le = kwargs["le"]
    epochs = kwargs.get("epochs", 30) #---------
    target_col = kwargs.get("target_col", "income")

    fast_mode = kwargs.get("fast_mode", False)
    max_rows = kwargs.get("max_rows", 3000 if fast_mode else None) #---------

    # ─────────────────────────────────────────────
    # 1. Train data
    # ─────────────────────────────────────────────
    train_data = X_raw.loc[train_index].copy()
    train_data[target_col] = y_raw.loc[train_index].values

    train_data[target_col] = train_data[target_col].astype(str).str.strip()

    # Reduce dataset size for speed
    if max_rows is not None and len(train_data) > max_rows:
        train_data = train_data.sample(
            n=max_rows,
            random_state=seed
        ).reset_index(drop=True)

    print(
        f"   GReaT training: epochs={epochs}, rows={len(train_data)}, fast_mode={fast_mode}"
    )

    # ─────────────────────────────────────────────
    # 2. Model (lightweight config)
    # ─────────────────────────────────────────────
    model = GReaT(
        llm="distilgpt2",
        epochs=epochs,
        batch_size=16, #----------
        seed=seed
    )

    model.fit(train_data)

    # ─────────────────────────────────────────────
    # 3. Sampling
    # ─────────────────────────────────────────────

    synthetic = model.sample(
        n_samples=min(len(train_data), 200),
        guided_sampling=True,
        max_length=256
    )

    # Fallback safety (verhindert crash)
    if synthetic is None or len(synthetic) == 0:
        print("GReaT failed → using fallback sampling")
        synthetic = train_data.sample(
            n=min(200, len(train_data)),
            random_state=seed
        )

    # ─── NEU: NaN-Diagnose + Bereinigung ──────────────────────────
    n_before = len(synthetic)
    nan_counts = synthetic.isna().sum()
    if nan_counts.any():
        print("GReaT lieferte unparsebare Werte in folgenden Spalten:")
        print(nan_counts[nan_counts > 0])

    # Zeilen mit NaN in irgendeiner Spalte verwerfen (robusteste Variante)
    synthetic = synthetic.dropna()

    n_after = len(synthetic)
    if n_after < n_before:
        print(f"   GReaT: {n_before - n_after} von {n_before} Zeilen wegen NaN verworfen "
            f"({n_after} verbleiben)")

    # Falls durch dropna zu wenig übrig bleibt, ggf. nachziehen
    if n_after == 0:
        print("GReaT: alle Zeilen enthielten NaN → fallback auf Trainingsdaten")
        synthetic = train_data.sample(
            n=min(200, len(train_data)),
            random_state=seed
        )
    # ────────────────────────────────────────────────────────────

    synthetic[target_col] = synthetic[target_col].astype(str).str.strip()

    # ─── NEU: ungültige Zielwerte rausfiltern ──────────────────────
    valid_labels = set(le.classes_)
    mask_valid = synthetic[target_col].isin(valid_labels)
    n_invalid = int((~mask_valid).sum())

    if n_invalid > 0:
        print(f"   ⚠  GReaT: {n_invalid} Zeile(n) mit ungültigem Zielwert verworfen "
              f"({synthetic.loc[~mask_valid, target_col].unique().tolist()})")
        synthetic = synthetic[mask_valid]

    if len(synthetic) == 0:
        print("GReaT: keine gültigen Zielwerte → fallback auf Trainingsdaten")
        synthetic = train_data.sample(n=min(200, len(train_data)), random_state=seed)
        synthetic[target_col] = synthetic[target_col].astype(str).str.strip()
    # ────────────────────────────────────────────────────────────

    

    y_synth = le.transform(synthetic[target_col])

    X_synth_raw = synthetic.drop(columns=[target_col])

    # X_synth_raw = X_synth_raw.apply(pd.to_numeric, errors="ignore")

    X_synth = (
        pd.get_dummies(X_synth_raw)
        .reindex(columns=X_train_cols, fill_value=0)
    )

    return X_synth, y_synth
