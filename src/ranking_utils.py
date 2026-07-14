# src/ranking_utils.py
import ast
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


# ══════════════════════════════════════════════════════════════════════════════
# PALETA E MAPEAMENTO
# ══════════════════════════════════════════════════════════════════════════════

MODEL_COLORS = {
    "Poisson":          "#1f77b4",
    "NegativeBinomial": "#ff7f0e",
    "ZIP":              "#2ca02c",
    "ZINB":             "#d62728",
    "RandomForest":     "#9467bd",
    "SVM_RBF":          "#8c564b",
}

MODEL_LABELS = {
    "Poisson":          "Poisson",
    "NegativeBinomial": "Binomial Negativa",
    "ZIP":              "ZIP",
    "ZINB":             "ZINB",
    "RandomForest":     "Random Forest",
    "SVM_RBF":          "SVR (RBF)",
}

_NAME_MAP = {
    "negativebinomial":  "NegativeBinomial",
    "negative binomial": "NegativeBinomial",
    "binomial negativa": "NegativeBinomial",
    "randomforest":      "RandomForest",
    "random forest":     "RandomForest",
    "rf":                "RandomForest",
    "svm_rbf":           "SVM_RBF",
    "svr (rbf)":         "SVM_RBF",
    "svr":               "SVM_RBF",
    "zip":               "ZIP",
    "zinb":              "ZINB",
    "poisson":           "Poisson",
}


def _norm(name: str) -> str:
    return _NAME_MAP.get(str(name).strip().lower(), str(name).strip())

def _parse_features(raw) -> list:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, tuple):
        return list(raw)
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("[") or raw.startswith("("):
            try:
                return list(ast.literal_eval(raw))
            except Exception:
                pass
        return [raw]
    try:
        return list(raw)
    except Exception:
        return [str(raw)]

def _select_winner(df_results: pd.DataFrame,
                   zinb_invalid_machines: list) -> pd.DataFrame:
    """
    Seleciona o modelo vencedor por máquina pelo critério Score.

    O ZINB é excluído da seleção para as máquinas listadas em
    zinb_invalid_machines — exactamente as identificadas na Sec. 4.2.1
    do documento.

    Isto garante que a distribuição de modelos vencedores aqui é idêntica
    à reportada no documento:
        ZINB: 47  |  Random Forest: 4  |  Poisson: 2  |  SVR: 1
    """
    df = df_results.copy()
    df = df.rename(columns={"tss_mae_mean": "mae_mean",
                             "tss_mae_std":  "mae_std"})
    df["model"]   = df["model"].apply(_norm)
    df            = df[df["mae_mean"] < 1e6].copy()
    df["mae_std"] = df["mae_std"].fillna(0)
    df["score"]   = df["mae_mean"] + 0.5 * df["mae_std"]

    # Excluir ZINB nas máquinas declaradas inválidas na Sec. 4.2.1
    zinb_invalido_mask = (
        (df["model"] == "ZINB") &
        (df["machine"].isin(zinb_invalid_machines))
    )
    n_excluidos = zinb_invalido_mask.sum()
    df = df[~zinb_invalido_mask].copy()

    # Melhor feature-set por (machine, model)
    df_best = (df.sort_values("mae_mean")
                 .groupby(["machine", "model"], as_index=False)
                 .first())

    # Modelo vencedor por máquina
    df_winner = (df_best.sort_values("score")
                        .groupby("machine", as_index=False)
                        .first()
                        .rename(columns={"features": "best_features"}))

    return df_winner[["machine", "model", "best_features",
                       "mae_mean", "mae_std", "score"]]



def _lambda_last_fold(machine: str,
                      model_name: str,
                      best_features,
                      fit_predict_map: dict,
                      df_clear: pd.DataFrame,
                      get_machine_df_fn,
                      build_daily_series_fn,
                      create_time_features_fn,
                      global_start,
                      global_end,
                      n_splits: int = 5,
                      max_lambda: float = 10.0) -> float:
    """
    λ̂_i = média das previsões do último fold TSS.
    Devolve np.nan se o modelo divergir (previsão > max_lambda) ou falhar.
    """
    if model_name not in fit_predict_map:
        return np.nan

    fit_fn, pred_fn = fit_predict_map[model_name]
    feats = _parse_features(best_features)
    if not feats:
        return np.nan

    try:
        mdf   = get_machine_df_fn(df_clear, machine)
        serie = build_daily_series_fn(mdf, global_start, global_end)
        serie = create_time_features_fn(serie).dropna().reset_index(drop=True)

        if serie.empty or serie["n_avarias"].sum() == 0:
            return np.nan
        if any(f not in serie.columns for f in feats):
            return np.nan

        tscv = TimeSeriesSplit(n_splits=n_splits)
        train_idx, test_idx = list(tscv.split(serie))[-1]
        train = serie.iloc[train_idx]
        test  = serie.iloc[test_idx]

        if train["n_avarias"].sum() < 3 or len(test) == 0:
            return np.nan

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model  = fit_fn(train, feats)
            y_pred = pred_fn(model, test, feats)

        arr = (y_pred.values if hasattr(y_pred, "values")
               else np.array(list(y_pred) if hasattr(y_pred, "__iter__")
                              else [float(y_pred)]))
        arr = np.maximum(arr, 0.0)

        media = float(np.mean(arr))
        if not np.isfinite(media) or media > max_lambda:
            return np.nan

        return media

    except Exception:
        return np.nan

def build_risk_ranking(df_results_final: pd.DataFrame,
                       df_clear: pd.DataFrame,
                       get_machine_df_fn,
                       build_daily_series_fn,
                       create_time_features_fn,
                       fit_predict_map: dict,
                       global_start,
                       global_end,
                       zinb_invalid_machines: list = None,
                       n_splits: int = 5,
                       max_lambda: float = 10.0,
                       verbose: bool = True) -> pd.DataFrame:
    """
    Constrói o ranking de risco de avarias.

    Parâmetros
    ----------
    df_results_final       : DataFrame com resultados do TSS
    df_clear               : dados brutos filtrados
    get_machine_df_fn      : get_machine_df de modelling
    build_daily_series_fn  : build_daily_series de modelling
    create_time_features_fn: create_time_features de modelling
    fit_predict_map        : dict {nome_modelo: (fit_fn, predict_fn)}
    global_start/end       : período de análise
    zinb_invalid_machines  : lista de máquinas onde ZINB é inválido
                             (identificadas na Sec. 4.2.1 do documento).
                             Default: ["MIP0000"] (por razões de confidencialidade)
    n_splits               : folds TSS (default 5)
    max_lambda             : limite físico para λ̂ (default 10 avarias/dia)

    Retorna
    -------
    DataFrame com colunas:
        rank, machine, winner_model, model, lambda_hat,
        fallback, mae_mean, mae_std, score
    """
    if zinb_invalid_machines is None:
        zinb_invalid_machines = ["MIP0000"]

    # 1. Modelo vencedor
    if verbose:
        print("▶ [1/3] A selecionar modelo vencedor por máquina...")
        print(f"  ZINB excluído nas máquinas inválidas: "
              f"{', '.join(zinb_invalid_machines)}")

    df_winner = _select_winner(df_results_final, zinb_invalid_machines)

    if verbose:
        print(f"  {len(df_winner)} máquinas identificadas.")
        print("  Distribuição de modelos vencedores:")
        for m, n in df_winner["model"].value_counts().items():
            print(f"    {MODEL_LABELS.get(m, m):25s}: {n}")

    df_winner["winner_model"] = df_winner["model"].copy()

    # 2. λ̂ para cada máquina
    if verbose:
        print("\n▶ [2/3] A calcular λ̂ (média previsões último fold TSS)...")

    lambdas  = []
    fallback = []

    for _, row in df_winner.iterrows():
        lam = _lambda_last_fold(
            machine=row["machine"], model_name=row["model"],
            best_features=row["best_features"],
            fit_predict_map=fit_predict_map, df_clear=df_clear,
            get_machine_df_fn=get_machine_df_fn,
            build_daily_series_fn=build_daily_series_fn,
            create_time_features_fn=create_time_features_fn,
            global_start=global_start, global_end=global_end,
            n_splits=n_splits, max_lambda=max_lambda,
        )
        lambdas.append(lam)
        fallback.append(False)

        if verbose:
            tag = f"λ̂ = {lam:.5f}" if (lam is not None and np.isfinite(lam)) \
                  else "sem estimativa válida"
            print(f"  {row['machine']:10s} | "
                  f"{MODEL_LABELS.get(row['model'], row['model']):22s} | {tag}")

    df_winner["lambda_hat"] = lambdas
    df_winner["fallback"]   = fallback

    # 3. Fallback Poisson para máquinas sem estimativa válida
    nan_mask = df_winner["lambda_hat"].isna() | \
               ~np.isfinite(df_winner["lambda_hat"].fillna(np.nan))
    n_nan = nan_mask.sum()

    if n_nan > 0:
        if verbose:
            print(f"\n  {n_nan} máquinas sem estimativa válida "
                  f"→ a aplicar fallback Poisson...")

        for idx in df_winner[nan_mask].index:
            row = df_winner.loc[idx]
            lam_fb = _lambda_last_fold(
                machine=row["machine"], model_name="Poisson",
                best_features=row["best_features"],
                fit_predict_map=fit_predict_map, df_clear=df_clear,
                get_machine_df_fn=get_machine_df_fn,
                build_daily_series_fn=build_daily_series_fn,
                create_time_features_fn=create_time_features_fn,
                global_start=global_start, global_end=global_end,
                n_splits=n_splits, max_lambda=max_lambda,
            )
            df_winner.loc[idx, "lambda_hat"] = lam_fb
            df_winner.loc[idx, "fallback"]   = True

            if verbose and lam_fb is not None and np.isfinite(lam_fb):
                print(f"  {row['machine']:10s} | "
                      f"fallback Poisson → λ̂ = {lam_fb:.5f}")

    # 4. Ranking
    if verbose:
        print("\n▶ [3/3] A construir ranking...")

    df_ranking = (df_winner
                  .sort_values("lambda_hat", ascending=False,
                               na_position="last")
                  .reset_index(drop=True))
    df_ranking.insert(0, "rank", range(1, len(df_ranking) + 1))

    n_ok  = df_ranking["lambda_hat"].notna().sum()
    n_nan = df_ranking["lambda_hat"].isna().sum()
    n_fb  = int(df_ranking["fallback"].sum())

    if verbose:
        print(f"\n✓ Ranking construído.")
        print(f"  Estimativas via modelo vencedor : {n_ok - n_fb}")
        print(f"  Estimativas via fallback Poisson: {n_fb}")
        print(f"  Sem estimativa                  : {n_nan}")
        print("\nTop 5 de maior risco:")
        top5 = df_ranking.head(5)[["rank", "machine", "winner_model",
                                    "lambda_hat", "fallback"]].copy()
        top5["winner_model"] = (top5["winner_model"]
                                .map(MODEL_LABELS)
                                .fillna(top5["winner_model"]))
        print(top5.to_string(index=False))

    return df_ranking[["rank", "machine", "winner_model", "model",
                        "lambda_hat", "fallback",
                        "mae_mean", "mae_std", "score","best_features"]]