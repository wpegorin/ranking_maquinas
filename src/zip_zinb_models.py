"""
zip_zinb_models.py
Implementação de ZIP e ZINB compatível com o pipeline existente
em modelling_d.py (fit_func / predict_func pattern).

Dependências: statsmodels >= 0.14
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import statsmodels.api as sm
from statsmodels.discrete.count_model import (
    ZeroInflatedPoisson,
    ZeroInflatedNegativeBinomialP,
)
#import warnings
#warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _prepare_X(df, features):
    """Adiciona constante e devolve X como array (necessário para ZIP/ZINB)."""
    X = sm.add_constant(df[features].copy(), has_constant="add")
    return X


def _safe_predict(result, X_test):
    """
    Devolve E[Y] = predict() clampado a [0, inf).
    ZIP/ZINB do statsmodels devolvem E[Y] directamente com predict().
    """
    y_pred = result.predict(X_test)
    return np.maximum(np.array(y_pred), 0.0)


# ─────────────────────────────────────────────────────────────
# ZIP — Zero-Inflated Poisson
# ─────────────────────────────────────────────────────────────

def fit_zip_model(df, features, maxiter=200, gtol=1e-5):
    """
    Ajusta um modelo ZIP usando statsmodels.
    A componente de inflação de zeros usa um logit com intercepto apenas
    (sem covariáveis no processo de mistura) — escolha conservadora
    adequada ao tamanho das séries individuais.

    Retorna o resultado ajustado ou lança excepção se não convergir.
    """
    X = _prepare_X(df, features)
    y = df["n_avarias"].values

    # exog_infl=1 → apenas intercepto no processo de mistura
    model = ZeroInflatedPoisson(
        endog=y,
        exog=X,
        exog_infl=np.ones((len(y), 1)),  # intercepto no logit de zeros
        inflation="logit",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(
            method="bfgs",
            maxiter=maxiter,
            gtol=gtol,
            disp=False,
        )

    return result


def zip_predict(result, df_test, features):
    """Previsão E[Y] para ZIP."""
    X_test = _prepare_X(df_test, features)
    # Para modelos ZIP/ZINB do statsmodels, predict() devolve E[Y]
    # que já incorpora a probabilidade de zero estrutural.
    exog_infl_test = np.ones((len(df_test), 1))
    y_pred = result.predict(X_test, exog_infl=exog_infl_test)
    return pd.Series(np.maximum(y_pred, 0.0), index=df_test.index)


# ─────────────────────────────────────────────────────────────
# ZINB — Zero-Inflated Negative Binomial
# ─────────────────────────────────────────────────────────────

def fit_zinb_model(df, features, maxiter=300, gtol=1e-5):
    """
    Ajusta um modelo ZINB (parametrização NB-P com P=2, equivalente a NB2).
    Componente de inflação: logit com intercepto apenas.
    """
    X = _prepare_X(df, features)
    y = df["n_avarias"].values

    model = ZeroInflatedNegativeBinomialP(
        endog=y,
        exog=X,
        exog_infl=np.ones((len(y), 1)),
        inflation="logit",
        p=2,  # NB2: Var = mu + alpha * mu^2
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(
            method="bfgs",
            maxiter=maxiter,
            gtol=gtol,
            disp=False,
        )

    return result


def zinb_predict(result, df_test, features):
    """Previsão E[Y] para ZINB."""
    X_test = _prepare_X(df_test, features)
    exog_infl_test = np.ones((len(df_test), 1))
    y_pred = result.predict(X_test, exog_infl=exog_infl_test)
    return pd.Series(np.maximum(y_pred, 0.0), index=df_test.index)


# ─────────────────────────────────────────────────────────────
# TSS ROBUSTO PARA ZIP/ZINB
# ─────────────────────────────────────────────────────────────

def evaluate_zip_zinb_for_machine(serie_d_clean, feature_sets,
                                   n_splits=5, min_train_events=5):
    """
    Corre ZIP e ZINB sobre todos os feature_sets para uma máquina.
    Usa TSS robusto — ignora folds com treino insuficiente.

    Parâmetros
    ----------
    serie_d_clean : DataFrame já com features e sem NaN
    feature_sets  : lista de tuplos de features (igual ao resto do pipeline)
    n_splits      : número de folds TSS
    min_train_events : mínimo de avarias no fold de treino para o usar

    Retorna
    -------
    DataFrame com colunas: model, features, tss_mae_mean, tss_mae_std
    """
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error

    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for feats in feature_sets:
        feats = list(feats)

        for model_name, fit_fn, pred_fn in [
            ("ZIP",  fit_zip_model,  zip_predict),
            ("ZINB", fit_zinb_model, zinb_predict),
        ]:
            mae_scores = []

            for train_idx, test_idx in tscv.split(serie_d_clean):
                train = serie_d_clean.iloc[train_idx]
                test  = serie_d_clean.iloc[test_idx]

                # Filtro: treino com eventos suficientes
                if train["n_avarias"].sum() < min_train_events:
                    continue
                if len(test) == 0:
                    continue

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model  = fit_fn(train, feats)
                        y_pred = pred_fn(model, test, feats)

                    y_true = test["n_avarias"].values
                    if hasattr(y_pred, "values"):
                        y_pred = y_pred.values

                    mae = mean_absolute_error(y_true, np.maximum(y_pred, 0))

                    if np.isfinite(mae) and mae < 1e6:
                        mae_scores.append(mae)

                except Exception:
                    continue

            if mae_scores:
                results.append({
                    "model":        model_name,
                    "features":     feats,
                    "tss_mae_mean": np.mean(mae_scores),
                    "tss_mae_std":  np.std(mae_scores),
                    "n_folds_validos": len(mae_scores),
                })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────
# PIPELINE COMPLETO — corre ZIP/ZINB para todas as máquinas
# ─────────────────────────────────────────────────────────────

def run_zip_zinb_all_machines(df_clear, df_results_existente,
                               get_machine_df_fn,
                               build_daily_series_fn,
                               create_time_features_fn,
                               global_start, global_end,
                               threshold_events=20,
                               feature_pool=None,
                               max_features=3,
                               n_splits=5,
                               mch_list=None):
    """
    Corre ZIP e ZINB para todas as máquinas elegíveis e junta os resultados
    ao DataFrame existente (df_results_existente).

    Uso no notebook:
    ----------------
    from scripts.zip_zinb_models import run_zip_zinb_all_machines

    df_results_final = run_zip_zinb_all_machines(
        df_clear, df_results_completo,
        get_machine_df, build_daily_series, create_time_features,
        global_start, global_end,
    )
    """
    from itertools import combinations
    from tqdm import tqdm

    if feature_pool is None:
        feature_pool = ["lag1", "lag2", "lag3",
                        "ma3", "ma7", "ma30",
                        "trend_7", "trend_15", "trend_30"]

    feature_sets = list(combinations(feature_pool, max_features))

    if mch_list is None:
        # Máquinas elegíveis (>= threshold_events)
        contagens = (df_clear.groupby("MCH_CODE")["MCH_CODE"]
                    .count()
                    .reset_index(name="n_eventos"))
        maquinas = contagens[contagens["n_eventos"] >= threshold_events]["MCH_CODE"].tolist()
    else:
        maquinas = mch_list
    

    todos_resultados = []

    for mch in tqdm(maquinas, desc="ZIP/ZINB por máquina"):
        mdf   = get_machine_df_fn(df_clear, mch)
        serie = build_daily_series_fn(mdf, global_start, global_end)
        serie = create_time_features_fn(serie).dropna().copy()

        # Verificação mínima de viabilidade
        if serie["n_avarias"].sum() < threshold_events:
            continue

        df_res = evaluate_zip_zinb_for_machine(
            serie, feature_sets, n_splits=n_splits)

        if not df_res.empty:
            df_res["machine"] = mch
            todos_resultados.append(df_res)

    if not todos_resultados:
        print("Nenhum resultado ZIP/ZINB gerado.")
        return df_results_existente

    df_zip_zinb = pd.concat(todos_resultados, ignore_index=True)

    # Garantir colunas consistentes com df_results_existente
    for col in ["tss_mae_std"]:
        if col not in df_results_existente.columns:
            df_results_existente[col] = np.nan

    df_final = pd.concat(
        [df_results_existente, df_zip_zinb[
            ["machine", "model", "features",
             "tss_mae_mean", "tss_mae_std"]]],
        ignore_index=True,
    )

    print(f"\n✓ ZIP/ZINB concluídos.")
    print(f"  Linhas adicionadas : {len(df_zip_zinb)}")
    print(f"  Total no DataFrame : {len(df_final)}")
    print("\nDistribuição por modelo:")
    print(df_final.groupby("model")["machine"].nunique())

    return df_final