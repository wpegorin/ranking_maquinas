import pandas as pd
import numpy as np
import warnings
import statsmodels.api as sm
from itertools import combinations
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit

from tqdm import tqdm
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def get_machine_df(df, machine_code):
    
    machine_df = df[df["MCH_CODE"] == machine_code].copy()
    machine_df = machine_df.sort_values("REG_DATE_DT")
    
    return machine_df

def temporal_train_test_split(df, train_size=0.8):

    split_index = int(len(df) * train_size)

    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()

    return train, test

def build_daily_series(machine_df, global_start=None, global_end=None):

    df = machine_df.copy()

    df["REG_DATE_DT"] = pd.to_datetime(df["REG_DATE_DT"])

    # definir intervalo global
    if global_start is None:
        global_start = df["REG_DATE_DT"].min().normalize()
    else:
        global_start = pd.to_datetime(global_start).normalize()
        
    if global_end is None:
        global_end = df["REG_DATE_DT"].max().normalize()
    else:
        global_end = pd.to_datetime(global_end).normalize()
    
    df = df.set_index("REG_DATE_DT")

    counts = df["MCH_CODE"].resample("D").count()

    full_range = pd.date_range(start=global_start, end=global_end, freq="D")

    serie_d = counts.reindex(full_range, fill_value=0)

    serie_d = serie_d.reset_index()
    serie_d.columns = ["REG_DATE_DT", "n_avarias"]
    
    return serie_d

def create_time_features(df):

    df = df.copy()

    df["lag1"] = df["n_avarias"].shift(1)
    df["lag2"] = df["n_avarias"].shift(2)
    df["lag3"] = df["n_avarias"].shift(3)

    # Médias móveis (períodos diferentes para capturar diferentes padrões)
    df['ma3'] = df['n_avarias'].shift(1).rolling(window=3, min_periods=1).mean()
    df['ma7'] = df['n_avarias'].shift(1).rolling(window=7, min_periods=1).mean()  # Semanal
    # df['ma14'] = df['n_avarias'].shift(1).rolling(window=14, min_periods=1).mean()  # Quinzenal
    df['ma30'] = df['n_avarias'].shift(1).rolling(window=30, min_periods=1).mean()  # Mensal


    # Tendência
    # df["trend"] = np.arange(len(df))
    df['trend_7'] = df['n_avarias'].shift(1).rolling(window=7).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 7 else 0
    )  # Inclinação nos últimos 7 dias

    df['trend_15'] = df['n_avarias'].shift(1).rolling(window=15).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 15 else 0
    )  # Inclinação nos últimos 15 dias

    df['trend_30'] = df['n_avarias'].shift(1).rolling(window=30).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 30 else 0
    )  # Inclinação nos últimos 30 dias

    df = df.dropna()

    return df

def generate_feature_sets(feature_pool, max_features=3):

    feature_sets = []

    for k in range(1, max_features+1):
        feature_sets += list(combinations(feature_pool, k))

    return feature_sets

def fit_poisson_model(df, features):

    X = sm.add_constant(df[features], has_constant='add')
    y = df["n_avarias"]

    model = sm.GLM(y, X, family=sm.families.Poisson())
    result = model.fit()

    return result

def poisson_predict(model, df_test, features):

    X_test = sm.add_constant(df_test[features], has_constant='add')
    
    y_pred = model.predict(X_test)

    return y_pred

def fit_negative_binomial(df, features, alpha=None):
    
    X = sm.add_constant(df[features], has_constant='add')
    y = df["n_avarias"]
    
    if alpha is None:
        poisson_model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        mu = poisson_model.predict(X)
        
        # Estimador simples de alpha baseado nos resíduos
        pearson_chi2 = np.sum((y - mu)**2 / mu)
        alpha = max(pearson_chi2 / (len(y) - len(features) - 1), 1e-6)
    
    nb_family = sm.families.NegativeBinomial(alpha=alpha)

    model = sm.GLM(y, X, family=nb_family)
    result = model.fit()
    
    return result

def nb_predict(model, df_test, features):
    
    X_test = sm.add_constant(df_test[features], has_constant='add')
    
    y_pred = model.predict(X_test)
    
    return y_pred

def fit_random_forest(df, features):

    X = df[features]
    y = df["n_avarias"]

    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )

    rf.fit(X, y)

    return rf

def rf_predict(model, df_test, features):

    X_test = df_test[features]

    y_pred = model.predict(X_test)

    return y_pred

def fit_svm(df, features, kernel='rbf', C=1.0, epsilon=0.1, scale_features=True):
    """
    Parameters:
    -----------
    kernel : string, optional (default='rbf')
        Especifica o tipo de kernel ('linear', 'poly', 'rbf', 'sigmoid')
    C : float, optional (default=1.0)
        Parâmetro de regularização. Força de regularização é inversamente proporcional a C
    epsilon : float, optional (default=0.1)
        Épsilon na função de perda epsilon-SVR. Especifica a margem onde nenhum erro é penalizado
    scale_features : bool, optional (default=True)
        Se True, padroniza as features (recomendado para SVM)
    """
    
    X = df[features].copy()
    y = df["n_avarias"].copy()
    
    if scale_features:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        svm_model = {
            'scaler': scaler,
            'features': features
        }
    else:
        X_scaled = X
        svm_model = {
            'scaler': None,
            'features': features
        }
    
    svr = SVR(
        kernel=kernel,
        C=C,
        epsilon=epsilon,
        cache_size=1000
    )
    
    svr.fit(X_scaled, y)
    
    svm_model['model'] = svr
    
    return svm_model

def svm_predict(model, df_test, features):
    X_test = df_test[features].copy()
    
    if model['scaler'] is not None:
        X_test_scaled = model['scaler'].transform(X_test)
    else:
        X_test_scaled = X_test
    
    y_pred = model['model'].predict(X_test_scaled)
    
    y_pred = np.maximum(y_pred, 0)
    
    return y_pred

def fit_svm_rbf(df, features, gamma='scale', C=1.0, epsilon=0.1):
    """SVM com kernel RBF (mais comum)"""
    X = df[features]
    y = df["n_avarias"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    svr = SVR(
        kernel='rbf',
        gamma=gamma,
        C=C,
        epsilon=epsilon
    )
    
    svr.fit(X_scaled, y)
    
    return {
        'model': svr,
        'scaler': scaler,
        'features': features
    }

def evaluate_predictions(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {
        "MAE": mae,
        "RMSE": rmse
    }

def expanding_window_validation(df, features, fit_func, predict_func, initial_train=90, step=15):
    """
    Validação com janela crescente (expanding window) e stride fixo.
    Treina com todos os dados até o momento, testa a cada step dias.
    """
    errors = []

    for i in range(initial_train, len(df), step):

        train = df.iloc[:i]
        test = df.iloc[i:i+1]

        model = fit_func(train, features)

        pred = predict_func(model, test, features)

        y_true = test["n_avarias"].values[0]
        y_pred = pred.iloc[0] if hasattr(pred, 'iloc') else pred[0]

        error = abs(y_true - y_pred)

        errors.append(error)

    return np.mean(errors)

def evaluate_with_tss(df, features, fit_func, predict_func, n_splits=5):
    """
    Avalia modelo usando Time Series Split.
    
    Parâmetros:
    -----------
    df : DataFrame
        Dados completos (já ordenados temporalmente)
    features : list
        Lista de features a serem usadas
    fit_func : function
        Função que recebe (df, features) e retorna modelo treinado
    predict_func : function
        Função que recebe (model, df_test, features) e retorna previsões
    n_splits : int
        Número de splits para validação
    
    Retorna:
    --------
    dict : Dicionário com MAE por fold, MAE médio e desvio padrão
    """
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    mae_scores = []
    fold_predictions = []
    fold_actuals = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        # Divide dados respeitando ordem temporal
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        
        # Usa suas funções existentes
        model = fit_func(train, features)
        y_pred = predict_func(model, test, features)
        
        # Garante que y_pred tem formato correto (pode ser Series ou array)
        if hasattr(y_pred, 'values'):
            y_pred = y_pred.values
        if hasattr(test['n_avarias'], 'values'):
            y_true = test['n_avarias'].values
        else:
            y_true = test['n_avarias']
        
        # Calcula MAE
        mae = mean_absolute_error(y_true, y_pred)
        mae_scores.append(mae)
        
        fold_predictions.append(y_pred)
        fold_actuals.append(y_true)
        
        # print(f"Fold {fold+1}/{n_splits} - {features} - MAE: {mae:.4f} | "
            #   f"Treino: {len(train)} | Teste: {len(test)}")
    
    return {
        'mae_scores': mae_scores,
        'mae_mean': np.mean(mae_scores),
        'mae_std': np.std(mae_scores),
        'predictions': fold_predictions,
        'actuals': fold_actuals
    }


def evaluate_models_for_machine(serie_d, feature_sets=None, last_year_failures_threshold=20, strategy='auto'):
    
    # Calcular falhas do período
    last_year_failures = serie_d['n_avarias'].sum()
    
    # Verificar se deve pular modelos de contagem
    skip_count_models = last_year_failures < last_year_failures_threshold and strategy != 'all'

    if feature_sets is None:
        feature_pool = ["lag1", "lag2", "lag3", 'ma3', 'ma7', 'ma30', 'trend_7', 'trend_15', 'trend_30']
        feature_sets = generate_feature_sets(feature_pool, max_features=3)

    results = []

    # for features in feature_sets:
    for features in tqdm(feature_sets, desc="  Combinando features", leave=False, unit="comb"):

        features = list(features)

        if not skip_count_models:
            # Poisson
            try:
                # score = rolling_forecast_origin(serie_d, features, fit_poisson_model, poisson_predict)
                score = evaluate_with_tss(serie_d, features, fit_poisson_model, poisson_predict)
                results.append({
                    "model": "Poisson", 
                    "features": features, 
                    "tss_mae_mean": score['mae_mean'], 
                    "tss_mae_std": score['mae_std']
                    })

            except:
                pass

            # Binomial Negativa
            try:
                # score = rolling_forecast_origin(serie_d, features, fit_negative_binomial, nb_predict)
                score = evaluate_with_tss(serie_d, features, fit_negative_binomial, nb_predict)
                results.append({
                    "model": "NegativeBinomial", 
                    "features": features, 
                    "tss_mae_mean": score['mae_mean'], 
                    "tss_mae_std": score['mae_std']
                    })
                
            except:
                pass

            # Random Forest
            try:

                # score = rolling_forecast_origin(serie_d, features, fit_random_forest, rf_predict)
                score = evaluate_with_tss(serie_d, features, fit_random_forest, rf_predict)
                results.append({
                    "model": "RandomForest", 
                    "features": features, 
                    "tss_mae_mean": score['mae_mean'], 
                    "tss_mae_std": score['mae_std']
                    })

            except:
                pass

            # SVM (usando kernel RBF como padrão)
            try:
                def fit_svm_wrapper(df, features):
                    return fit_svm_rbf(df, features, gamma='scale', C=1.0, epsilon=0.1)
                
                # score = rolling_forecast_origin(serie_d, features, fit_svm_wrapper, svm_predict)
                score = evaluate_with_tss(serie_d, features, fit_svm_wrapper, svm_predict)
                results.append({
                    "model": "SVM_RBF", 
                    "features": features, 
                    "tss_mae_mean": score['mae_mean'], 
                    "tss_mae_std": score['mae_std']
                    })
                
            except:
                pass

    return pd.DataFrame(results)


def process_machine(df, machine_code, feature_sets, global_start, global_end):

    machine_df = get_machine_df(df, machine_code)

    serie_d = build_daily_series(machine_df, global_start, global_end)

    serie_d = create_time_features(serie_d)

    # Para avaliação de modelos, precisamos remover NaNs iniciais das features
    serie_d_clean = serie_d.dropna().copy()

    ranking = evaluate_models_for_machine(
        serie_d_clean,
        feature_sets
    )

    ranking["machine"] = machine_code

    return ranking

def evaluate_all_machines(df):

    machines = df["MCH_CODE"].unique()

    global_start = df["REG_DATE_DT"].min()
    global_end   = df["REG_DATE_DT"].max()

    feature_pool = ["lag1", "lag2", "lag3", 'ma3', 'ma7', 'ma30', 'trend_7', 'trend_15', 'trend_30']

    feature_sets = generate_feature_sets(feature_pool, 3)

    results = Parallel(n_jobs=-1)(
        delayed(process_machine)(df, m, feature_sets, global_start, global_end)
        for m in tqdm(machines, desc="Processando máquinas")
    )

    return pd.concat(results)
