import sys
import os
import pandas as pd
import warnings
from datetime import datetime

# Ignorar warnings de overflow do statsmodels (comuns, mas inofensivos)
warnings.filterwarnings("ignore")

# Configurações de caminho
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'src'))

DATA_PATH = os.path.join(BASE_DIR, 'data', 'Work_Orders_MIP.xlsx')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Imports das suas funções
from src.loader import get_cleaned_data
from src.modelling import (
    evaluate_all_machines, get_machine_df, build_daily_series, create_time_features,
    fit_poisson_model, poisson_predict,
    fit_negative_binomial, nb_predict,
    fit_random_forest, rf_predict,
    fit_svm_rbf, svm_predict, 
)
from src.zip_zinb_models import run_zip_zinb_all_machines, fit_zinb_model, zinb_predict, fit_zip_model, zip_predict
from src.ranking_utils import build_risk_ranking

def main():
    print("🚀 Iniciando processamento completo...")
    print("⏳ Isso pode levar várias horas. Aguarde.")

    # 1. Carregar dados
    df = get_cleaned_data(DATA_PATH)
    print(f"📊 Dados carregados: {len(df)} linhas.")

    # 2. Etapa 1: evaluate_all_machines
    print("⏳ Executando evaluate_all_machines...")
    df_results_completo = evaluate_all_machines(df)

    # 3. Etapa 2: ZIP/ZINB
    print("⏳ Executando ZIP/ZINB...")
    global_start = df["REG_DATE_DT"].min().normalize()
    global_end   = df["REG_DATE_DT"].max().normalize()

    df_results_final = run_zip_zinb_all_machines(
        df_clear             = df,
        df_results_existente = df_results_completo,
        get_machine_df_fn    = get_machine_df,
        build_daily_series_fn    = build_daily_series,
        create_time_features_fn  = create_time_features,
        global_start = global_start,
        global_end   = global_end,
    )

    # 4. Gerar ranking
    # 4.1. Melhor resultado por máquina e modelo (menor MAE médio)
    df_best = (df_results_final
            .sort_values("tss_mae_mean")
            .groupby(["machine", "model"], as_index=False)
            .first())

    # 4.2. Critério combinado: Score = MAE_mean + 0.5 * MAE_std
    #    (penaliza instabilidade temporal — ajusta o peso 0.5 se quiseres)
    df_best["cv_mae"] = df_best["tss_mae_std"] / df_best["tss_mae_mean"]
    df_best["score"]  = df_best["tss_mae_mean"] + 0.5 * df_best["tss_mae_std"]

    # 4.3. Modelo vencedor por máquina
    df_winner = (df_best
                .sort_values("score")
                .groupby("machine", as_index=False)
                .first())

    # 4.4. Estatísticas globais por modelo
    df_global = (df_best
                .groupby("model")
                .agg(
                    mae_medio        = ("tss_mae_mean", "mean"),
                    mae_mediana      = ("tss_mae_mean", "median"),
                    mae_std_medio    = ("tss_mae_std",  "mean"),
                    cv_medio         = ("cv_mae",        "mean"),
                    score_medio      = ("score",         "mean"),
                )
                .round(5)
                .sort_values("score_medio"))
    
    # ── Filtro de validade para ZINB ──

    # Limiar: ZINB só é aceite se MAE < 3× mediana do Poisson na mesma máquina
    # (critério conservador que elimina explosões sem excluir casos legítimos)
    pivot = df_best.pivot(index="machine", columns="model", values="tss_mae_mean").reset_index()

    # Calcular limiar por máquina (3x o MAE do Poisson)
    pivot["limiar_zinb"] = pivot["Poisson"] * 3

    # Identificar máquinas onde ZINB é válido
    pivot["zinb_valido"] = pivot["ZINB"] < pivot["limiar_zinb"]

    # Criar df_best_filtrado onde ZINB inválido é substituído pelo seu MAE real
    maquinas_zinb_invalido = pivot[~pivot["zinb_valido"]]["machine"].tolist()

    df_best_filtrado = df_best.copy()
    df_best_filtrado["zinb_invalido"] = (
        (df_best_filtrado["model"] == "ZINB") &
        (df_best_filtrado["machine"].isin(maquinas_zinb_invalido))
    )

    # Para seleção do vencedor: excluir ZINB inválido
    df_para_winner = df_best_filtrado[~df_best_filtrado["zinb_invalido"]].copy()

    df_winner_v2 = (df_para_winner
                    .sort_values("score")
                    .groupby("machine", as_index=False)
                    .first())

    # Tabela global recalculada — separar ZINB válido de inválido
    df_zinb_valido   = df_best[
        (df_best["model"] == "ZINB") &
        (~df_best["machine"].isin(maquinas_zinb_invalido))
    ]
    df_zinb_invalido = df_best[
        (df_best["model"] == "ZINB") &
        (df_best["machine"].isin(maquinas_zinb_invalido))
    ]

    df_outros = df_best[df_best["model"] != "ZINB"]
    df_zinb_valido_v2 = df_zinb_valido.copy()
    df_zinb_valido_v2["model"] = "ZINB (válido)"

    df_global_v2 = (pd.concat([df_outros, df_zinb_valido_v2])
                    .groupby("model")
                    .agg(
                        n_maquinas    = ("machine",       "count"),
                        mae_medio     = ("tss_mae_mean",  "mean"),
                        mae_mediana   = ("tss_mae_mean",  "median"),
                        mae_std_medio = ("tss_mae_std",   "mean"),
                        cv_medio      = ("cv_mae",        "mean"),
                        score_medio   = ("score",         "mean"),
                    )
                    .round(5)
                    .sort_values("score_medio"))

    print("\n=== Desempenho global por modelo ===")
    print(df_global_v2.to_string())

    df_results_testar = df_winner_v2.copy()

    # Filtrar valores explosivos (caso com Poisson)
    results_clean = df_results_testar[df_results_testar['tss_mae_mean'] < 1e10].copy()

    # Calcular Coeficiente de Variação
    results_clean['cv'] = results_clean['tss_mae_std'] / results_clean['tss_mae_mean']

    # Calcular score composto
    results_clean['score'] = results_clean['tss_mae_mean']  + 0.5 *results_clean['tss_mae_std']


    def select_best_per_machine(df, metric='tss_mae_mean', tiebreaker='cv'):
        """
        Seleciona o melhor modelo por máquina com critério de desempate.
        """
        # Ordenar: primeiro pela máquina, depois pela métrica principal, depois desempate
        sorted_df = df.sort_values(
            ['machine', metric, tiebreaker],
            ascending=[True, True, True]
        )
        # Pegar o primeiro registro de cada máquina
        best = sorted_df.groupby('machine', as_index=False).first()
        
        return best


    # Selecionar melhor por máquina para cada critério
    best_by_criterion = {
        # Critério 1: Menor MAE
        # 'menor_mae': select_best_per_machine(results_clean, metric='tss_mae_mean', tiebreaker='cv'),
        # Critério 2: Menor CV (mais estável)
        # 'menor_cv': select_best_per_machine(results_clean, metric='cv', tiebreaker='tss_mae_mean'),
        # Critério 3: Menor score (MAE + penalidade por instabilidade)
        'menor_score': select_best_per_machine(results_clean, metric='score', tiebreaker='cv'),
    }

    # Verificar se cada máquina tem exatamente um registro
    for name, df_ in best_by_criterion.items():
        n_machines = df_['machine'].nunique()
        n_rows = len(df_)
        print(f"{name}: {n_rows} registros para {n_machines} máquinas")
        if n_rows != n_machines:
            print(f"ATENÇÃO: {n_rows - n_machines} máquinas duplicadas!")

    best_models_df = best_by_criterion["menor_score"].copy()
    best_models_df = best_models_df.rename(columns={'features': 'best_features'})

    FIT_PREDICT = {
        "Poisson":          (fit_poisson_model,     poisson_predict),
        "NegativeBinomial": (fit_negative_binomial, nb_predict),
        "ZIP":              (fit_zip_model,         zip_predict),
        "ZINB":             (fit_zinb_model,        zinb_predict),
        "RandomForest":     (fit_random_forest,     rf_predict),
        "SVM_RBF":          (fit_svm_rbf,           svm_predict),
    }

    df_clear = df[(df["EVENT_TAG"]=='avaria') & (~df["SOBREPOSICAO_REPARO"])].copy()

    # Gerar ranking com lambda_hat
    df_ranking = build_risk_ranking(
        df_results_final= df_results_final,
        df_clear=df_clear,
        fit_predict_map=FIT_PREDICT,
        get_machine_df_fn=get_machine_df,
        build_daily_series_fn=build_daily_series,
        create_time_features_fn=create_time_features,
        global_start=global_start,
        global_end=global_end,
        zinb_invalid_machines = df_zinb_invalido,
        n_splits=5,
        max_lambda=10.0
    )

    df_ranking = df_ranking.drop(columns=['winner_model']).copy()
    df_ranking
    df_ranking['cv'] = df_ranking['mae_std'] / df_ranking['mae_mean']
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_ranking['data_geracao'] = timestamp
    df_ranking = df_ranking.rename(columns={
        'mae_mean': 'tss_mae_mean',
        'mae_std': 'tss_mae_std'
    })
    colunas_ordenadas = [
        'rank',
        'machine',
        'model',
        'best_features',
        'tss_mae_mean',
        'tss_mae_std',
        'cv',
        'score',
        'lambda_hat',
        'fallback',
        'data_geracao'
    ]
    df_ranking = df_ranking[colunas_ordenadas]

    # Salvar o resultado
    output_csv_time = os.path.join(OUTPUT_DIR, f'ranking_risco_{timestamp}.csv')
    output_csv = os.path.join(OUTPUT_DIR, 'ranking_risco.csv')
    df_ranking.to_csv(output_csv_time, index=False)
    df_ranking.to_csv(output_csv, index=False)

    print(f"✅ PROCESSAMENTO CONCLUÍDO!")
    print(f"📁 Resultado salvo em: {output_csv}")

if __name__ == "__main__":
    main()