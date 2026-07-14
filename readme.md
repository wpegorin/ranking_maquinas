# Ranking de Risco de Avarias em Máquinas de Injeção de Polímeros

Este projeto implementa uma pipeline automatizada para estimar o risco de avarias em máquinas de injeção de polímeros (MIPs) com base exclusivamente no histórico de ordens de trabalho (OT). O modelo seleciona automaticamente o melhor modelo de contagem (Poisson, Binomial Negativa, ZIP, ZINB) e modelos de Machine Learning (Random Forest, SVR) por máquina, validando temporalmente o desempenho.

---

## 📁 Estrutura do Projeto

```
ranking_maquinas/
│
├── data/
│   └── Work_Orders_MIP.xlsx    # Dados brutos (formato descrito abaixo)
│
├── src/
│   ├── __init__.py
│   ├── loader.py                    			# Carregamento e limpeza dos dados
│   ├── modelling.py                 			# Modelos base (Poisson, BN, RF, SVR)
│   ├── zip_zinb_models.py           			# Modelos inflacionados de zeros (ZIP/ZINB)
│   └── ranking_utils.py             			# Funções para construção do ranking
│
├── outputs/                         			# Resultados gerados (criado automaticamente)
│   ├── ranking_risco_YYYYMMDD_HHMMSS.csv      	# Ranking final (com timestamp)
│   └── ranking_risco.csv            			# Ranking final de risco por máquina
│
├── run_ranking.py                    			# Script principal
├── requirements.txt                  			# Dependências do projeto
└── README.md                         			# Este ficheiro
```
## 📂 Formato dos dados de entrada

O script espera um ficheiro Excel na pasta `data/` com o nome `Work_Orders_MIP.xlsx`, contendo uma worksheet chamada `work_orders` com as seguintes colunas:

| Coluna | Descrição |
|--------|-----------|
| `MCH_CODE` | Identificador da máquina (ex.: MCH0000) |
| `WORK_TYPE_DESC` | Tipo de ordem de trabalho (ex.: manutenção corretiva, setup) |
| `REG_DATE` | Data/hora de registo da ordem de trabalho |
| `ERR_DESCR` | Descrição textual da avaria ou intervenção |
| `REAL_S_DATE` | Data/hora real de início da intervenção |
| `REAL_F_DATE` | Data/hora real de fim da intervenção |
| `WO_STATUS_ID` | Estado da ordem de trabalho (ex.: FINISHED, CANCELED) |

> Nota: Os dados fornecidos neste repositório são apenas um exemplo da estrutura esperada. Os dados reais da OLI não são partilhados por razões de confidencialidade.

---

## 🚀 Instalação

### 1. Clonar ou copiar o repositório

```bash
git clone <url-do-repositorio>  # ou apenas copie a pasta
cd ranking_maquinas
```

### 2. Criar e ativar o ambiente virtual

**Windows (PowerShell):**
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (Prompt de Comando):**
```bash
python -m venv .venv
.\.venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

---

## 🏃 Como executar

Com o ambiente virtual ativado, execute:

```bash
python run_ranking.py
```

O script irá:

1. Carregar e limpar os dados de ordens de trabalho.
2. Avaliar todos os modelos (Poisson, BN, ZIP, ZINB, RF, SVR) para cada máquina.
3. Selecionar o melhor modelo por máquina segundo o critério **Score** (MAE + 0.5 × σ_MAE).
4. Estimar o risco \(\hat{\lambda}_i\) para cada máquina.
5. Gerar o ficheiro `outputs/ranking_risco.csv` e 'outputs/ranking_risco_YYYYMMDD_HHMMSS.csv' com o ranking final.

---

## 📊 Resultados

Após a execução, os ficheiros `outputs/ranking_risco.csv` e `outputs/ranking_risco_YYYYMMDD_HHMMSS.csv` contém:

| Coluna | Descrição |
|--------|-----------|
| `rank` | Posição no ranking (1 = maior risco) |
| `machine` | Identificador da máquina (ex.: MCH0000) |
| `model` | Modelo selecionado para essa máquina |
| `best_features` | Subconjunto de variáveis temporais selecionadas |
| `tss_mae_mean` | MAE médio na validação temporal |
| `tss_mae_std` | Desvio-padrão do MAE entre *folds* |
| `cv` | Coeficiente de variação (MAE_std / MAE_mean) |
| `score` | Score composto (MAE + 0.5 × σ_MAE) |
| `lambda_hat` | Intensidade esperada de avarias (\(\hat{\lambda}_i\)) |
| `fallback` | `True` se a estimativa veio de Poisson (fallback) |
| `data_geracao` | Data e hora da execução do script |

---

## ⚙️ Dependências

As principais bibliotecas utilizadas são:

- [pandas](https://pandas.pydata.org/) – Manipulação de dados
- [numpy](https://numpy.org/) – Cálculos numéricos
- [statsmodels](https://www.statsmodels.org/) – Modelos de contagem (Poisson, BN)
- [scikit-learn](https://scikit-learn.org/) – Random Forest, SVR
- [scipy](https://scipy.org/) – Suporte estatístico
- [tqdm](https://tqdm.github.io/) – Barras de progresso
- [openpyxl](https://openpyxl.readthedocs.io/) – Leitura de ficheiros Excel

---

## ⚠️ Notas importantes

- O script foi desenhado para dados com um mínimo de **20 avarias por máquina** no período de 711 dias.
- O tempo de execução completo pode exceder **várias horas**, dependendo do número de máquinas e da especificação do computador.
- O modelo ZINB é **numericamente instável** em algumas máquinas; nestes casos, é automaticamente substituído por Poisson (mecanismo de *fallback*).

---

## 👤 Autor

**Willian Pegorin**  
Mestrado em Matemática e Aplicações – Universidade de Aveiro  
Orientador: Prof. Nelson Vieira | Co-orientador: Prof. Miguel Zilhão  
Supervisores na Empresa:  Paulo Ribeiro e Renata Loio

---

## 📬 Contacto

Para questões ou sugestões, contacte: `wpegorin@ua.pt` ou `wpegorin@gmail.com`
```
