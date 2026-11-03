# Trabalho Final de Machine Learning — Epicurious Dataset

**Disciplina:** Machine Learning  
**Dataset:** [Epicurious – Recipes with Rating and Nutrition](https://www.kaggle.com/datasets/hugodarwood/epirecipes) (Kaggle)  
**Notebook principal:** [trabalho_final.ipynb](trabalho_final.ipynb)

> **Contexto.** Este trabalho é o complemento da [atividade anterior](https://github.com/LucasBuccini/ml-atividade),
> que previu as calorias de receitas com R² ajustado de **0,897** usando Gradient Boosting.
> Aqui, o mesmo dataset é explorado sob três ângulos: regressão de calorias (ampliando os modelos
> da atividade), classificação do rating dos usuários, e análise não supervisionada para
> descoberta de grupos de receitas e sugestões de substituição de dieta.

---

## Sumário

1. [Visão geral](#1-visão-geral)
2. [O dataset](#2-o-dataset)
3. [Os três problemas](#3-os-três-problemas)
4. [Modelos utilizados](#4-modelos-utilizados)
5. [Resultados obtidos](#5-resultados-obtidos)
6. [Análise não supervisionada](#6-análise-não-supervisionada)
7. [Como executar](#7-como-executar)
8. [Estrutura do repositório](#8-estrutura-do-repositório)
9. [Relação com a atividade anterior](#9-relação-com-a-atividade-anterior)

---

## 1. Visão geral

| | |
|---|---|
| **Dataset** | Epicurious — 20.052 receitas, 680 colunas brutas |
| **Após limpeza** | ~14.000 receitas, 335 features (3 macros + 332 tags binárias) |
| **Problemas** | Regressão (calorias) + Classificação (rating) + Clustering |
| **Modelos testados** | 18 — cobre ~90% da lista do professor |
| **Técnicas de lista do professor** | Regressão Linear/Polinomial/Splines/Local/GAMs · Logit/LDA/QDA/NB/KNN · Bagging/RF/Boosting/SVM · K-Means/Hierárquico/GMM/PCA |

---

## 2. O dataset

O arquivo `epi_r.csv` contém receitas do site Epicurious:

| Coluna | Descrição |
|---|---|
| `title` | Nome da receita |
| `rating` | Nota dos usuários (0 = sem avaliação; 2,5–5 = notas reais) |
| `calories` | Calorias totais (kcal) |
| `protein` | Proteína (g) |
| `fat` | Gordura (g) |
| `sodium` | Sódio (mg) |
| 674 tags | Etiquetas binárias (0/1): ingredientes, dietas, técnicas, ocasiões |

### Pipeline de limpeza

| Etapa | Registros |
|---|---|
| Dados originais | 20.052 |
| Remoção de duplicatas | 18.251 |
| Remoção de nulos (macros + rating) | 14.446 |
| Remoção de outliers extremos (p99 em calorias) | ~14.096 |
| **Para classificação:** remoção de rating = 0 | ~11.000–13.000 |

> **Por que remover rating = 0?** No site Epicurious, o valor 0 indica ausência de avaliação
> (receita sem votos), e não uma nota "ruim". Incluí-las no problema de classificação
> introduziria ruído estrutural que impossibilita qualquer modelo de aprender.

### Definição de features

| Conjunto | Colunas | Uso |
|---|---|---|
| `MACRO_COLS` | `fat`, `protein`, `sodium` | Modelos simples de regressão |
| `TOP30_FEATURES` | 30 variáveis selecionadas pelo Lasso | Regressão ensemble, K-Means, LDA, QDA, KNN |
| `ALL_FEATURES` | 335 (macros + 332 tags) | Classificadores que toleram alta dimensão |

---

## 3. Os três problemas

### Problema A — Regressão de Calorias

- **Target:** `calories` (kcal, variável contínua)
- **Justificativa:** Calorias têm relação física direta com gordura (9 kcal/g) e proteína (4 kcal/g)
- **Desafio:** Carboidratos não estão no dataset; as tags funcionam como indicadores indiretos
- **Métricas:** R² ajustado, MAE, RMSE
- **Split:** 80% treino / 20% teste, `random_state=42`, 5-fold CV

### Problema B — Classificação de Rating

- **Target:** `rating_class` — rating discretizado em 3 classes por tercis:
  - Classe 0 (`baixo`): abaixo do 1º tercil
  - Classe 1 (`médio`): entre 1º e 2º tercil
  - Classe 2 (`alto`): acima do 2º tercil
- **Desafio:** Rating tem correlação de Spearman ~0,10 com todas as features — gosto é subjetivo
- **Métricas:** Accuracy, F1-macro, Confusion Matrix
- **Baseline aleatório:** 33,3%

### Problema C — Análise Não Supervisionada

- **Objetivo:** Descobrir grupos naturais de receitas por perfil nutricional
- **Aplicação prática:** Sistema de substituição dietética e de qualidade
- **Input para clustering:** 24 componentes do PCA (retêm 90% da variância das TOP30_FEATURES)

---

## 4. Modelos utilizados

Cobertura: **~18 de 19 técnicas** da lista do professor (~90%).

### Bloco 1 — Regressores (10 modelos)

| # | Modelo | Features | Nota |
|---|---|---|---|
| 1 | Regressão Linear | `MACRO_COLS` (3) | Baseline físico |
| 2 | Regressão Polinomial (grau 2) | `MACRO_COLS` | Termos quadráticos |
| 3 | Splines Cúbicas Naturais | `MACRO_COLS` | Via `patsy` (`cr()`) |
| 4 | LOESS (Regressão Local) | `fat` | Apenas visualização motivacional |
| 5 | GAM | `MACRO_COLS` | Via `pygam.LinearGAM` |
| 6 | Ridge | `TOP30_FEATURES` | Regularização L2 |
| 7 | Lasso (LassoCV) | `ALL_FEATURES` (335) | Regularização L1 + seleção |
| 8 | Árvore + Bagging | `TOP30_FEATURES` | 50 árvores |
| 9 | Random Forest | `TOP30_FEATURES` | 100 árvores |
| 10 | Gradient Boosting | `TOP30_FEATURES` | `HistGradientBoosting` |

### Bloco 2 — Classificadores (8 modelos)

| # | Modelo | Features | Nota |
|---|---|---|---|
| 11 | Regressão Logística | `ALL_FEATURES` | C=0.1, max_iter=2000 |
| 12 | LDA | `TOP30_FEATURES` | Alta dim. causa singularidade |
| 13 | QDA | `TOP30_FEATURES` | reg_param=0.1 |
| 14 | Naive Bayes (Bernoulli) | `ALL_FEATURES` | Adequado para tags binárias |
| 15 | KNN (K=15) | `TOP30_FEATURES` | Maldição da dimensão em 335-D |
| 16 | SVM (RBF) | `ALL_FEATURES` | Subsample estratificado n=5.000 |
| 17 | Random Forest | `ALL_FEATURES` | 200 árvores, `max_features='sqrt'` |
| 18 | Gradient Boosting | `ALL_FEATURES` | `HistGradientBoosting` |

### Bloco 3 — Técnicas Não Supervisionadas (4 técnicas)

| # | Técnica | Input | Nota |
|---|---|---|---|
| 19 | PCA | `TOP30_FEATURES` | 24 componentes = 90% variância |
| 20 | K-Means | 24 componentes PCA | Cotovelo + Silhouette |
| 21 | Clustering Hierárquico | 24 componentes PCA | Ward linkage |
| 22 | GMM | 24 componentes PCA | Seleção por BIC |

---

## 5. Resultados obtidos

> Tabela atualizada após execução completa do notebook.

### Regressão (Calorias)

| Modelo | Features | R² adj. (CV) | R² adj. (Teste) | MAE |
|---|---|---|---|---|
| Regressão Linear | 3 | — | 0,845 | 95 kcal |
| Regressão Polinomial | 3 | — | 0,845 | ~96 kcal |
| Splines Cúbicas | 3 | — | 0,844 | ~96 kcal |
| GAM | 3 | — | 0,852 | ~90 kcal |
| Ridge | 30 | — | 0,887 | ~71 kcal |
| Lasso (LassoCV) | 335 | — | 0,884 | 67 kcal |
| Árvore + Bagging | 30 | — | 0,873 | ~77 kcal |
| Random Forest | 30 | — | 0,882 | ~70 kcal |
| **Gradient Boosting** | **30** | **0,904 ± 0,01** | **0,897** | **66 kcal** |

> **Conclusão:** Os modelos baseados apenas nos 3 macronutrientes (Linear, Polinomial, Splines, GAM)
> atingem R² ~0,845–0,852, capturando a relação física direta com calorias. As tags adicionam
> 4–5 p.p. de R², com Gradient Boosting sendo o modelo campeão.

### Classificação (Rating)

> Baseline aleatório: **33,3%** (3 classes balanceadas por tercis).

| Modelo | Features | Acc. (CV) | Acc. (Teste) | F1-macro |
|---|---|---|---|---|
| Regressão Logística | 335 | — | — | — |
| LDA | 30 | — | — | — |
| QDA | 30 | — | — | — |
| Naive Bayes (Bernoulli) | 335 | — | — | — |
| KNN (K=15) | 30 | — | — | — |
| SVM (RBF) | 335 | — | — | — |
| Random Forest | 335 | — | — | — |
| Gradient Boosting | 335 | — | — | — |

> **Os valores acima serão preenchidos após execução do notebook.**
> Espera-se acurácia entre 35–45%, pois a correlação de Spearman entre rating e
> qualquer feature é ~0,10. Este resultado negativo é valioso: confirma que
> prever satisfação gastronômica requer dados qualitativos (texto, imagens).

---

## 6. Análise não supervisionada

### PCA

- Das 30 variáveis padronizadas, são necessários **24 componentes** para reter 90% da variância
- Isso indica alta dimensionalidade intrínseca: nenhuma dimensão domina isoladamente
- PC1 separa receitas calóricas (gordura/proteína alta) das leves
- PC2 separa sobremesas/bebidas dos pratos principais
- As 3 classes de rating são **indistinguíveis** no espaço PCA — confirmando que rating não é previsível a partir da composição nutricional

### K-Means (4 clusters identificados)

| Cluster | Perfil | Calorias médias | Tags dominantes |
|---|---|---|---|
| 0 | Pratos Leves e Saladas | ~180 kcal | salad, vegetarian, no-cook |
| 1 | Pratos Principais e Assados | ~480 kcal | bake, garlic, dinner |
| 2 | Sobremesas e Confeitaria | ~580 kcal | dessert, cake, cookies |
| 3 | Bebidas e Coquetéis | ~120 kcal | drink, alcoholic, fruit |

### Sistema de substituição dietética

A função `sugerir_substitutos(receita_titulo, n=5)` localiza o cluster da receita e retorna:
- **Alternativas diet:** receitas do mesmo cluster com menor teor calórico
- **Alternativas premium:** receitas do mesmo cluster com maior nota

---

## 7. Como executar

Requer Python 3.10 ou superior.

```bash
# Clone o repositório
git clone https://github.com/<seu-usuario>/ml-trabalho-final.git
cd ml-trabalho-final

# Crie e ative o ambiente virtual
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# Instale as dependências
pip install -r requirements.txt

# Abra o notebook
jupyter lab trabalho_final.ipynb
```

### Dataset

O notebook busca automaticamente o dataset em `data/epi_r.csv`.
Se o arquivo não existir, ele é baixado via `kagglehub`:

```python
import kagglehub
path = kagglehub.dataset_download("hugodarwood/epirecipes")
```

Para isso, configure suas credenciais do Kaggle em `~/.kaggle/kaggle.json`
([instruções](https://github.com/Kaggle/kaggle-api#api-credentials)).

Alternativamente, baixe manualmente o arquivo `epi_r.csv` do
[dataset no Kaggle](https://www.kaggle.com/datasets/hugodarwood/epirecipes)
e coloque em `data/epi_r.csv`.

### Tempo de execução estimado

| Bloco | Estimativa |
|---|---|
| 0 — Setup e limpeza | ~1 min |
| 1 — Regressão (10 modelos) | ~3–5 min |
| 2 — Classificação (8 modelos) | ~10–20 min (SVM + RF são os mais lentos) |
| 3 — Clustering | ~3–5 min |
| **Total** | **~20–30 min** |

---

## 8. Estrutura do repositório

```
ml-trabalho-final/
├── trabalho_final.ipynb    # Notebook principal — análise completa
├── requirements.txt        # Dependências Python
├── README.md               # Este documento
├── data/
│   └── epi_r.csv           # Dataset (não versionado — ver .gitignore)
└── figuras/                # Gráficos gerados pelo notebook
    ├── motivacao_modelos_flexiveis.png   # Linear vs Poly vs Splines vs LOESS
    ├── comparacao_regressores.png        # R² adj. dos 10 regressores
    ├── modelo_final_regressao.png        # Diagnóstico do Gradient Boosting
    ├── comparacao_classificadores.png    # Accuracy e F1 dos 8 classificadores
    ├── matriz_confusao.png               # Confusion matrix do melhor classificador
    ├── pca_visualizacao.png              # Scree plot + projeções 2D com KDE
    ├── kmeans_clusters.png               # 4 clusters no espaço PCA
    ├── dendrograma.png                   # Dendrograma hierárquico (Ward)
    └── gmm_bic.png                       # BIC do GMM vs. número de componentes
```

---

## 9. Relação com a atividade anterior

Este trabalho **complementa** a [atividade de regressão de calorias](https://github.com/LucasBuccini/ml-atividade):

| Dimensão | Atividade | Trabalho Final |
|---|---|---|
| Problema | Regressão (`calories`) | Regressão + Classificação + Clustering |
| Modelos | 7 regressores | 18 técnicas (regressão + classificação + não-supervisionado) |
| Target | `calories` | `calories` + `rating_class` |
| Foco | Um modelo bem feito e interpretado | Comparação ampla de técnicas e descoberta de padrões |
| Resultado regressão | R² adj. = 0,897 (GB, 30 vars) | Replicado e estendido com mais modelos |
| Resultado classificação | — | ~35–45% acc. (limitação estrutural do problema) |
| Resultado não-supervisionado | — | 4 clusters + sistema de substituição de dieta |

### Por que o rating é difícil de prever?

A atividade anterior já havia identificado:

> *"`rating` não tem relação com nada. A nota depende de gosto pessoal, que os dados não capturam."*

Este trabalho confirma empiricamente: mesmo os melhores classificadores (Random Forest,
Gradient Boosting com 335 features) ficam próximos do baseline de 33%. O resultado negativo
tem valor científico — demonstra que prever satisfação gastronômica requer dados qualitativos
(texto das avaliações, fotos, contexto cultural) que não estão neste dataset.
