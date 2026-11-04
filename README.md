# Trabalho Final de Aprendizado de Máquina — Epicurious Dataset

**Disciplina:** Aprendizado de Máquina — Prof. Marcelo Guzella (IBMEC)
**Autor:** Lucas Buccini
**Dataset:** [Epicurious – Recipes with Rating and Nutrition](https://www.kaggle.com/datasets/hugodarwood/epirecipes) (Kaggle)
**Notebook principal:** [trabalho_final.ipynb](trabalho_final.ipynb)

> Complemento da [atividade anterior](https://github.com/LucasBuccini/ml-atividade), que previu calorias
> com R² ajustado de **0,897**. Aqui o mesmo dataset é explorado com **23 técnicas** em três frentes:
> regressão de calorias, categorização automática do tipo de prato e segmentação
> nutricional não supervisionada.

---

## Sumário

1. [Visão geral](#1-visão-geral)
2. [Dados e preparação](#2-dados-e-preparação)
3. [Bloco 1 — Regressão](#3-bloco-1--regressão-de-calorias)
4. [Bloco 2 — Classificação](#4-bloco-2--classificação-tipo-de-prato)
5. [Bloco 3 — Não supervisionado](#5-bloco-3--segmentação-nutricional)
6. [Ética e limitações](#6-ética-e-limitações)
7. [Como executar](#7-como-executar)
8. [Estrutura do repositório](#8-estrutura-do-repositório)

---

## 1. Visão geral

| Bloco | Pergunta | Técnicas | Melhor resultado (teste) |
|---|---|---|---|
| 1 — Regressão | Quantas calorias tem a receita? | 10 | Gradient Boosting: R² aj. = 0,897, MAE = 66 kcal |
| 2 — Classificação | Qual o tipo de prato (6 classes)? | 8 + seleção por consenso | Gradient Boosting: F1-macro = 0,766, acurácia = 0,833 |
| 3 — Não supervisionado | Quais perfis nutricionais existem? | PCA, K-Means, Hierárquico, GMM, DBSCAN | 7 perfis interpretáveis + recomendador de substituição |

---

## 2. Dados e preparação

Dois arquivos do mesmo dataset Kaggle:
- `epi_r.csv` — título, `rating`, `calories`, `protein`, `fat`, `sodium` e 674 tags binárias.
- `full_format_recipes.json` — ingredientes, passos e data de publicação (nº de ingredientes, nº de passos e ano, usados no Bloco 2).

| Etapa de limpeza | Registros |
|---|---|
| Dados originais | 20.052 |
| Remoção de duplicatas | 18.251 |
| Remoção de nulos (macros + rating) | 14.446 |
| Remoção de outliers (p99 de cada variável nutricional) | 14.096 |
| Bloco 2: receitas com exatamente 1 categoria (modelagem) | 7.704 |

- Tags com menos de 50 ocorrências são descartadas (332 mantidas); `low cal` é removida por vazamento do alvo de regressão.
- Divisão treino/teste 80/20 com `random_state=42` (estratificada no Bloco 2). Escalonamento, seleção de variáveis, hiperparâmetros e limiar usam **somente o treino**.

---

## 3. Bloco 1 — Regressão de calorias

Validação cruzada de 5 folds no treino + avaliação no teste. Seleção de variáveis explícita via Lasso
(`TOP30_FEATURES`, 21/30 coincidem com o Top-30 do Lasso refeito no treino atual).

| Modelo | Variáveis | R² aj. (teste) | MAE (kcal) |
|---|---|---|---|
| Regressão Linear | 3 | 0,845 | 95 |
| Polinomial (grau 2) | 3 | 0,845 | 96 |
| Splines cúbicas naturais | 3 | 0,844 | 96 |
| GAM | 3 | 0,852 | 90 |
| Ridge | 30 | 0,887 | 71 |
| Lasso (LassoCV) | 335 (246 ≠ 0) | 0,880 | 67 |
| Árvore de decisão | 30 | 0,819 | 87 |
| Bagging | 30 | 0,873 | 77 |
| Random Forest | 30 | 0,882 | 70 |
| **Gradient Boosting** | **30** | **0,897** | **66** |

---

## 4. Bloco 2 — Classificação (tipo de prato)

### Por que não o rating?
O rating foi testado em **6 formulações** com o mesmo protocolo (Gradient Boosting, divisão 80/20):

| Formulação | Mede | Resultado |
|---|---|---|
| Regressão da nota | qualidade | R² = 0,04 |
| 3 classes (≤ 3,75 / 4,375 / 5,0) | qualidade | F1-macro = 0,48 (acaso 0,33) |
| Bem avaliada (≥ 4,375) | qualidade | ROC-AUC = 0,62 |
| Mal avaliada (≤ 3,125) | qualidade | ROC-AUC = 0,62 |
| Nota máxima (= 5,0) | exposição | ROC-AUC = 0,75 (0,72 sem o ano) |
| Receberá avaliação? | exposição | ROC-AUC = 0,88 (0,84 sem o ano) |

A qualidade percebida não é previsível com estes dados; o sinal existente vem do **ano de publicação e da fonte
editorial** (exposição), porque o dataset não traz o número de avaliações. Também foram descartados: restrições
alimentares (rótulos inconsistentes) e classificação por texto/TF-IDF (fora do escopo da disciplina; +5 p.p. de F1).

### Problema escolhido
**39% das receitas não têm categoria** (bebida, sobremesa, acompanhamento, entrada, prato principal, café da manhã)
e ficam fora dos filtros do site. O classificador sugere a categoria a partir de ingredientes, técnicas e nutrição.

| Situação do rótulo | Receitas | Uso |
|---|---|---|
| Exatamente 1 categoria | 7.704 | treino/teste (80/20 estratificado) |
| 2+ categorias | 888 | validação complementar |
| Sem categoria | 5.504 | aplicação |

**Vazamento:** excluídas as tags que definem a categoria (`dessert`, `drink`...) e as que nomeiam um tipo de prato
(`cake`, `cocktail`, `salad`, `soup/stew`...; pureza de até 100%).

### Seleção de variáveis (somente no treino)
| Etapa | Método | Resultado |
|---|---|---|
| Filtro | Variância (tags < 1%) | −115 tags |
| Filtro | Redundância (\|r\| > 0,90) | nenhum par |
| Filtro | Informação mútua, k por CV aninhada | k = 160 |
| Embutido | Lasso multinomial (L1) | 177 não nulas |
| Embutido | Permutação (Gradient Boosting) | 77 significativas |
| Wrapper | RFECV | 148 |
| **Consenso** | **≥ 2 votos** | **175 de 313** (todas as 9 numéricas) |

### Resultados (teste)
| Modelo | F1-macro (CV) | Acurácia | F1-macro | ROC-AUC (OvR) |
|---|---|---|---|---|
| Baseline (classe majoritária) | 0,080 | 0,317 | 0,080 | 0,500 |
| Regressão Logística | 0,728 | 0,788 | 0,732 | 0,949 |
| LDA | 0,735 | 0,799 | 0,722 | 0,941 |
| QDA | 0,689 | 0,761 | 0,695 | 0,934 |
| Naive Bayes (Bernoulli) | 0,685 | 0,784 | 0,686 | 0,933 |
| KNN | 0,632 | 0,746 | 0,639 | 0,883 |
| SVM (kernel RBF) | 0,744 | 0,830 | 0,751 | 0,956 |
| Random Forest | 0,736 | 0,818 | 0,744 | 0,946 |
| **Gradient Boosting** ★ | **0,769** | **0,833** | **0,766** | **0,956** |

★ Escolhido pelo F1-macro em validação cruzada. Bebida e sobremesa: F1 ≈ 0,96; "entrada" é a classe difícil (0,45),
confundida com acompanhamento e principal (categorias editorialmente ambíguas). **Classificação com rejeição:**
com confiança ≥ 0,96, 64% das receitas são categorizadas automaticamente com 94,9% de acerto; o resto vai para revisão.
Nas receitas sem categoria (menos tags), 37% passam no limiar.

---

## 5. Bloco 3 — Segmentação nutricional

**Seleção de atributos:** em vez de reutilizar `TOP30_FEATURES` (selecionadas para prever calorias e 90% binárias),
partimos de 9 candidatas nutricionais e aplicamos critérios explícitos: transformação log (assimetria), uma única
variável de escala (as quantidades absolutas medem o tamanho da porção: PC1 = 63%), remoção de redundância e de
combinação linear exata. Variáveis finais: `log_calorias`, `pct_energia_gordura`, `pct_energia_proteina`,
`log_sodio_por_100kcal`.

| Espaço | Hopkins | Silhouette (k=4) | Davies–Bouldin | PC1+PC2 |
|---|---|---|---|---|
| TOP30 (versão anterior) | 0,973* | 0,178 | 1,87 | 17,7% |
| **4 selecionadas** | 0,910 | 0,253 | 1,24 | **66,8%** |

\* inflado por vetores binários idênticos.

**PCA:** Kaiser → 2 componentes (66,8%). PC1 = eixo salgado/proteico × doce/carboidrato; PC2 = densidade energética.

**K-Means:** k escolhido por votação (cotovelo, silhouette, Davies–Bouldin, Calinski–Harabasz) entre os k
estáveis no bootstrap → **k = 7** (silhouette = 0,25; estabilidade ARI = 0,93).

| Perfil | kcal | Energia G / P / C | Tags características |
|---|---|---|---|
| Molhos e petiscos leves | 70 | 30 / 16 / 53% | no-cook, sauce, condiment |
| Bebidas e coquetéis | 233 | 7 / 3 / 90% | gin, cocktail, alcoholic |
| Saladas e entradas | 237 | 71 / 10 / 19% | salad, appetizer, leafy green |
| Sobremesas ricas | 398 | 60 / 6 / 34% | chocolate, almond, dessert |
| Massas, pães e acompanhamentos | 411 | 34 / 11 / 55% | pasta, cake, bake, side |
| Proteína magra (peixes/aves) | 434 | 40 / 40 / 20% | turkey, fish, shrimp |
| Pratos principais fartos | 875 | 60 / 21 / 19% | beef, chicken, pork, roast |

**Outros métodos:** Hierárquico Ward (escolhido entre 4 linkages; ARI = 0,49 com K-Means), GMM (BIC → 10
componentes; 18,7% das receitas híbridas) e DBSCAN (97% em um núcleo denso + 2,2% de atípicas). Todos indicam que
os perfis formam um **contínuo**: os clusters são uma segmentação útil, não grupos isolados.

**Aplicação:** `sugerir_substitutos("Chocolate Cake")` retorna receitas do mesmo perfil, com ≥ 20% menos calorias,
ordenadas por similaridade de cosseno entre as tags.

---

## 6. Ética e limitações

- Dataset público sem dados pessoais. Uma aplicação real com histórico de usuários trataria **dados sensíveis de saúde**
  (LGPD, art. 11): exigiria base legal específica, consentimento e minimização.
- Categorias e rating refletem o público do Epicurious (viés cultural); ranquear pelo rating criaria *feedback loops* de exposição.
- Carboidratos são estimados por diferença; o número de avaliações não está disponível; as substituições não
  substituem orientação nutricional.

---

## 7. Como executar

Requer Python 3.10+ e scikit-learn ≥ 1.8.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
jupyter lab trabalho_final.ipynb
```

O notebook procura `data/epi_r.csv` e `data/full_format_recipes.json`; se não existirem, baixa via `kagglehub`
(credenciais em `~/.kaggle/kaggle.json`) ou podem ser baixados manualmente do
[Kaggle](https://www.kaggle.com/datasets/hugodarwood/epirecipes).

Execução completa sem interface: `jupyter nbconvert --to notebook --execute --inplace trabalho_final.ipynb` (~20 min).

---

## 8. Estrutura do repositório

```
ml-trabalho-final/
├── trabalho_final.ipynb                    # Notebook principal (executado, com saídas)
├── apresentacao_trabalho_final.pptx       # Apresentação (10 slides, com notas do apresentador)
├── requirements.txt
├── README.md
├── data/                                   # epi_r.csv + full_format_recipes.json (não versionados)
└── figuras/
    ├── motivacao_modelos_flexiveis.png     # Bloco 1: linear × polinomial × spline × LOESS
    ├── selecao_lasso_regressao.png         # Bloco 1: ranking do Lasso
    ├── comparacao_regressores.png
    ├── modelo_final_regressao.png
    ├── rating_viabilidade.png              # Bloco 2: viés de exposição do rating
    ├── classificacao_alvo.png              # Bloco 2: alvo e desbalanceamento
    ├── selecao_variaveis_classificacao.png # Bloco 2: MI, L1, permutação, RFECV
    ├── consenso_variaveis.png
    ├── comparacao_classificadores.png
    ├── matriz_confusao.png                 # matriz 6×6 + métricas por classe + ROC
    ├── importancia_classificacao.png       # variáveis por classe
    ├── rejeicao_cobertura.png              # cobertura × acurácia
    ├── correlacao_candidatas_cluster.png   # Bloco 3
    ├── pca_visualizacao.png                # scree + loadings + biplot
    ├── kmeans_selecao_k.png
    ├── kmeans_clusters.png                 # clusters + silhouette + centróides
    ├── dendrograma.png
    ├── gmm_bic.png
    └── dbscan.png
```
