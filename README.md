# Identificação de Fatores Relevantes para Melhoria de Ranqueamento em SERP a partir de Métricas de SEO

Código-fonte dos experimentos do projeto da disciplina **Reconhecimento de Padrões (PPGCC21)**, Mestrado em Ciência da Computação, UTFPR Campus Campo Mourão.

**Autor:** Felipe Augusto Medici de Oliveira (RA 2458772)

EN: *Identification of Relevant SEO Factors for SERP Ranking Improvement Using Dataset-Based Analysis*

## Visão geral

O projeto trata uma tarefa de **classificação binária supervisionada**: prever a variável-alvo `ranking_improved` (1 = a página melhorou de posição na SERP; 0 = não melhorou) a partir de 13 métricas de SEO. O objetivo central não é apenas classificar, mas **identificar quais fatores de SEO mais se associam à melhoria de ranqueamento**, respondendo à pergunta de pesquisa:

> Quais métricas de SEO possuem maior relevância para prever a melhoria do ranqueamento de uma página em SERP?

## Conjunto de dados

Utiliza-se o **SEO Web Content Dataset** (Kaggle, autor *Ziya*): 500 instâncias, 14 colunas (13 preditoras + 1 alvo), sem valores ausentes, com distribuição-alvo **312 (62,4%) / 188 (37,6%)** e correlações lineares fracas com o alvo (|r| máximo ≈ 0,09). O arquivo `data/seo_dataset.csv` acompanha o repositório para reprodutibilidade.

> A licença de uso deve ser confirmada na página do Kaggle antes de qualquer redistribuição.

| Atributo | Tipo | Descrição |
|---|---|---|
| content_length | inteiro | Quantidade de palavras da página |
| keyword_density | decimal | Percentual de uso de palavras-chave |
| num_internal_links | inteiro | Número de links internos |
| num_external_links | inteiro | Número de links externos |
| has_meta_description | binário | Presença de meta description |
| has_alt_text | binário | Presença de texto alternativo em imagens |
| avg_time_on_page_sec | inteiro | Tempo médio na página (segundos) |
| bounce_rate | decimal | Percentual de rejeição |
| scroll_depth_percent | decimal | Profundidade média de rolagem |
| domain_authority | inteiro | Autoridade do domínio (1 a 100) |
| page_authority | inteiro | Autoridade da página (1 a 100) |
| backlink_count | inteiro | Quantidade de backlinks |
| serp_position_before | inteiro | Posição na SERP antes dos ajustes |
| **ranking_improved** | binário | **Variável-alvo** |

## Estrutura do repositório

```
seo-serp-ranking/
├── data/
│   └── seo_dataset.csv          # conjunto de dados
├── src/
│   ├── config.py                # caminhos, semente, alvo, grupos de fatores
│   ├── data_loader.py           # carga e validação de schema
│   ├── eda.py                   # análise exploratória e figuras
│   ├── preprocessing.py         # pipelines anti-vazamento (scaler/SMOTE por fold)
│   ├── models.py                # modelos, espaços de busca, ensembles
│   ├── evaluation.py            # CV repetida, CV aninhada, holdout, métricas
│   ├── stats_tests.py           # Friedman, Nemenyi, Wilcoxon
│   ├── interpretability.py      # importância (permutação, MI, grupos), ablação
│   └── experiment.py            # orquestrador: gera tabelas e figuras
├── tests/
│   └── test_pipeline.py         # testes (pytest)
├── results/
│   ├── tables/                  # métricas em CSV
│   └── figures/                 # figuras em PNG
├── requirements.txt
└── README.md
```

## Instalação

Requer **Python ≥ 3.10**.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Execução

Reproduzir todos os experimentos (gera tabelas em `results/tables` e figuras em `results/figures`):

```bash
python -m src.experiment
```

Executar os testes:

```bash
python -m pytest tests/ -q
```

## Protocolo experimental (resumo)

1. **Pré-processamento anti-vazamento:** escalonamento (`StandardScaler`) e reamostragem (`SMOTE`) ajustados somente no fold de treino, via `Pipeline` do scikit-learn / imbalanced-learn. Árvores não recebem escalonamento.
2. **Modelos:** `DummyClassifier` (baseline majoritário), Regressão Logística, k-NN, SVM-RBF, Random Forest, XGBoost, além de combinações por *Soft Voting* e *Stacking*. Desbalanceamento tratado por `class_weight='balanced'` e `scale_pos_weight`.
3. **Validação:** validação cruzada estratificada repetida (5 folds × 3 repetições) para a comparação principal e CV aninhada (externa 5-fold + interna 3-fold com `RandomizedSearchCV`) para os modelos ajustados. Holdout estratificado de 20% para confirmação final.
4. **Métricas:** PR-AUC (primária), F1-macro, Balanced Accuracy, ROC-AUC, Precision/Recall/F1 da classe positiva e acurácia.
5. **Testes estatísticos:** Friedman + pós-teste de Nemenyi (diagrama de diferença crítica) e Wilcoxon pareado contra o baseline (α = 0,05).
6. **Interpretabilidade:** importância por permutação agregada em folds, informação mútua e agregação por grupo de fatores; ablação do confundidor `serp_position_before`.

## Principais resultados

- O sinal preditivo do conjunto é fraco: nenhum modelo supera substancialmente o baseline majoritário; ROC-AUC ≈ 0,5.
- A **Regressão Logística** foi o melhor modelo (PR-AUC médio 0,413), sendo o único a superar o baseline de forma estatisticamente significativa (Wilcoxon p = 0,003).
- Modelos complexos (Random Forest, XGBoost, Stacking) não trouxeram ganho sob sinal fraco, o que contraria a hipótese inicial de superioridade de ensembles.
- Por informação mútua, os fatores mais associados ao alvo são **comportamentais** (tempo na página, taxa de rejeição), de **autoridade** (autoridade do domínio, backlinks) e **contextuais** (posição anterior na SERP); métricas puramente *on-page* contribuem menos.

Os valores completos estão em `results/tables/` e as figuras em `results/figures/`.

## Reprodutibilidade

Sementes fixas (`random_state = 42`) em NumPy, scikit-learn, XGBoost, SMOTE e nas partições; versões de bibliotecas fixadas em `requirements.txt`; todo o pré-processamento encapsulado em `Pipeline`, garantindo transformações idênticas por fold e ausência de vazamento.

## Uso de IA generativa (Spec Driven Development)

O desenvolvimento empregou ativamente **Spec Driven Development** com modelos de linguagem (LLMs), conduzido com o **Claude Code** (interface agêntica de linha de comando da Anthropic) utilizando o modelo **Claude Opus 4.8**. A partir de especificações escritas, foram montados agentes de busca sobre o tema, estruturado o projeto, definidos e avaliados métodos e algoritmos, gerados e testados códigos com diferentes metodologias, e conduzidas revisões de texto e de organização do projeto. O uso concentrou-se em dois eixos: a criação e a verificação do código dos experimentos e a otimização do estudo (comparação entre modelos, protocolo de avaliação e estruturação do texto científico). Todo o conteúdo técnico e os resultados foram verificados pelo autor.

## Licença

Código liberado para fins acadêmicos. A licença do conjunto de dados deve ser confirmada diretamente no Kaggle.
