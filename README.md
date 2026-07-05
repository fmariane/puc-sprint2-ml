# Predição de Severidade de Ocorrências Aeronáuticas por Tipo de Operação

MVP individual de Machine Learning & Analytics — Sprint de Machine Learning, PUC.

**Notebook principal:** [`mvp_aviation_severity.ipynb`](./mvp_aviation_severity.ipynb)
**Autora:** Mariane Oliveira

---

## 1. Definição do problema

- **Tipo de tarefa:** classificação multiclasse supervisionada.
- **Variável-alvo:** `aeronave_nivel_dano`, com quatro classes — `NENHUM`, `LEVE`, `SUBSTANCIAL`, `DESTRUÍDA`.
- **Objetivo do modelo:** a partir de características da aeronave e do tipo de operação **disponíveis antes da ocorrência**, prever o nível de dano esperado.
- **Por que é um problema de ML:** existe associação documentada entre características estruturais da operação (tipo de operação, fase de voo, tipo de motor) e a severidade do dano, mas essa relação não é trivialmente descritível por regras determinísticas — há padrões estatísticos que um classificador supervisionado pode aprender a partir de milhares de ocorrências históricas.
- **Pergunta principal:** é possível prever `aeronave_nivel_dano` com desempenho superior a um classificador de referência (baseline), usando apenas características da aeronave e do tipo de operação?
- **Pergunta secundária:** quais tipos de operação apresentam maior proporção de danos substanciais ou destruição total?
- **Premissas e restrições:**
  - Somente variáveis disponíveis antes da ocorrência entram como feature; informações produzidas pela investigação posterior (ex.: `fator_contribuinte`) configurariam *data leakage* e foram descartadas.
  - `aeronave_tipo_operacao` entra como **feature**, não como target — é hipótese do projeto que ela carrega sinal relevante sobre a severidade.
  - O recorte temporal (ocorrências desde 2000, foco analítico em 2010–2025) assume que mudanças regulatórias recentes tornam períodos muito antigos pouco representativos do risco contemporâneo.
  - O desbalanceamento de classes (`DESTRUÍDA` ≈ 4% dos registros) é tratado como característica estrutural do domínio, não como defeito a corrigir a qualquer custo.

## 2. Dataset

- **Fonte:** CENIPA/FAB (Centro de Investigação e Prevenção de Acidentes Aeronáuticos), dados públicos do Portal de Dados Abertos do Governo Federal.
- **Carga:** feita diretamente via URL pública no notebook (`raw.githubusercontent.com/.../data/raw/CENIPA_FAB/`) — não requer upload manual, login ou chave de API.
- **Estrutura:** relacional, com arquivos distintos unidos por `codigo_ocorrencia`:

  | Arquivo | Conteúdo | Registros (~) |
  |---|---|---|
  | `ocorrencia.csv` | Data, localização, classificação do evento | 13.185 |
  | `aeronave.csv` | Tipo de operação, fase do voo, motor, nível de dano | 13.301 |
  | `ocorrencia_tipo.csv` | Categoria taxonômica ICAO do evento | 13.900 |
  | `fator_contribuinte.csv` | Fatores identificados em investigação formal | 8.613 |

- **Variáveis utilizadas como feature:** `aeronave_tipo_operacao`, `ocorrencia_classificacao`, `aeronave_fase_operacao`, `aeronave_motor_tipo`, `aeronave_motor_quantidade`, `ocorrencia_tipo_categoria`.
- **Variável-alvo:** `aeronave_nivel_dano`.
- **Limitações conhecidas:**
  - `fator_contribuinte.csv` cobre apenas ~16% das ocorrências (documentado só após investigação formal), o que o torna inutilizável como feature sem introduzir leakage.
  - Ausência de denominador de exposição (horas voadas/movimentos) por tipo de operação: os resultados descrevem severidade *dado que uma ocorrência foi registrada*, não risco absoluto por operação realizada.
  - Forte desbalanceamento na classe `DESTRUÍDA` (~4%), o que limita o recall alcançável nessa classe.

## 3. Análise exploratória

Realizada na Seção 3 do notebook: tipos de dados, estatísticas descritivas, tratamento do marcador `***` do CENIPA como ausente, mapa de valores ausentes, distribuição da variável-alvo (desbalanceada), severidade por tipo de operação (proporção, não contagem absoluta), distribuição de fase de operação/tipo de motor/classificação da ocorrência, e evolução temporal das ocorrências (2010–2025).

## 4. Preparação dos dados

- **Valores ausentes:** marcador `***` convertido para `NaN`; imputação categórica pela **moda**, feita dentro do pipeline e **após** o split treino/teste (evita leakage).
- **Remoção de leakage:** `fator_contribuinte` descartado por vazar informação de severidade indiretamente (sua taxa de cobertura salta de 7,9% em `LEVE` para 57,0% em `SUBSTANCIAL`).
- **Classes raras:** linhas com classe-alvo com menos de 30 exemplos foram removidas antes do split — abaixo desse limiar, métricas por classe não são confiáveis.
- **Encoding:** `OneHotEncoding` para as features categóricas (evita ordinalidade artificial); `LabelEncoder` apenas para o target (formato inteiro esperado pelos classificadores).
- **Pipeline:** `ColumnTransformer` + `Pipeline` do scikit-learn encapsulam imputação e encoding, garantindo que o conjunto de teste nunca influencie parâmetros aprendidos no treino.

## 5. Divisão dos dados

- Divisão **80/20** treino/teste, com `stratify=y` para preservar a proporção de classes.
- Split realizado **antes** de qualquer pré-processamento; `fit` do pipeline apenas no treino, `transform` aplicado ao teste.
- Validação cruzada (`StratifiedKFold`/`RandomizedSearchCV` com `cv=5`) usada especificamente na etapa de otimização de hiperparâmetros, para não expor o conjunto de teste durante a busca.

## 6. Modelagem

Duas estratégias comparadas empiricamente para testar se `aeronave_tipo_operacao` deve ser feature de um modelo único ou base para modelos especializados:

| | Opção A | Opção B |
|---|---|---|
| Estratégia | XGBoost dedicado por tipo de operação | Modelo único, com tipo de operação como feature OHE |
| Vantagem | Captura padrões específicos por contexto | Estável para tipos com poucos dados; um único modelo a manter |
| Risco | Tipos raros geram modelos frágeis | Pode suavizar diferenças reais entre tipos |

Na Opção B, quatro modelos foram treinados e comparados:

| Modelo | Papel | F1 ponderado (teste) |
|---|---|---|
| `DummyClassifier` (classe majoritária) | **Baseline** | 0,3472 |
| Logistic Regression | Candidato (linear) | 0,7587 |
| Random Forest | Candidato (ensemble de árvores) | 0,7610 |
| **XGBoost** | Candidato principal | **0,7747** |

Na Opção A, um XGBoost independente por tipo de operação (mínimo de 100 registros por tipo) produziu F1 entre 0,44 e 0,81, variando por segmento.

## 7. Otimização de hiperparâmetros

- **Modelo otimizado:** XGBoost (vencedor da Opção B).
- **Estratégia:** `RandomizedSearchCV`, 30 iterações × 5 folds (150 avaliações), otimizando `f1_weighted`.
- **Hiperparâmetros buscados:** `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`.
- **Resultado:** a busca **não superou** o modelo com parâmetros padrão — F1 do tunado = 0,7677 vs. 0,7747 do padrão (Δ = −0,0070), e o recall de `DESTRUÍDA` caiu de 0,38 para 0,26. Causa provável: `f1_weighted` pondera pelo suporte de cada classe, e configurações que melhoram esse valor médio podem ao mesmo tempo tornar o modelo mais conservador na classe mais rara (`DESTRUÍDA`). A busca foi executada corretamente (sem uso do teste durante a otimização), mas o **XGBoost padrão foi mantido como modelo final**, por ser esse o critério do projeto (recall na classe crítica), não apenas o F1 médio.

## 8. Avaliação dos resultados

- **Métricas:** acurácia, precisão/recall/F1 ponderados, matriz de confusão e classification report — F1 ponderado escolhido como métrica principal por penalizar modelos que ignoram classes minoritárias em um problema desbalanceado.
- **Melhor resultado:** XGBoost padrão (Opção B), F1 ponderado = **0,7747**, muito acima do baseline (0,3472).
- **Diagnóstico de generalização:** F1 treino = 0,8005, F1 CV = 0,7800, F1 teste = 0,7677 (modelo tunado) — gap treino-teste de +0,033, classificado como *overfitting leve, mas aceitável*.
- **Limitação central:** recall de `DESTRUÍDA` ≈ 0,38 (modelo padrão) é o principal ponto fraco — não é corrigível apenas por ajuste de algoritmo ou hiperparâmetros; reflete a raridade estrutural dessa classe (~4% dos dados) e a ausência de mais sinal nas seis variáveis categóricas disponíveis.
- **Caminhos de enriquecimento descartados e por quê:** uso de `fator_contribuinte` (leakage), SMOTE sobre features categóricas OHE (gera valores fracionários sem interpretação válida).

## 9. Conclusão

O projeto demonstrou que características estruturais disponíveis antes da ocorrência — sem qualquer informação produzida por investigação posterior — permitem prever `aeronave_nivel_dano` com F1 ponderado de 0,7747, muito acima do baseline de classe majoritária (0,3472). A Opção B (modelo único com tipo de operação como feature) foi preferida à Opção A (modelos por tipo) por não falhar catastroficamente em nenhum segmento e exigir manutenção de um único artefato. A limitação central e documentada é o recall de `DESTRUÍDA`, consequência da raridade estrutural dessa classe no domínio, não de uma falha metodológica.

**Próximos passos sugeridos:**
- Calibração de limiar de decisão para `DESTRUÍDA` (faixa 0,25–0,35), sem necessidade de retreinamento.
- Uso de `sample_weight` como alternativa ao oversampling para dar mais peso à classe minoritária.
- Incorporação de dados não estruturados (relatórios narrativos do CENIPA) via TF-IDF/BERTimbau, reformulando `DESTRUÍDA` como detecção de anomalia — enquadrada como análise retrospectiva, não predição em tempo real.

## Como executar

O notebook roda do início ao fim sem configuração manual — os dados são carregados via URL pública.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook mvp_aviation_severity.ipynb
```

Também pode ser aberto e executado diretamente no Google Colab.

## Estrutura do repositório

```
mvp_aviation_severity.ipynb   # notebook principal (entrega do MVP)
fator_contribuinte_eda.ipynb  # EDA de apoio sobre fatores contribuintes (não faz parte da entrega)
data/raw/CENIPA_FAB/          # cópia local dos CSVs de origem (carga real é via URL pública no notebook)
data/raw/VRA/                 # dados de voos regulares (não utilizados neste MVP)
requirements.txt              # dependências do ambiente local
```

## Principais bibliotecas

`pandas`, `numpy`, `scikit-learn`, `xgboost`, `plotly` (visualizações interativas).
