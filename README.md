# RAG RRF PURO — TF-IDF global + Embeddings global fundidos por RRF (Boeing 737 AMM) + RAGAS

Projeto **irmão** do [`avaliadorRAGAS-FAISS`](../avaliadorRAGAS-FAISS): mesma base
de conhecimento, mesmas questões, mesmo avaliador — **só a recuperação muda**.

## O método (RRF puro, grupo de controle)

Dois rankings **globais** sobre os 26.039 chunks, fundidos por Reciprocal Rank
Fusion:

| Ranking | Como |
|---|---|
| Esparso | TF-IDF (scikit-learn) sobre todos os chunks |
| Denso | embeddings (API NVIDIA) — vetores globais |
| Fusão | `score(d) = 1/(k+rank_sparse(d)) + 1/(k+rank_dense(d))`, k=60 |

Os top-3 chunks fundidos vão para o LLM gerador.

## Reaproveitamento do cache de embeddings

Por padrão os vetores densos são **reaproveitados do projeto FAISS**
(`--indice-externo`, somente leitura — evita re-embeddar 26k chunks). Se o cache
externo não existir ou não bater com o corpus local, o pipeline embedda tudo e
salva cache próprio em `output/`.

## O que é igual ao projeto FAISS (regra de ouro)

Corpus, chunking (1000/200), 26.039 chunks, 52 questões, LLM gerador
(`nemotron-3.5-lightning-30b-a3b`, temperatura 0, `max_completion_tokens=16384`, thinking LIGADO), prompt,
`top_k=3`, `evaluate_ragas.py` e `prepare_dataset.py` **cópias idênticas**.

## Ambiente

```powershell
cd C:\Users\joser\Desktop\cursos\avaliadorRAGAS-RRF
# usa o interpretador do sistema que ja tem todas as dependencias (inclusive ragas):
$py = "C:\Users\joser\AppData\Local\Python\bin\python.exe"
& $py src\rag_rrf.py -n 5
& $py src\evaluate_ragas.py
```

> Não é necessário criar venv: o ragas não instala em Python 3.14 sem dor
> (scikit-network sem wheel), e o interpretador do sistema já tem tudo.
> A chave da API está no `.env` (copiada do projeto FAISS).

## Como executar

```powershell
& $py src\rag_rrf.py -n 5                  # 5 primeiras questoes
& $py src\rag_rrf.py -r -n 5 -s 123        # sorteio com seed fixa (comparacao head-to-head)
& $py src\rag_rrf.py -n 52                 # rodada oficial (todas as questoes)
& $py src\rag_rrf.py -n 5 --max-chunks 2000  # teste rapido (subset)
& $py src\evaluate_ragas.py                # juiz RAGAS (copia identica)
```

- A primeira execução ajusta o TF-IDF (~1 min) e reaproveita o cache de
  embeddings do FAISS (instantâneo).
- Saída: `output/predictions.json` (mesmo schema do projeto FAISS).

## Avaliação (2 caminhos)

```powershell
# 1) avaliação completa de uma vez (funciona quando a API está tranquila):
& $py src\evaluate_ragas.py

# 2) avaliação EM LOTES, serial e com retomada (robusta ao limite de 40 req/min
#    da cota free da NVIDIA — o padrão do ragas dispara 16 workers em paralelo
#    e estoura a cota com erro 429, zerando as métricas):
& $py src\avaliar_lote.py                    # lotes de 13, pausa 45s, salva cada lote
& $py src\avaliar_lote.py --retomar           # continua de onde parou
& $py src\avaliar_lote.py --pred output\predictions_5q.json --out output\resultado_5q.csv --lote 5
```

> **Lição importante:** com a cota free, use sempre `avaliar_lote.py`
> (`RunConfig(max_workers=1)`). Com os 16 workers padrão, o RAGAS toma 429 em
> massa e grava 0 em todas as métricas — parece derrota do método, mas é
> falha de infra.

## Estrutura

```
avaliadorRAGAS-RRF/
├── src/rag_rrf.py           # NOVO: pipeline RRF puro
├── src/evaluate_ragas.py    # copia IDENTICA do FAISS (o juiz)
├── src/prepare_dataset.py   # copia identica
├── data/ (corpus + train.json)
├── output/                  # predictions.json, cache proprio (fallback), CSV
└── results.md               # resultados e comparacao head-to-head
```

## Correções de 24/10/2026 (aplicadas nos 4 projetos comparáveis)

> **Regra de ouro:** estas correções foram aplicadas **iguais** no FAISS, TF-IDF,
> MISTURA e RRF. O `evaluate_ragas.py` dos 4 é agora **byte-idêntico**. Se você mudar
> algo aqui, mude nos 4 — senão a comparação head-to-head deixa de ser justa.

### 1. Gerador: `max_tokens` limita *raciocínio + resposta* juntos

O `nemotron-3.5-lightning` é modelo de **raciocínio** e o raciocínio não tem teto
(medido: 2,3k–6,5k tokens por questão). Com `max_completion_tokens=16384`, thinking LIGADO ele estourava o
orçamento e a API devolvia `finish_reason=length` com `content` **vazio** — foram
**7 respostas vazias em 52** no projeto FAISS. Não era determinístico, mesmo com
`temperature=0` (3 chamadas idênticas: 2 respondiam, 1 vinha vazia).

**Correção:** `GERADOR_MAX_TOKENS = 16384` com thinking **LIGADO**, `timeout=300` e
retry com backoff que trata resposta vazia como falha.

Duas opções foram testadas antes de escolher: desligar o thinking zerava as vazias,
mas fazia **2 respostas boas virarem fallback** (`gap-20-20`, `gt_014`). Com thinking
ligado e 16.384 tokens as duas voltaram a responder certo.

### 2. Juiz RAGAS: a mesma doença + um terceiro modo de falha

O juiz (`nemotron-3.5-super-vl-120b-a12b`) também é de raciocínio: mediu
**6.906 de 8.192 tokens** num prompt simples (84% do orçamento).

Além do conteúdo vazio, há um modo de falha mais sutil: o `ChatNVIDIA` propaga
`finish_reason` para o `response_metadata`, e o ragas só aceita
`["stop","STOP","MAX_TOKENS","eos_token"]` em `is_finished()`, levantando
`LLMDidNotFinishException` caso contrário. Resposta truncada com conteúdo não-vazio
também virava NaN.

**Correção:** classe `JuizNVIDIA` com retry que rejeita **conteúdo vazio** *e*
`finish_reason != "stop"`; `JUIZ_MAX_TOKENS = 16384`; `timeout=600`.

### 3. O vilão final: `fillna(0.0)` transformava falha em nota 0

O `df_res.fillna(0.0)` convertia a falha do juiz (NaN) em **nota 0**, corrompendo as
médias. Prova: `context_precision` **não depende da resposta** (só de pergunta +
contextos + ground truth) e a recuperação era idêntica, mas ele diferia em **21 das 52
questões** entre duas execuções.

**Correção:** `fillna` **removido**. Falhas ficam NaN, são **contadas por métrica** no
resumo final e excluídas das médias.

### 4. `RunConfig` do ragas: timeout e workers

O padrão (`timeout=180`, `max_workers=16`) matava jobs lentos e provocava
`429 Too Many Requests` por excesso de concorrência. Agora:
`RunConfig(timeout=900, max_workers=4)`.

### 5. Juiz configurável por provedor (`JUIZ_PROVIDER`)

| Valor | Juiz |
|---|---|
| `nvidia` (padrão) | `nvidia/nemotron-3.5-super-vl-120b-a12b` (histórico) |
| `groq` | `llama-3.3-70b-versatile` via `ChatOpenAI` + `base_url` do Groq |

Por que o Groq: é **instruct puro** — não raciocina, logo não estoura o orçamento de
tokens. Bônus: o ragas lista `ChatOpenAI` em `MULTIPLE_COMPLETION_SUPPORTED`
(`ragas/llms/base.py`) e o `ChatNVIDIA` **não** — com Groq ele agrupa as *n* completions
em **1 request**, o que reduz os 429.

**Não precisa instalar nada:** a API do Groq é OpenAI-compatible e `langchain-openai`
já é dependência do projeto.

Para usar, adicione ao `.env`:

```
JUIZ_PROVIDER=groq
GROQ_API_KEY=gsk-...      # obtenha em https://console.groq.com/keys
```

> ⚠️ **Trocar o juiz muda os valores absolutos das notas.** Os 4 projetos precisam usar
> o **mesmo** juiz para a comparação continuar válida.

### 6. Relevância corrigida para embeddings NVIDIA assimétricos

O `answer_relevancy` padrão do RAGAS 0.4.3 codifica a pergunta original como
`query`, mas as perguntas geradas a partir da resposta como `passage`. O
`nvidia/nemotron-3-embed-1b` usa espaços vetoriais distintos para esses modos,
o que penaliza respostas corretas por diferenças terminológicas.

A correção está em `src/metricas_ragas.py`: mantém a geração de perguntas e o
cálculo de similaridade do RAGAS, mas codifica todos os textos no modo `query`.
Também faz o embedding em lote, reduzindo as chamadas de embedding à API. Os
HTTP 429 do juiz LLM são independentes e são tratados por retry, execução
serial e lotes menores. A mudança foi aplicada igualmente nos projetos FAISS,
TF-IDF, MISTURA e RRF.

> Métricas antigas e novas têm escalas diferentes. Para a comparação
> head-to-head, regrave os quatro projetos com a mesma versão do avaliador.

### Como ler as notas agora

O resumo final mostra, por métrica, quantas questões **falharam** e quantas são
**válidas**. Exemplo (rodada do FAISS com juiz NVIDIA):

| Métrica | Nota | Válidas |
|---|---|---|
| Faithfulness | 0,81 | 24/52 |
| Answer Relevancy | 0,22 | 6/52 — amostra pequena, **não use** |
| Context Precision | 0,44 | 21/52 |
| Context Recall | 0,56 | 29/52 |

Se a cobertura estiver baixa, o número não é confiável: rerode (ou troque o juiz para
`groq`).
