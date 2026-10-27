# Resultados — RAG RRF PURO (TF-IDF global + embeddings global)

> Preencha **antes** de comparar com os projetos TF-IDF e FAISS.

## Configuração da rodada

| Item | Valor |
|---|---|
| Data | 2026-10-21 |
| Questões | 5 (head-to-head com baseline FAISS) |
| Retrieval | RRF (TF-IDF global + embedding global), top_k=3 |
| Embeddings (retrieval) | nvidia/nemotron-3-embed-1b (cache do FAISS) |
| LLM gerador | nvidia/nemotron-3.5-lightning-30b-a3b (temperature 0) |
| Juiz RAGAS | nvidia/nemotron-3.5-super-vl-120b-a12b |
| Chunks | 26.039 (splitter 1000/200) |
| k do RRF | 60 |

## Notas desta rodada

| Métrica | Nota |
|---|---|
| Faithfulness | 0,000 |
| Answer Relevancy | 0,000 |
| Context Precision | 0,100 |
| Context Recall | 0,400 |
| **Geral (média)** | **0,125** |
| Questões com total == 0 | 3 de 5 |
| Respostas vazias | 2 (gerador devolveu conteúdo vazio) |

## Comparação head-to-head (mesmas questões)

| Projeto | Retrieval | Faithfulness | Relevancy | Precision | Recall | **Geral** | total==0 |
|---|---|---|---|---|---|---|---|
| avaliadorRAGASTF-IDF | TF-IDF em camadas | | | | | | |
| avaliadorRAGAS-FAISS | FAISS top-k global | 0,760 | 0,199 | 0,367 | 0,511 | **0,459** | 0 |
| **avaliadorRAGAS-RRF (este)** | RRF global | 0,000 | 0,000 | 0,100 | 0,400 | 0,125 | 3 |
| avaliadorRAGAS-MISTURA | escopo + RRF | 0,333 | 0,000 | 0,000 | 0,000 | 0,083 | 3 |

## Observações

- (ex.: comportamento do RRF vs. denso puro, perguntas em que a fusão
  recuperou/errou, custo/latência)
