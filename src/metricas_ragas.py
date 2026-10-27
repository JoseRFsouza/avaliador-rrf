"""Métricas RAGAS ajustadas para embeddings NVIDIA assimétricos.

O RAGAS 0.4.3 calcula ``answer_relevancy`` comparando a pergunta original
codificada como ``query`` com as perguntas geradas a partir da resposta
codificadas como ``passage``. No ``nvidia/nemotron-3-embed-1b`` esses modos
pertencem a espaços vetoriais distintos, o que penaliza respostas corretas
apenas por diferenças terminológicas.

A classe abaixo preserva a geração de perguntas, o verificador de resposta
evasiva e o cálculo de similaridade do RAGAS, mas codifica todos os textos
no modo ``query``. A chamada em lote reduz as requisições de embedding à API
NVIDIA; os HTTP 429 do juiz LLM são independentes e exigem execução serial
ou lotes menores.
"""

import numpy as np
from ragas.metrics._answer_relevance import AnswerRelevancy


class AnswerRelevancyNvidia(AnswerRelevancy):
    """Relevância com comparação simétrica no modo de embedding ``query``."""

    name: str = "answer_relevancy"

    def calculate_similarity(self, question: str, generated_questions: list[str]):
        if self.embeddings is None:
            raise ValueError("A métrica answer_relevancy requer embeddings.")

        texts = [question, *generated_questions]
        provider = getattr(self.embeddings, "embeddings", self.embeddings)

        embed_batch = getattr(provider, "_embed", None)
        if callable(embed_batch):
            vectors = np.asarray(
                embed_batch(texts, model_type="query"),
                dtype=float,
            )
        else:
            vectors = np.asarray(
                [provider.embed_query(text) for text in texts],
                dtype=float,
            )

        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms == 0):
            raise ValueError("Embedding com norma zero recebido do provedor.")

        normalized = vectors / norms[:, None]
        return normalized[1:] @ normalized[0]


answer_relevancy = AnswerRelevancyNvidia()
