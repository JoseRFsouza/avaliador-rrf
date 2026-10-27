r"""Pipeline RAG "RRF PURO" sobre o manual AMM Boeing 737.

Projeto irmao do avaliadorRAGAS-FAISS. A UNICA diferenca de metodo eh o retrieval,
que aqui funde dois rankings GLOBAIS por Reciprocal Rank Fusion (RRF):

  - esparso : TF-IDF (scikit-learn) sobre os 26.039 chunks
  - denso   : embeddings (API NVIDIA) — vetores globais
  - fusao   : score(d) = 1/(k+rank_sparse(d)) + 1/(k+rank_dense(d)), k=60
  - saida   : top_k chunks -> LLM gerador

Por padrão os vetores densos globais sao REAPROVEITADOS do projeto FAISS
(--indice-externo, somente leitura — evita re-embeddar 26k chunks). Se o cache
externo nao existir ou nao bater com o corpus local, o pipeline embedda tudo e
salva cache proprio em output/.

Todo o resto eh IDENTICO ao projeto FAISS (comparacao justa):
  - mesmo corpus, mesmo chunking (1000/200), mesmo LLM gerador (temp 0),
    mesmo prompt, mesmo top_k (3), mesma logica de sorteio de questoes
  - mesmo evaluate_ragas.py (copia identica)

Uso:
    python src\rag_rrf.py -n 5
    python src\rag_rrf.py -r -n 5 -s 123
    python src\rag_rrf.py -n 52                        # rodada oficial
    python src\rag_rrf.py --indice-externo ..\avaliadorRAGAS-FAISS\output\faiss_embeddings.npy
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv(override=True)

try:  # garante UTF-8 no stdout mesmo com pipe/redirecionamento no Windows
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ------------------------------------------------------------------ caminhos
BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "data" / "corpus"
TRAIN_FILE = BASE_DIR / "data" / "train.json"
OUTPUT_DIR = BASE_DIR / "output"
PRED_FILE = OUTPUT_DIR / "predictions.json"
CACHE_VECTORS_PROPRIO = OUTPUT_DIR / "faiss_embeddings.npy"
CACHE_CHUNKS_PROPRIO = OUTPUT_DIR / "faiss_chunks.json"
TOC_FILE = CORPUS_DIR / "amm_toc_database.json"
# cache externo (projeto FAISS) — somente leitura
CACHE_EXTERNO_DEFAULT = Path(r"C:\Users\joser\Desktop\cursos\avaliadorRAGAS-FAISS\output")

# ------------------------------------------------------- config (fixa, para comparacao justa)
GERADOR_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"   # MESMO do projeto FAISS
# O gerador eh modelo de RACIOCINIO e o max_tokens limita (raciocinio + resposta)
# juntos. O raciocinio nao tem teto (medido: 2,3k-6,5k tokens/questao), entao com
# 4096 ele estourava o orcamento e a API devolvia finish_reason=length com content
# VAZIO -- foi a causa de 7 respostas vazias em 52 no projeto FAISS.
# Duas configuracoes foram testadas: thinking OFF zerava as vazias MAS fazia 2
# respostas boas virarem fallback; thinking ON + 16384 resolveu as duas coisas.
GERADOR_THINKING = True
GERADOR_MAX_TOKENS = 16384
LLM_TIMEOUT = 300
GERADOR_TENTATIVAS = 4
EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b"            # MESMO do projeto FAISS
NVIDIA_EMB_URL = "https://integrate.api.nvidia.com/v1/embeddings"
CHUNK_SIZE = 1000                                        # MESMO splitter do FAISS
CHUNK_OVERLAP = 200
TOP_K = 3                                                # MESMO orcamento de contexto
BATCH_SIZE = 100
RRF_K = 60              # constante do Reciprocal Rank Fusion


# ------------------------------------------------------------------ corpus (loader identico ao FAISS)
def carregar_documentos():
    docs = []
    for json_path in sorted(CORPUS_DIR.glob("*.json")):
        if json_path.name == TOC_FILE.name:
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            secoes = json.load(f)
        if not isinstance(secoes, list):
            continue
        for item in secoes:
            conteudo = item.get("content", "")
            if not conteudo:
                continue
            docs.append(Document(
                page_content=f"{item.get('title', '')}\n{conteudo}",
                metadata={"source": json_path.name, "task_id": item.get("task_id", "")},
            ))
    if not docs:
        raise FileNotFoundError(f"Nenhuma secao encontrada em {CORPUS_DIR}.")
    return docs


def gerar_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    return [c.page_content for c in chunks]


# ------------------------------------------------------------------ embeddings (API NVIDIA, com retry)
def _embed_lote(textos, input_type="passage", tentativas=4):
    key = os.environ.get("NVIDIA_API_KEY", "")
    if not key:
        raise RuntimeError("NVIDIA_API_KEY nao encontrada. Configure o arquivo .env.")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": EMBEDDING_MODEL, "input": textos, "input_type": input_type}
    delay = 3
    for tentativa in range(1, tentativas + 1):
        try:
            resp = requests.post(NVIDIA_EMB_URL, headers=headers, json=payload, timeout=180)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            dados = sorted(resp.json()["data"], key=lambda d: d.get("index", 0))
            return np.array([d["embedding"] for d in dados], dtype="float32")
        except Exception as e:
            if tentativa == tentativas:
                raise
            print(f"   falha no lote ({e}); tentativa {tentativa + 1}/{tentativas} em {delay}s...")
            time.sleep(delay)
            delay *= 2


def carregar_vetores_global(chunk_texts, indice_externo):
    """Vetores densos globais: tenta cache externo (FAISS); se nao bater, embedda tudo."""
    # 1) cache proprio
    if CACHE_VECTORS_PROPRIO.exists() and CACHE_CHUNKS_PROPRIO.exists():
        meta = json.loads(CACHE_CHUNKS_PROPRIO.read_text(encoding="utf-8"))
        if meta.get("chunks") == chunk_texts:
            print(f"   cache proprio reutilizado ({len(chunk_texts)} chunks).")
            return np.load(CACHE_VECTORS_PROPRIO)
    # 2) cache externo (projeto FAISS, somente leitura)
    if indice_externo:
        v_ext = indice_externo / "faiss_embeddings.npy"
        c_ext = indice_externo / "faiss_chunks.json"
        if v_ext.exists() and c_ext.exists():
            meta = json.loads(c_ext.read_text(encoding="utf-8"))
            if meta.get("chunks") == chunk_texts:
                print(f"   cache externo reutilizado ({len(chunk_texts)} chunks de {indice_externo}).")
                return np.load(v_ext)
            print("   cache externo nao bate com o corpus local (fingerprint diferente).")
    # 3) embedda tudo (caminho caro)
    print(f"   embeddando {len(chunk_texts)} chunks ({EMBEDDING_MODEL})... primeira execucao leva minutos.")
    partes = []
    for inicio in range(0, len(chunk_texts), BATCH_SIZE):
        partes.append(_embed_lote(chunk_texts[inicio:inicio + BATCH_SIZE], input_type="passage"))
        if (inicio // BATCH_SIZE) % 10 == 9:
            print(f"   ...{inicio + BATCH_SIZE}/{len(chunk_texts)} embeddados")
    vetores = np.vstack(partes)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(CACHE_VECTORS_PROPRIO, vetores)
    CACHE_CHUNKS_PROPRIO.write_text(
        json.dumps({"model": EMBEDDING_MODEL, "count": len(chunk_texts), "chunks": chunk_texts},
                   ensure_ascii=False), encoding="utf-8")
    print(f"   embeddings concluidos e salvos em {CACHE_VECTORS_PROPRIO}.")
    return vetores


# ------------------------------------------------------------------ RRF
def rrf_fundir(scores_sparse, scores_dense, k=RRF_K):
    """Funde dois rankings (arrays de scores) por RRF; devolve a ordem final."""
    ranks_sparse = np.argsort(np.argsort(-scores_sparse))
    ranks_dense = np.argsort(np.argsort(-scores_dense))
    fusao = 1.0 / (k + ranks_sparse) + 1.0 / (k + ranks_dense)
    return np.argsort(-fusao)


# ------------------------------------------------------------------ pipeline
PROMPT_TEMPLATE = PromptTemplate.from_template("""
Você é um especialista técnico de manutenção aeronáutica do Boeing 737 (AMM).
Com base estritamente no contexto recuperado dos manuais abaixo, responda à pergunta do mecânico de forma precisa e direta. Se a informação não estiver contida no contexto, declare que não possui informações suficientes no manual fornecido.

Contexto Técnico Recuperado:
{context}

Pergunta: {question}

**IMPORTANTE: Responda SEMPRE em português brasileiro técnico.**`r`n`r`nResposta Técnica:
""")


def _gerar_com_retry(llm, prompt, tentativas=GERADOR_TENTATIVAS):
    """Chama o LLM gerador com retry/backoff.

    Alem de timeout/erro de rede, trata RESPOSTA VAZIA como falha: com o thinking
    ligado o raciocinio pode consumir todo o orcamento de tokens e a API devolve
    finish_reason=length sem conteudo. Se persistir, devolve um marcador explicito
    em vez de derrubar a execucao inteira (o run de 52 questoes eh longo).
    """
    delay = 5
    for tentativa in range(1, tentativas + 1):
        try:
            conteudo = llm.invoke(prompt).content
            if conteudo and conteudo.strip():
                return conteudo
            falha = "resposta vazia (raciocinio estourou o orcamento de tokens)"
        except Exception as e:
            falha = f"{type(e).__name__}: {e}"
        if tentativa == tentativas:
            break
        print(f"   falha na geracao ({falha}); "
              f"tentativa {tentativa + 1}/{tentativas} em {delay}s...")
        time.sleep(delay)
        delay *= 2
    marcador = f"[GERACAO FALHOU: {falha} apos {tentativas} tentativas]"
    print(f"   [AVISO] {marcador}")
    return marcador


def executar_pipeline(limit=None, is_random=False, seed=None, indice_externo=None, max_chunks=None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Carregando secoes do manual em data/corpus/ (identico ao FAISS)...")
    docs = carregar_documentos()
    print(f"   {len(docs)} secoes.")

    print("2. Gerando chunks (splitter 1000/200 — igual ao FAISS)...")
    chunk_texts = gerar_chunks(docs)
    if max_chunks:
        chunk_texts = chunk_texts[:max_chunks]
        print(f"   MODO TESTE: {len(chunk_texts)} chunks.")
    print(f"   {len(chunk_texts)} chunks.")

    print("3. Ajustando TF-IDF sobre os chunks (ranking esparso)...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
    matriz_sparse = vectorizer.fit_transform(chunk_texts)
    print(f"   TF-IDF: {matriz_sparse.shape[0]} chunks x {matriz_sparse.shape[1]} termos ({time.time()-t0:.0f}s).")

    print("4. Carregando vetores densos globais (ranking denso)...")
    vetores = carregar_vetores_global(chunk_texts, indice_externo)
    V = np.ascontiguousarray(vetores, dtype="float32")
    V = V / np.linalg.norm(V, axis=1, keepdims=True)

    print(f"5. Inicializando o LLM gerador ({GERADOR_MODEL}, temp 0)...")
    # ATENCAO (comparabilidade): desde 24/10/2026 o gerador roda com thinking LIGADO
    # e orcamento de GERADOR_MAX_TOKENS tokens. Com 4096, respostas vinham VAZIAS
    # porque o raciocinio do modelo estourava o max_tokens (finish_reason=length,
    # content vazio). Aplique a MESMA configuracao nos projetos comparaveis
    # (TF-IDF, MISTURA, RRF, FAISS).
    llm = ChatNVIDIA(
        model=GERADOR_MODEL,
        temperature=0.0,
        max_completion_tokens=GERADOR_MAX_TOKENS,
        timeout=LLM_TIMEOUT,
    )
    if not GERADOR_THINKING:
        llm = llm.with_thinking_mode(False)

    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        perguntas_gt = json.load(f)

    if is_random and limit:
        if seed is None:
            seed = random.randrange(1, 1_000_000)
        random.seed(seed)
        perguntas_gt = random.sample(perguntas_gt, min(limit, len(perguntas_gt)))
        print(f"[SORTEIO] {len(perguntas_gt)} questões aleatórias (Seed {seed}).")
    elif limit:
        perguntas_gt = perguntas_gt[:limit]
        print(f"[RAPIDO] Primeiras {len(perguntas_gt)} questões.")

    predictions = []
    print("\n" + "=" * 80)
    print(" EXECUTANDO GERADOR RAG (RRF PURO: TF-IDF global + embedding global -> RRF -> LLM)")
    print("=" * 80)

    for i, item in enumerate(perguntas_gt, 1):
        pergunta = item.get("question")
        gabarito = item.get("ground_truth")

        sparse = (vectorizer.transform([pergunta]) @ matriz_sparse.T).toarray()[0]
        q = _embed_lote([pergunta], input_type="query")
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        denso = (q @ V.T)[0]

        ordem = rrf_fundir(sparse, denso)
        contextos = [chunk_texts[j] for j in ordem[:TOP_K] if 0 <= j < len(chunk_texts)]

        contexto_unificado = "\n\n--- CHUNK ---\n\n".join(contextos)
        prompt_formatado = PROMPT_TEMPLATE.format(context=contexto_unificado, question=pergunta)
        resposta_llm = _gerar_com_retry(llm, prompt_formatado)

        print(f"\n[Questao {i}/{len(perguntas_gt)}]")
        print(f"PERGUNTA: {pergunta}")
        print(f"GROUND TRUTH: {gabarito}")
        print(f"RESPOSTA GERADA: {resposta_llm}")
        print("-" * 80)

        predictions.append({
            "question_id": item.get("question_id"),
            "question": pergunta,
            "answer": resposta_llm,
            "contexts": contextos,
            "ground_truth": gabarito,
        })

    with open(PRED_FILE, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Execucao concluida. Respostas salvas em: {PRED_FILE}")
    print("   Proximo passo: python src\\evaluate_ragas.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline RAG RRF puro (TF-IDF global + embedding global)")
    parser.add_argument("-n", "--limit", type=int, default=None, help="Quantidade de perguntas")
    parser.add_argument("-r", "--random", action="store_true", help="Amostragem aleatória")
    parser.add_argument("-s", "--seed", type=int, default=None, help="Semente do sorteio")
    parser.add_argument("--indice-externo", type=Path, default=CACHE_EXTERNO_DEFAULT,
                        help="Pasta com faiss_embeddings.npy do projeto FAISS (somente leitura)")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="MODO TESTE: usa apenas os N primeiros chunks do índice")

    args = parser.parse_args()
    executar_pipeline(
        limit=args.limit,
        is_random=args.random,
        seed=args.seed,
        indice_externo=args.indice_externo,
        max_chunks=args.max_chunks,
    )
