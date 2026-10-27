r"""Avaliacao RAGAS EM LOTES (robusta a 429/timeout do juiz).

Mesmas 4 metricas e mesmo juiz do evaluate_ragas.py, mas processando o
predictions.json em lotes menores com pausa entre eles e retry por lote.
Junta tudo no CSV final, no MESMO formato do evaluate_ragas.py.

O juiz vem da fabrica _criar_juiz() de evaluate_ragas.py (fonte unica: provedor,
orcamento, retry em resposta vazia/truncada). Trocar de juiz passa a ser so
JUIZ_PROVIDER=nvidia|groq no .env.

DIFERENCA IMPORTANTE em relacao a versao anterior: falhas do juiz NAO sao mais
preenchidas com 0.0 -- isso transformava falha em nota 0 e corrompia as medias.
Agora ficam como NaN, sao contadas no resumo e excluidas das medias.

Uso:
    python src\avaliar_lote.py                      # padrao: lote 13, pausa 45s
    python src\avaliar_lote.py --lote 10 --pausa 60
    python src\avaliar_lote.py --retomar            # pula lotes ja salvos
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
)
from metricas_ragas import answer_relevancy
from ragas.run_config import RunConfig

BASE_DIR = Path(__file__).resolve().parent.parent
PRED_FILE = BASE_DIR / "output" / "predictions.json"
CSV_OUTPUT = BASE_DIR / "output" / "resultado_ragas_metrics.csv"
DIR_LOTES = BASE_DIR / "output" / "lotes"

load_dotenv(BASE_DIR / ".env", override=True)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# fabrica do juiz (mesma do evaluate_ragas.py, para os dois entrarem em sincronia)
from evaluate_ragas import _criar_juiz, RAGAS_JOB_TIMEOUT  # noqa: E402

METRICAS = [faithfulness, answer_relevancy, context_precision, context_recall]
COLUNAS_METRICAS = ["faithfulness", "answer_relevancy",
                    "context_precision", "context_recall"]


def _df_vazio(n):
    """DataFrame de falha: NaN (nao 0) para nao contaminar as medias."""
    return pd.DataFrame({c: [float("nan")] * n for c in COLUNAS_METRICAS})


def avaliar_lote(lote, llm, emb):
    ds = Dataset.from_dict({
        "question": [x["question"] for x in lote],
        "contexts": [x["contexts"] for x in lote],
        "answer": [x["answer"] for x in lote],
        "ground_truth": [x.get("ground_truth", "") for x in lote],
    })
    # serial (max_workers=1): a cota free da NVIDIA (~40 req/min) estoura com os
    # 16 workers padrao do ragas; retry com backoff absorve 429 residuais.
    # timeout folgado: o padrao (180s) mata jobs lentos do juiz de raciocinio.
    rc = RunConfig(timeout=RAGAS_JOB_TIMEOUT, max_retries=10, max_wait=60, max_workers=1)
    res = evaluate(dataset=ds, metrics=METRICAS, llm=llm, embeddings=emb, run_config=rc)
    df = res.to_pandas()
    for col in COLUNAS_METRICAS:
        if col not in df.columns:
            df[col] = float("nan")
    # SEM fillna(0.0): falha do juiz continua NaN e eh contada no resumo
    return df[COLUNAS_METRICAS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(PRED_FILE))
    ap.add_argument("--out", default=str(CSV_OUTPUT))
    ap.add_argument("--lote", type=int, default=13)
    ap.add_argument("--pausa", type=int, default=45)
    ap.add_argument("--retomar", action="store_true", help="pula lotes ja salvos")
    args = ap.parse_args()

    dados = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    DIR_LOTES.mkdir(parents=True, exist_ok=True)
    n_lotes = (len(dados) + args.lote - 1) // args.lote
    print(f"1. {len(dados)} predicoes de {args.pred} — {n_lotes} lotes de {args.lote} "
          f"(pausa {args.pausa}s, retomar={args.retomar}).")

    llm = _criar_juiz()
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
    emb = NVIDIAEmbeddings(model="nvidia/nemotron-3-embed-1b")

    for inicio in range(0, len(dados), args.lote):
        lote = dados[inicio:inicio + args.lote]
        num = inicio // args.lote + 1
        arq = DIR_LOTES / f"lote_{num:02d}.csv"
        if args.retomar and arq.exists():
            print(f"2. Lote {num}/{n_lotes}: ja salvo, pulando.")
            continue
        print(f"2. Lote {num}/{n_lotes} ({len(lote)} questoes)...")
        for tentativa in range(1, 4):
            try:
                df = avaliar_lote(lote, llm, emb)
                df.to_csv(arq, index=False, encoding="utf-8-sig")
                print(f"   lote {num} OK e salvo ({arq.name}).")
                break
            except Exception as e:
                print(f"   falha no lote ({type(e).__name__}: {str(e)[:100]}); "
                      f"retry {tentativa}/3 em 60s...")
                time.sleep(60)
        else:
            print(f"   lote {num} falhou 3x — gravado como NaN (nao 0) para "
                  f"nao contaminar as medias.")
            _df_vazio(len(lote)).to_csv(arq, index=False, encoding="utf-8-sig")
        if inicio + args.lote < len(dados):
            print(f"   pausa de {args.pausa}s...")
            time.sleep(args.pausa)

    # junta os lotes (na ordem) e calcula as notas
    pedacos = []
    for inicio in range(0, len(dados), args.lote):
        num = inicio // args.lote + 1
        arq = DIR_LOTES / f"lote_{num:02d}.csv"
        if arq.exists():
            pedacos.append(pd.read_csv(arq))
        else:
            print(f"AVISO: lote {num} sem arquivo — completando com NaN.")
            pedacos.append(_df_vazio(args.lote))
    df = pd.concat(pedacos, ignore_index=True).head(len(dados))
    df.insert(0, "question", [x["question"] for x in dados])
    df.insert(1, "ground_truth", [x.get("ground_truth", "") for x in dados])
    df.insert(2, "answer", [x["answer"] for x in dados])
    for c in COLUNAS_METRICAS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in COLUNAS_METRICAS:
        df["nota_" + c] = df[c]
    # media por questao usando apenas as metricas validas
    df["total_nota_questao"] = df[COLUNAS_METRICAS].mean(axis=1, skipna=True)

    print("\n" + "#" * 78)
    print(" RESUMO (avaliacao em lotes)")
    print("#" * 78)
    for c in COLUNAS_METRICAS:
        col = "nota_" + c
        n_falha = int(df[col].isna().sum())
        n_zero = int((df[col] == 0).sum())
        print(f"  {col:32s} media={df[col].mean():.4f}  "
              f"mediana={df[col].median():.4f}  nota0={n_zero}  FALHOU(juiz)={n_falha}")
    n_falha_total = int(df["total_nota_questao"].isna().sum())
    print(f"  {'total_nota_questao':32s} media={df['total_nota_questao'].mean():.4f}  "
          f"FALHOU(juiz)={n_falha_total}")
    print(f"  respostas vazias: "
          f"{sum(1 for x in dados if not (x.get('answer') or '').strip())}")
    if n_falha_total:
        print("  -> medias calculadas sobre as questoes validas; rerode os lotes "
              "com falha (apague output/lotes/*.csv e use --retomar).")

    colunas = ["question", "ground_truth", "answer"] + ["nota_" + c for c in COLUNAS_METRICAS] + [
        "total_nota_questao"]
    df[[c for c in colunas if c in df.columns]].to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\nCSV salvo em: {args.out}")


if __name__ == "__main__":
    main()
