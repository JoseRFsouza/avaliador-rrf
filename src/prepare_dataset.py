import json
import os
from pathlib import Path

# Caminhos
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_GT = BASE_DIR / "ground_truth_approved.json"
OUTPUT_TRAIN = BASE_DIR / "data" / "train.json"

def converter_ground_truth():
    os.makedirs(OUTPUT_TRAIN.parent, exist_ok=True)
    
    if not INPUT_GT.exists():
        print(f"Erro: Arquivo {INPUT_GT} não encontrado!")
        return

    with open(INPUT_GT, "r", encoding="utf-8") as f:
        orig_data = json.load(f)

    train_data = []
    for item in orig_data:
        source_file = item.get("source_file")
        source_pdf = f"{source_file}.pdf" if source_file else None
        
        train_data.append({
            "question_id": item.get("question_id"),
            "question": item.get("question"),
            "ground_truth": item.get("ground_truth"),
            "source_file": source_pdf,
            "expected_facts": item.get("expected_facts", []),
            "source_task_code": item.get("source_task_code", "")
        })

    with open(OUTPUT_TRAIN, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Arquivo {OUTPUT_TRAIN} gerado com sucesso! ({len(train_data)} itens)")

if __name__ == "__main__":
    converter_ground_truth()