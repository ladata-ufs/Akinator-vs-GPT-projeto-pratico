#!/usr/bin/env python3
"""Compara o Akinator Bayesiano (5 perguntas / livre) com os resultados do agente LLM.

Lê `results/llm_resultados.json` e gera `results/comparacao.md` e `results/comparacao.csv`.
"""
from __future__ import annotations

import csv
import json
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from robustez import jogar  # noqa: E402
from src.akinator_engine import carregar_matriz  # noqa: E402
import random  # noqa: E402

R = REPO_ROOT / "results"


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "").encode("ascii", "ignore").decode().lower()
    return " ".join(s.split())


def acertou(real: str, palpite: str) -> bool:
    """Acerta se o primeiro e o último nome do docente aparecem no palpite."""
    partes = norm(real).split()
    p = norm(palpite)
    return bool(partes) and partes[0] in p and partes[-1] in p


def main() -> None:
    matriz = carregar_matriz(REPO_ROOT / "processed" / "perguntas_matriz.json")
    docs = matriz["docentes"]
    llm = json.loads((R / "llm_resultados.json").read_text(encoding="utf-8"))
    rng = random.Random(0)
    linhas = []
    for item in llm:
        idx = next(i for i, d in enumerate(docs) if d["id"] == item["id"])
        b5 = jogar(matriz, idx, 5, 0.85, 0.0, 0.0, rng)
        bl = jogar(matriz, idx, 12, 0.85, 0.0, 0.0, rng)
        linhas.append({
            "docente": docs[idx]["nome"],
            "bayes5_top1": b5[0], "bayes5_top3": b5[1],
            "bayes_livre_top1": bl[0], "bayes_livre_n": bl[2],
            "llm_acertou": acertou(docs[idx]["nome"], item.get("palpite", "")),
            "llm_dicas": item.get("dicas_usadas"), "llm_palpite": item.get("palpite", ""),
        })
    n = len(linhas)
    soma = lambda k: sum(1 for l in linhas if l[k])  # noqa: E731
    md = ["# Comparação Bayes × LLM (amostra de %d docentes)\n" % n,
          "| Método | Acertos Top-1 | % |", "|---|---|---|"]
    for nome, k in [("Bayes 5 perguntas", "bayes5_top1"), ("Bayes livre (<=12)", "bayes_livre_top1"),
                    ("LLM (5 dicas)", "llm_acertou")]:
        md.append(f"| {nome} | {soma(k)}/{n} | {100*soma(k)/n:.1f}% |")
    md.append(f"\nBayes 5 perguntas, Top-3: {soma('bayes5_top3')}/{n}.\n")
    md += ["## Por docente\n", "| Docente | Bayes5 | Bayes livre (nº perg.) | LLM | Palpite do LLM |", "|---|---|---|---|---|"]
    ok = lambda b: "✅" if b else "❌"  # noqa: E731
    for l in linhas:
        md.append(f"| {l['docente']} | {ok(l['bayes5_top1'])} | {ok(l['bayes_livre_top1'])} ({l['bayes_livre_n']}) | "
                  f"{ok(l['llm_acertou'])} ({l['llm_dicas']} dicas) | {l['llm_palpite']} |")
    md.append("\n> Ressalva: as 5 dicas foram geradas pelo próprio motor Bayesiano (mesma informação das 5 "
              "primeiras perguntas), o que pode favorecer o Bayes. A amostra inclui os 6 docentes mais difíceis para ele.")
    (R / "comparacao.md").write_text("\n".join(md), encoding="utf-8")
    with (R / "comparacao.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0])); w.writeheader(); w.writerows(linhas)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        nomes = ["Bayes 5", "Bayes livre", "LLM 5 dicas"]
        vals = [100 * soma(k) / n for k in ("bayes5_top1", "bayes_livre_top1", "llm_acertou")]
        plt.bar(nomes, vals, color=["#3b82f6", "#22c55e", "#f59e0b"])
        plt.ylabel("Top-1 (%)"); plt.ylim(0, 100); plt.title(f"Acurácia na amostra (n={n})")
        plt.savefig(R / "comparacao.png", dpi=150, bbox_inches="tight")
    except ImportError:
        print("matplotlib ausente: gráfico não gerado")
    print("\n".join(md))


if __name__ == "__main__":
    main()
