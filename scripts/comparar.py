#!/usr/bin/env python3
"""Compara o Akinator Bayesiano (5 perguntas / livre) com os resultados do agente LLM.

Lê `results/llm_resultados.json` e gera `results/comparacao.csv` e `results/comparacao.svg`.
"""
from __future__ import annotations

import csv
import json
import random
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from robustez import jogar  # noqa: E402
from src.akinator_engine import carregar_matriz  # noqa: E402

R = REPO_ROOT / "results"


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "").encode("ascii", "ignore").decode().lower()
    return " ".join(s.split())


def acertou(real: str, palpite: str) -> bool:
    """Acerta se o primeiro e o último nome do docente aparecem no palpite."""
    partes = norm(real).split()
    p = norm(palpite)
    return bool(partes) and partes[0] in p and partes[-1] in p


def gerar_svg(n: int, b5: float, b5_t3: float, bl: float, llm: float, destino: Path) -> None:
    bars = [
        ("Bayes 5 perg. (Top-1)", b5, "#3b82f6"),
        ("Bayes 5 perg. (Top-3)", b5_t3, "#60a5fa"),
        ("Bayes livre <=12 (Top-1)", bl, "#22c55e"),
        ("LLM n8n+RAG 5 dicas", llm, "#f59e0b"),
    ]
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="680" height="270" style="background:#0f172a;font-family:system-ui,sans-serif">',
        f'<text x="20" y="32" fill="#f8fafc" font-size="16" font-weight="bold">Akinator Bayesiano vs LLM (n8n + GPT-5-mini + RAG) — Amostra n={n}</text>',
        '<text x="20" y="52" fill="#94a3b8" font-size="11">Amostra inclui os 6 docentes mais difíceis para o Bayes em 5 perguntas + 4 aleatórios</text>',
    ]
    for i, (label, val, col) in enumerate(bars):
        y = 80 + i * 44
        w = max(4, int(val * 3.8))
        lines.append(f'<text x="20" y="{y+18}" fill="#e2e8f0" font-size="13">{label}</text>')
        lines.append(f'<rect x="210" y="{y+4}" width="380" height="20" rx="4" fill="#1e293b"/>')
        lines.append(f'<rect x="210" y="{y+4}" width="{w}" height="20" rx="4" fill="{col}"/>')
        lines.append(f'<text x="602" y="{y+19}" fill="#f8fafc" font-size="13" font-weight="bold">{val:.1f}%</text>')
    lines.append("</svg>")
    destino.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    matriz = carregar_matriz(REPO_ROOT / "data" / "processed" / "perguntas_matriz.json")
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
            "bayes5_top1": b5[0],
            "bayes5_top3": b5[1],
            "bayes_livre_top1": bl[0],
            "bayes_livre_n": bl[2],
            "llm_acertou": acertou(docs[idx]["nome"], item.get("palpite", "")),
            "llm_dicas": item.get("dicas_usadas"),
            "llm_palpite": item.get("palpite", ""),
        })

    n = len(linhas)
    pct = lambda k: 100.0 * sum(1 for l in linhas if l[k]) / n  # noqa: E731
    with (R / "comparacao.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    gerar_svg(n, pct("bayes5_top1"), pct("bayes5_top3"), pct("bayes_livre_top1"), pct("llm_acertou"), R / "comparacao.svg")

    print(f"Amostra n={n} | Bayes5 Top-1: {pct('bayes5_top1'):.1f}% (Top-3: {pct('bayes5_top3'):.1f}%) | "
          f"Bayes livre: {pct('bayes_livre_top1'):.1f}% | LLM: {pct('llm_acertou'):.1f}%")
    print(f"Salvo em {R / 'comparacao.csv'} e {R / 'comparacao.svg'}")


if __name__ == "__main__":
    main()
