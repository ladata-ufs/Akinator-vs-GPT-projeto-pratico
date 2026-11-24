#!/usr/bin/env python3
"""Avalia a robustez do Akinator Bayesiano a respostas erradas e 'nao_sei'.

Uso: python3 scripts/robustez.py [--repeticoes 30] [--semente 42]
Saída: results/robustez.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.akinator_engine import (  # noqa: E402
    atualizar_bayes, carregar_matriz, deve_parar, escolher_melhor_pergunta,
    inicializar_priors, ranking_top_k,
)

MATRIZ = REPO_ROOT / "data" / "processed" / "perguntas_matriz.json"
SAIDA = REPO_ROOT / "results" / "robustez.csv"


def resposta_ruidosa(pergunta, idx_doc, p_erro, p_nao_sei, rng):
    """Resposta do oráculo, com troca sim<->nao (p_erro) ou 'nao_sei' (p_nao_sei)."""
    resp = max(("sim", "nao", "nao_sei"), key=lambda r: pergunta["lik"][r][idx_doc])
    u = rng.random()
    if u < p_nao_sei:
        return "nao_sei"
    if resp in ("sim", "nao") and rng.random() < p_erro:
        return "nao" if resp == "sim" else "sim"
    return resp


def jogar(matriz, idx_alvo, max_perguntas, limiar, p_erro, p_nao_sei, rng):
    docentes, perguntas = matriz["docentes"], matriz["perguntas"]
    probs = inicializar_priors(len(docentes))
    feitas: set[int] = set()
    n = 0
    while not deve_parar(probs, n, max_perguntas=max_perguntas, limiar_confianca=limiar):
        q = escolher_melhor_pergunta(probs, perguntas, feitas)
        if q is None:
            break
        feitas.add(q)
        resp = resposta_ruidosa(perguntas[q], idx_alvo, p_erro, p_nao_sei, rng)
        probs = atualizar_bayes(probs, perguntas[q]["lik"][resp])
        n += 1
    top3 = [d["siape"] for d, _ in ranking_top_k(probs, docentes, 3)]
    alvo = docentes[idx_alvo]["siape"]
    return top3[0] == alvo, alvo in top3, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeticoes", type=int, default=30)
    ap.add_argument("--semente", type=int, default=42)
    args = ap.parse_args()

    matriz = carregar_matriz(MATRIZ)
    n_doc = len(matriz["docentes"])
    rng = random.Random(args.semente)
    cenarios = [("erro", p) for p in (0.0, 0.05, 0.10, 0.20)] + [
        ("nao_sei", p) for p in (0.10, 0.20, 0.30)
    ]
    modos = [("5_perguntas", 5, 0.85), ("livre", 12, 0.85)]

    linhas = []
    for modo, maxq, lim in modos:
        for tipo, p in cenarios:
            reps = 1 if p == 0.0 else args.repeticoes
            t1 = t3 = nq = tot = 0
            for idx in range(n_doc):
                for _ in range(reps):
                    pe, pn = (p, 0.0) if tipo == "erro" else (0.0, p)
                    a, b, n = jogar(matriz, idx, maxq, lim, pe, pn, rng)
                    t1 += a; t3 += b; nq += n; tot += 1
            linhas.append({
                "modo": modo, "tipo_ruido": tipo, "prob_ruido": p, "partidas": tot,
                "top1_pct": round(100 * t1 / tot, 1), "top3_pct": round(100 * t3 / tot, 1),
                "media_perguntas": round(nq / tot, 2),
            })
            print(linhas[-1])

    SAIDA.parent.mkdir(exist_ok=True)
    with SAIDA.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader(); w.writerows(linhas)
    print(f"Salvo em {SAIDA}")


if __name__ == "__main__":
    main()
