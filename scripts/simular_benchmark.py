#!/usr/bin/env python3
"""Benchmark do Akinator Bayesiano nos 33 docentes do DCOMP/UFS.

Avalia dois modos:
1. Comparativo (`max_perguntas=5`): mesma restrição de 5 interações da abordagem LLM.
2. Convergência livre (`max_perguntas=12`, `limiar_confianca=0.85`).

Uso:
    python3 scripts/simular_benchmark.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.akinator_engine import (  # noqa: E402
    RESPOSTAS_VALIDAS,
    atualizar_bayes,
    carregar_matriz,
    deve_parar,
    escolher_melhor_pergunta,
    inicializar_priors,
    ranking_top_k,
)

DEFAULT_MATRIZ = REPO_ROOT / "data" / "processed" / "perguntas_matriz.json"


def resposta_oraculo(pergunta: dict[str, Any], idx_docente: int) -> str:
    """Simula um jogador perfeito escolhendo a resposta mais provável para o docente."""
    return max(RESPOSTAS_VALIDAS, key=lambda r: pergunta["lik"][r][idx_docente])


def simular_modo(
    matriz: dict[str, Any],
    *,
    max_perguntas: int,
    limiar_confianca: float,
    usar_prioridade_aluno: bool = False,
) -> dict[str, Any]:
    docentes = matriz["docentes"]
    perguntas = matriz["perguntas"]
    n = len(docentes)

    acertos_top1 = 0
    acertos_top3 = 0
    soma_perguntas = 0
    soma_confianca = 0.0
    erros: list[dict[str, Any]] = []

    for idx_alvo, alvo in enumerate(docentes):
        probs = inicializar_priors(n)
        ja_feitas: set[int] = set()
        perguntas_feitas = 0

        while not deve_parar(
            probs,
            perguntas_feitas,
            max_perguntas=max_perguntas,
            limiar_confianca=limiar_confianca,
        ):
            q_idx = escolher_melhor_pergunta(
                probs,
                perguntas,
                ja_feitas,
                usar_prioridade_aluno=usar_prioridade_aluno,
            )
            if q_idx is None:
                break
            ja_feitas.add(q_idx)
            resp = resposta_oraculo(perguntas[q_idx], idx_alvo)
            probs = atualizar_bayes(probs, perguntas[q_idx]["lik"][resp])
            perguntas_feitas += 1

        top3 = ranking_top_k(probs, docentes, k=3)
        top1_doc, top1_prob = top3[0]
        top3_siapes = {d["siape"] for d, _ in top3}

        soma_perguntas += perguntas_feitas
        soma_confianca += top1_prob

        if top1_doc["siape"] == alvo["siape"]:
            acertos_top1 += 1
        else:
            erros.append(
                {
                    "alvo": alvo["nome"],
                    "chute_top1": top1_doc["nome"],
                    "prob_top1": round(top1_prob, 4),
                    "no_top3": alvo["siape"] in top3_siapes,
                    "perguntas": perguntas_feitas,
                }
            )

        if alvo["siape"] in top3_siapes:
            acertos_top3 += 1

    return {
        "total_docentes": n,
        "max_perguntas": max_perguntas,
        "limiar_confianca": limiar_confianca,
        "top1": acertos_top1,
        "top1_pct": round(100.0 * acertos_top1 / n, 1),
        "top3": acertos_top3,
        "top3_pct": round(100.0 * acertos_top3 / n, 1),
        "media_perguntas": round(soma_perguntas / n, 2),
        "media_confianca": round(100.0 * soma_confianca / n, 1),
        "erros": erros,
    }


def imprimir_relatorio(titulo: str, res: dict[str, Any]) -> None:
    print(f"\n=== {titulo} ===")
    print(
        f"Top-1: {res['top1']}/{res['total_docentes']} ({res['top1_pct']}%) | "
        f"Top-3: {res['top3']}/{res['total_docentes']} ({res['top3_pct']}%)"
    )
    print(
        f"Média de perguntas: {res['media_perguntas']} | "
        f"Confiança média no chute: {res['media_confianca']}%"
    )
    if res["erros"]:
        print("Docentes fora do Top-1:")
        for e in res["erros"]:
            tag = "no Top-3" if e["no_top3"] else "fora do Top-3"
            print(
                f"  - Alvo: {e['alvo']} -> Chute: {e['chute_top1']} "
                f"({e['prob_top1']*100:.1f}%, {tag}, {e['perguntas']} perg.)"
            )
    else:
        print("100% dos docentes identificados corretamente em Top-1!")


def main() -> None:
    if not DEFAULT_MATRIZ.exists():
        raise SystemExit(
            f"Arquivo {DEFAULT_MATRIZ} não encontrado. "
            "Rode primeiro: python3 scripts/build_likelihood_matrix.py"
        )

    matriz = carregar_matriz(DEFAULT_MATRIZ)
    res_5 = simular_modo(matriz, max_perguntas=5, limiar_confianca=0.85)
    res_livre = simular_modo(matriz, max_perguntas=12, limiar_confianca=0.85)

    imprimir_relatorio("MODO COMPARATIVO (máx. 5 perguntas — igual à LLM)", res_5)
    imprimir_relatorio("MODO CONVERGÊNCIA LIVRE (máx. 12 perguntas, limiar 85%)", res_livre)


if __name__ == "__main__":
    main()
