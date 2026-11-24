#!/usr/bin/env python3
"""CLI interativa para testar o Akinator Bayesiano DCOMP/UFS no terminal.

Uso:
    python3 scripts/jogar_cli.py --modo 5
    python3 scripts/jogar_cli.py --modo livre
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.akinator_engine import (  # noqa: E402
    atualizar_bayes,
    calcular_gini,
    carregar_matriz,
    deve_parar,
    escolher_melhor_pergunta,
    inicializar_priors,
    ranking_top_k,
)

DEFAULT_MATRIZ = REPO_ROOT / "data" / "processed" / "perguntas_matriz.json"

MAPA_ENTRADA = {
    "s": "sim",
    "sim": "sim",
    "1": "sim",
    "n": "nao",
    "nao": "nao",
    "não": "nao",
    "2": "nao",
    "?": "nao_sei",
    "ns": "nao_sei",
    "nao sei": "nao_sei",
    "não sei": "nao_sei",
    "nao_sei": "nao_sei",
    "3": "nao_sei",
}


def ler_resposta() -> str:
    while True:
        bruto = input("Sua resposta [1=Sim | 2=Não | 3=Não Sei]: ").strip().lower()
        if bruto in MAPA_ENTRADA:
            return MAPA_ENTRADA[bruto]
        print("Entrada inválida. Digite 1 (Sim), 2 (Não) ou 3 (Não Sei).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Jogue o Akinator Bayesiano DCOMP/UFS no terminal.")
    parser.add_argument(
        "--modo",
        choices=("5", "livre"),
        default="5",
        help="'5' = limite de 5 perguntas (comparação LLM); 'livre' = até 12 perguntas / 85%%.",
    )
    parser.add_argument(
        "--deterministico",
        action="store_true",
        help="Desativa o sorteio Top-3 de Gini na 1ª pergunta e desempates próximos.",
    )
    args = parser.parse_args()

    if not DEFAULT_MATRIZ.exists():
        raise SystemExit(
            f"Arquivo {DEFAULT_MATRIZ} não encontrado. "
            "Rode primeiro: python3 scripts/build_likelihood_matrix.py"
        )

    matriz = carregar_matriz(DEFAULT_MATRIZ)
    docentes = matriz["docentes"]
    perguntas = matriz["perguntas"]

    max_perguntas = 5 if args.modo == "5" else 12
    probs = inicializar_priors(len(docentes))
    ja_feitas: set[int] = set()
    passo = 0

    print("\n=== Akinator Bayesiano DCOMP/UFS ===")
    print(f"Pense em um(a) dos {len(docentes)} docentes do DCOMP/UFS! (Modo: {args.modo})\n")

    while not deve_parar(probs, passo, max_perguntas=max_perguntas, limiar_confianca=0.85):
        if args.deterministico:
            q_idx = escolher_melhor_pergunta(probs, perguntas, ja_feitas)
        elif passo == 0:
            q_idx = escolher_melhor_pergunta(probs, perguntas, ja_feitas, top_k_aleatorio=3)
        else:
            q_idx = escolher_melhor_pergunta(probs, perguntas, ja_feitas, margem_empate=0.002)
        if q_idx is None:
            break
        ja_feitas.add(q_idx)
        passo += 1

        q = perguntas[q_idx]
        print(f"Pergunta {passo}/{max_perguntas} [{q['categoria']}]: {q['texto']}")
        resp = ler_resposta()
        probs = atualizar_bayes(probs, q["lik"][resp])

        top3 = ranking_top_k(probs, docentes, k=3)
        gini_atual = calcular_gini(probs)
        print(f"  -> Gini atual: {gini_atual:.4f} | Top-3:")
        for pos, (doc, prob) in enumerate(top3, start=1):
            print(f"     {pos}. {doc['nome']} ({prob * 100:.1f}%)")
        print()

    top3 = ranking_top_k(probs, docentes, k=3)
    vencedor, conf = top3[0]
    print(f"=== PALPITE FINAL (após {passo} perguntas) ===")
    print(f"Você pensou em: {vencedor['nome']} ({conf * 100:.1f}% de confiança)!")
    if len(top3) > 1:
        print("Outras possibilidades no Top-3:")
        for doc, prob in top3[1:]:
            print(f"  - {doc['nome']} ({prob * 100:.1f}%)")


if __name__ == "__main__":
    main()
