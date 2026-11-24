"""Motor Bayesiano e seleção dinâmica de perguntas por Índice de Gini."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RESPOSTAS_VALIDAS = ("sim", "nao", "nao_sei")


@dataclass
class AkinatorState:
    """Estado de uma partida do Akinator Bayesiano."""

    probs: list[float]
    ja_feitas: set[int] = field(default_factory=set)
    historico: list[dict[str, Any]] = field(default_factory=list)


def carregar_matriz(json_path: Path) -> dict[str, Any]:
    """Carrega o arquivo processed/perguntas_matriz.json."""
    return json.loads(json_path.read_text(encoding="utf-8"))


def ranking_top_k(
    probs: list[float],
    docentes: list[dict[str, Any]],
    k: int = 3,
) -> list[tuple[dict[str, Any], float]]:
    """Retorna os `k` docentes mais prováveis em ordem decrescente de probabilidade."""
    pares = list(zip(docentes, probs))
    pares.sort(key=lambda item: item[1], reverse=True)
    return pares[:k]


def inicializar_priors(n_docentes: int) -> list[float]:
    """Retorna a distribuição a priori uniforme [1/N, ..., 1/N]."""
    if n_docentes <= 0:
        raise ValueError("n_docentes deve ser positivo")

    p_inicial = 1 / n_docentes
    return [p_inicial for _ in range(n_docentes)]


def calcular_gini(probs: list[float]) -> float:
    """Calcula a impureza de Gini da distribuição atual: 1 - sum(p_i ** 2)."""
    return 1 - sum(p ** 2 for p in probs)


def atualizar_bayes(probs: list[float], lik_resposta: list[float]) -> list[float]:
    """Atualiza a distribuição a posteriori via Regra de Bayes."""
    produtos = [p * l for p, l in zip(probs, lik_resposta)]
    soma = sum(produtos)

    if soma == 0:
        return list(probs)

    return [x / soma for x in produtos]


def escolher_melhor_pergunta(
    probs: list[float],
    perguntas: list[dict[str, Any]],
    ja_feitas: set[int],
    *,
    usar_prioridade_aluno: bool = False,
    top_k_aleatorio: int = 1,
    margem_empate: float = 0.0,
    rng: random.Random | None = None,
) -> int | None:
    """Seleciona a próxima pergunta que minimiza o Índice de Gini esperado.

    Com `top_k_aleatorio=1` e `margem_empate=0.0`, a seleção é determinística.
    Valores maiores permitem sortear entre as melhores perguntas mantendo a eficácia.
    """
    candidatos: list[tuple[float, int]] = []

    for idx, q in enumerate(perguntas):
        if idx in ja_feitas:
            continue

        gini_esperado = 0.0
        for resp in RESPOSTAS_VALIDAS:
            lik = q["lik"][resp]
            p_resp = sum(p * l for p, l in zip(probs, lik))
            if p_resp <= 0:
                continue
            posterior = atualizar_bayes(probs, lik)
            gini_esperado += p_resp * calcular_gini(posterior)

        prioridade = q.get("prioridade_aluno", 1.0) if usar_prioridade_aluno else 1.0
        score = gini_esperado / prioridade if prioridade > 0 else gini_esperado
        candidatos.append((score, idx))

    if not candidatos:
        return None

    candidatos.sort(key=lambda item: (item[0], item[1]))
    if top_k_aleatorio <= 1 and margem_empate <= 0.0:
        return candidatos[0][1]

    melhor_score = candidatos[0][0]
    if margem_empate > 0.0:
        elegiveis = [idx for s, idx in candidatos if s <= melhor_score + margem_empate]
        if top_k_aleatorio > 1:
            elegiveis = elegiveis[:top_k_aleatorio]
    else:
        elegiveis = [idx for _, idx in candidatos[:top_k_aleatorio]]

    escolhedor = rng if rng is not None else random
    return escolhedor.choice(elegiveis)


def deve_parar(
    probs: list[float],
    num_perguntas: int,
    *,
    max_perguntas: int = 5,
    limiar_confianca: float = 0.85,
) -> bool:
    """Retorna True se atingir `max_perguntas` ou `limiar_confianca`."""
    if num_perguntas >= max_perguntas:
        return True
    return bool(probs) and max(probs) >= limiar_confianca
