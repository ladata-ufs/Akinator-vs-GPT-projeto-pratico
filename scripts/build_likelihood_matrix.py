#!/usr/bin/env python3
"""Gera a matriz de verossimilhança P(E | H) em processed/perguntas_matriz.json
a partir dos dados brutos de docentes e disciplinas em raw/.
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DOCENTES = REPO_ROOT / "raw" / "docentes_dcomp_ufs.csv"
RAW_DISCIPLINAS = REPO_ROOT / "raw" / "disciplinas_dcomp_ufs.csv"
OUT_MATRIZ = REPO_ROOT / "processed" / "perguntas_matriz.json"
OUT_WEB_JS = REPO_ROOT / "web" / "matriz.js"

RESPOSTAS_VALIDAS = ("sim", "nao", "nao_sei")


def normalizar_texto(texto: str) -> str:
    """Remove acentos, converte para minúsculas e normaliza espaços."""
    sem_acento = unicodedata.normalize("NFD", texto or "")
    sem_acento = "".join(ch for ch in sem_acento if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", sem_acento).strip().lower()


def carregar_dados_brutos(
    docentes_csv: Path = RAW_DOCENTES,
    disciplinas_csv: Path = RAW_DISCIPLINAS,
) -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]]]:
    """Carrega os 33 docentes e agrupa as disciplinas ministradas por SIAPE."""
    with docentes_csv.open(encoding="utf-8", newline="") as f:
        docentes = list(csv.DictReader(f))

    disciplinas_por_siape: dict[str, list[dict[str, str]]] = {d["siape"]: [] for d in docentes}
    with disciplinas_csv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            siape = row["siape"]
            if siape in disciplinas_por_siape:
                disciplinas_por_siape[siape].append(row)

    return docentes, disciplinas_por_siape


def avaliar_disciplina(
    disciplinas_docente: list[dict[str, str]],
    termos: tuple[str, ...],
    *,
    apenas_graduacao: bool = True,
) -> str:
    """Classifica se o docente ministra disciplinas compatíveis com `termos`.

    Retorna:
    - "positivo": ministrou recentemente (2024–2026) OU >= 3 ofertas históricas.
    - "parcial": ministrou apenas 1–2 vezes antes de 2024.
    - "negativo": nunca ministrou disciplina com esses termos.
    """
    termos_norm = tuple(normalizar_texto(t) for t in termos)
    ofertas_total = 0
    recente = False

    for disc in disciplinas_docente:
        if apenas_graduacao and disc.get("nivel") != "graduacao":
            continue
        nome_norm = normalizar_texto(disc.get("nome_disciplina", ""))
        if any(t in nome_norm for t in termos_norm):
            ofertas_total += int(disc.get("qtd_ofertas") or 1)
            sem = disc.get("ultimo_semestre", "")
            if sem.startswith(("2024.", "2025.", "2026.")):
                recente = True

    if recente or ofertas_total >= 3:
        return "positivo"
    if ofertas_total > 0:
        return "parcial"
    return "negativo"


def avaliar_area_interesse(docente: dict[str, str], termos: tuple[str, ...]) -> str:
    """Verifica se algum termo aparece em `areas_interesse_consolidadas` ou descrição."""
    areas = normalizar_texto(docente.get("areas_interesse_consolidadas", ""))
    desc = normalizar_texto(docente.get("descricao_atuacao_sigaa", ""))
    termos_norm = tuple(normalizar_texto(t) for t in termos)

    if any(t in areas for t in termos_norm):
        return "positivo"
    if any(t in desc for t in termos_norm):
        return "parcial"
    return "negativo"


def avaliar_booleano_csv(docente: dict[str, str], coluna: str) -> str:
    """Converte coluna booleana do CSV ('True', 'False', '') para estado semântico."""
    val = (docente.get(coluna) or "").strip()
    if val == "True":
        return "positivo"
    if val == "False":
        return "negativo"
    return "desconhecido"



def classificar_estado_para_probs(estado: str) -> tuple[float, float, float]:
    """Converte um estado semântico na tripla suavizada (P(sim), P(nao), P(nao_sei))."""

    tabela = {
        "positivo": (0.89, 0.05, 0.06),
        "negativo": (0.05, 0.89, 0.06),
        "parcial": (0.62, 0.24, 0.14),
        "desconhecido": (0.34, 0.33, 0.33),
    }
    if estado not in tabela:
        raise ValueError(f"Estado invalido: {estado!r}")

    return tabela[estado]


@dataclass(frozen=True)
class RegraPergunta:
    id: str
    categoria: str
    texto: str
    prioridade_aluno: float
    avaliador: Callable[[dict[str, str], list[dict[str, str]]], str]


def montar_catalogo_perguntas() -> list[RegraPergunta]:
    """Retorna o catálogo de perguntas que compõem a matriz P(E | H)."""
    perguntas: list[RegraPergunta] = [
        RegraPergunta(
            id="disc_poo",
            categoria="disciplinas",
            texto="Esse(a) docente costuma ministrar Programação Orientada a Objetos (POO)?",
            prioridade_aluno=1.0,
            avaliador=lambda doc, discs: avaliar_disciplina(
                discs, ("programacao orientada a objetos",)
            ),
        ),
        
        RegraPergunta(
            id="atua_pos_graduacao",
            categoria="pos_graduacao",
            texto="Esse(a) docente atua na Pós-Graduação (mestrado/doutorado)?",
            prioridade_aluno=1.0,
            avaliador=lambda doc, discs: avaliar_booleano_csv(doc, "atua_pos_graduacao"),
        ),
        # Bloco A — Disciplinas Ministradas
        RegraPergunta(
            id="disc_prog_imperativa",
            categoria="disciplinas",
            texto="Esse(a) docente costuma ministrar Programação Imperativa ou Programação A/B/C?",
            prioridade_aluno=1.0,
            avaliador=lambda doc, discs: avaliar_disciplina(
                discs,
                ("programacao imperativa", "programacao a", "programacao b", "programacao c"),
            ),
        ),
        RegraPergunta(
            id="disc_ed_grafos",
            categoria="disciplinas",
            texto="Esse(a) docente costuma ministrar Estruturas de Dados ou Grafos?",
            prioridade_aluno=1.0,
            avaliador=lambda doc, discs: avaliar_disciplina(
                discs, ("estrutura de dados", "estruturas de dados", "grafos")
            ),
        ),
        RegraPergunta(
            id="disc_redes_sd",
            categoria="disciplinas",
            texto="Esse(a) docente costuma ministrar Redes de Computadores ou Sistemas Distribuídos?",
            prioridade_aluno=1.0,
            avaliador=lambda doc, discs: avaliar_disciplina(
                discs, ("redes de computadores", "sistemas distribuidos", "computacao em nuvem")
            ),
        ),
        RegraPergunta(
            id="disc_arquitetura_hardware",
            categoria="disciplinas",
            texto="Esse(a) docente costuma ministrar Arquitetura de Computadores, Sistemas Digitais ou Embarcados?",
            prioridade_aluno=1.0,
            avaliador=lambda doc, discs: avaliar_disciplina(
                discs,
                ("arquitetura de computadores", "sistemas digitais", "sistemas embarcados", "hardware"),
            ),
        ),
        RegraPergunta(
            id="disc_eng_software",
            categoria="disciplinas",
            texto="Esse(a) docente costuma ministrar Engenharia, Qualidade ou Teste de Software?",
            prioridade_aluno=1.0,
            avaliador=lambda doc, discs: avaliar_disciplina(
                discs,
                ("engenharia de software", "qualidade de software", "teste de software", "evolucao de software"),
            ),
        ),
        RegraPergunta(
            id="disc_ia_ml",
            categoria="disciplinas",
            texto="Esse(a) docente costuma ministrar Inteligência Artificial ou Aprendizagem de Máquina?",
            prioridade_aluno=1.0,
            avaliador=lambda doc, discs: avaliar_disciplina(
                discs,
                ("inteligencia artificial", "aprendizagem de maquina", "linguagem natural", "redes neurais"),
            ),
        ),
        RegraPergunta(
            id="disc_imagens_cg",
            categoria="disciplinas",
            texto="Esse(a) docente costuma ministrar Processamento de Imagens ou Computação Gráfica?",
            prioridade_aluno=1.0,
            avaliador=lambda doc, discs: avaliar_disciplina(
                discs, ("processamento de imagens", "computacao grafica")
            ),
        ),
        RegraPergunta(
            id="disc_compiladores_lfc",
            categoria="disciplinas",
            texto="Esse(a) docente costuma ministrar Compiladores, Linguagens Formais ou Programação Funcional?",
            prioridade_aluno=1.0,
            avaliador=lambda doc, discs: avaliar_disciplina(
                discs,
                ("compiladores", "linguagens formais", "teoria da computacao", "programacao funcional"),
            ),
        ),
        RegraPergunta(
            id="disc_bd_dados",
            categoria="disciplinas",
            texto="Esse(a) docente costuma ministrar Banco de Dados, Engenharia de Dados ou Sistemas de Apoio à Decisão?",
            prioridade_aluno=1.0,
            avaliador=lambda doc, discs: avaliar_disciplina(
                discs, ("banco de dados", "engenharia de dados", "apoio a decisao")
            ),
        ),
        RegraPergunta(
            id="disc_so_concorrente",
            categoria="disciplinas",
            texto="Esse(a) docente costuma ministrar Sistemas Operacionais ou Programação Paralela e Concorrente?",
            prioridade_aluno=1.0,
            avaliador=lambda doc, discs: avaliar_disciplina(
                discs, ("sistemas operacionais", "programacao paralela")
            ),
        ),
        RegraPergunta(
            id="disc_etica_sociedade",
            categoria="disciplinas",
            texto="Esse(a) docente costuma ministrar Informática, Ética e Sociedade?",
            prioridade_aluno=1.0,
            avaliador=lambda doc, discs: avaliar_disciplina(
                discs, ("etica e sociedade",)
            ),
        ),
        # Bloco B — Áreas de Atuação
        RegraPergunta(
            id="area_educacao_ihc",
            categoria="areas",
            texto="Esse(a) docente atua com Informática na Educação, Jogos Educativos ou IHC?",
            prioridade_aluno=1.0,
            avaliador=lambda doc, discs: avaliar_area_interesse(
                doc, ("educacao", "educativa", "jogos", "interface humano", "interacao humano")
            ),
        ),
        RegraPergunta(
            id="area_otimizacao_algoritmos",
            categoria="areas",
            texto="Esse(a) docente pesquisa Otimização, Metaheurísticas ou Teoria/Algoritmos?",
            prioridade_aluno=1.0,
            avaliador=lambda doc, discs: avaliar_area_interesse(
                doc, ("otimizacao", "metaheuristica", "algoritmos", "grafos", "teoria")
            ),
        ),
        RegraPergunta(
            id="area_seguranca_iot",
            categoria="areas",
            texto="Esse(a) docente atua com Segurança, IoT/Sensores ou Computação em Nuvem?",
            prioridade_aluno=1.0,
            avaliador=lambda doc, discs: avaliar_area_interesse(
                doc, ("seguranca", "criptografia", "internet das coisas", "iot", "sensores", "nuvem")
            ),
        ),
        RegraPergunta(
            id="area_saude_bio",
            categoria="areas",
            texto="Esse(a) docente trabalha com aplicações em Saúde Digital ou Bioinformática?",
            prioridade_aluno=1.0,
            avaliador=lambda doc, discs: avaliar_area_interesse(
                doc, ("saude", "bioinformatica", "biologia", "medica", "e-health")
            ),
        ),
        # Bloco C — Pós-Graduação
        RegraPergunta(
            id="pos_procc",
            categoria="pos_graduacao",
            texto="Esse(a) docente atua no Programa de Pós-Graduação em Ciência da Computação (PROCC)?",
            prioridade_aluno=1.0,
            avaliador=lambda doc, discs: (
                "positivo"
                if "PROCC" in [p.strip() for p in doc["programas_pos_sigaa"].split(";")]
                else "negativo"
            ),
        ),
        RegraPergunta(
            id="pos_outros_programas",
            categoria="pos_graduacao",
            texto="Esse(a) docente atua em outro programa de pós fora do PROCC (ex.: PROEE, PPGITS, NPGME, PPGED, PPGPSI, PROFIAP)?",
            prioridade_aluno=1.0,
            avaliador=lambda doc, discs: (
                "positivo"
                if any(
                    p.strip() and p.strip() != "PROCC"
                    for p in doc["programas_pos_sigaa"].split(";")
                )
                else "negativo"
            ),
        ),
        # Bloco D — Formação e Desempate
        RegraPergunta(
            id="form_graduacao_ufs",
            categoria="formacao",
            texto="Esse(a) docente fez graduação na própria UFS?",
            prioridade_aluno=0.85,
            avaliador=lambda doc, discs: avaliar_booleano_csv(doc, "graduacao_ufs"),
        ),
        RegraPergunta(
            id="form_exterior",
            categoria="formacao",
            texto="Esse(a) docente fez parte da formação acadêmica no exterior?",
            prioridade_aluno=0.85,
            avaliador=lambda doc, discs: avaliar_booleano_csv(doc, "formacao_exterior"),
        ),
        RegraPergunta(
            id="form_pos_doc",
            categoria="formacao",
            texto="Esse(a) docente possui Pós-Doutorado?",
            prioridade_aluno=0.85,
            avaliador=lambda doc, discs: avaliar_booleano_csv(doc, "pos_doutorado"),
        ),
        RegraPergunta(
            id="vinculo_substituto",
            categoria="formacao",
            texto="Esse(a) docente é Professor(a) Substituto(a)?",
            prioridade_aluno=0.85,
            avaliador=lambda doc, discs: (
                "positivo" if doc["categoria"] == "Professor Substituto" else "negativo"
            ),
        ),
    ]
    return perguntas


def construir_matriz(
    docentes_csv: Path = RAW_DOCENTES,
    disciplinas_csv: Path = RAW_DISCIPLINAS,
) -> dict[str, Any]:
    """Gera o dicionário completo pronto para serializar em processed/perguntas_matriz.json."""
    docentes, disciplinas_por_siape = carregar_dados_brutos(docentes_csv, disciplinas_csv)
    catalogo = montar_catalogo_perguntas()

    docentes_out = [
        {
            "id": int(d["id"]),
            "siape": d["siape"],
            "nome": d["nome"],
        }
        for d in docentes
    ]

    perguntas_out: list[dict[str, Any]] = []
    for regra in catalogo:
        lik_sim: list[float] = []
        lik_nao: list[float] = []
        lik_nao_sei: list[float] = []

        for doc in docentes:
            discs = disciplinas_por_siape.get(doc["siape"], [])
            estado = regra.avaliador(doc, discs)
            p_sim, p_nao, p_ns = classificar_estado_para_probs(estado)
            lik_sim.append(round(p_sim, 4))
            lik_nao.append(round(p_nao, 4))
            lik_nao_sei.append(round(p_ns, 4))

        perguntas_out.append(
            {
                "id": regra.id,
                "categoria": regra.categoria,
                "texto": regra.texto,
                "prioridade_aluno": regra.prioridade_aluno,
                "lik": {
                    "sim": lik_sim,
                    "nao": lik_nao,
                    "nao_sei": lik_nao_sei,
                },
            }
        )

    return {
        "respostas_validas": list(RESPOSTAS_VALIDAS),
        "docentes": docentes_out,
        "perguntas": perguntas_out,
    }


def main() -> None:
    matriz = construir_matriz()
    OUT_MATRIZ.parent.mkdir(parents=True, exist_ok=True)
    OUT_MATRIZ.write_text(
        json.dumps(matriz, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUT_WEB_JS.parent.mkdir(parents=True, exist_ok=True)
    OUT_WEB_JS.write_text(
        "window.MATRIZ=" + json.dumps(matriz, ensure_ascii=False) + ";",
        encoding="utf-8",
    )
    print(
        f"Gerado {OUT_MATRIZ} e {OUT_WEB_JS} com {len(matriz['docentes'])} docentes "
        f"e {len(matriz['perguntas'])} perguntas."
    )


if __name__ == "__main__":
    main()
