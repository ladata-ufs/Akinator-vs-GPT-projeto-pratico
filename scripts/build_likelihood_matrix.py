#!/usr/bin/env python3
"""Gera a matriz de verossimilhança P(E | H) em data/processed/perguntas_matriz.json
usando exclusivamente a mesma base de conhecimento da abordagem LLM + RAG
(`llm/base_de_conhecimento_rag_docentes_dcomp_ufs.md`).
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
RAG_BASE = REPO_ROOT / "llm" / "base_de_conhecimento_rag_docentes_dcomp_ufs.md"
OUT_MATRIZ = REPO_ROOT / "data" / "processed" / "perguntas_matriz.json"
OUT_WEB_JS = REPO_ROOT / "web" / "matriz.js"

RESPOSTAS_VALIDAS = ("sim", "nao", "nao_sei")


def normalizar_texto(texto: str) -> str:
    """Remove acentos, converte para minúsculas e normaliza espaços."""
    sem_acento = unicodedata.normalize("NFD", texto or "")
    sem_acento = "".join(ch for ch in sem_acento if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", sem_acento).strip().lower()


def _extrair_campo(bloco: str, rotulo: str) -> str:
    m = re.search(rf"-\s+\*\*{re.escape(rotulo)}:\*\*\s*(.+)", bloco)
    return m.group(1).strip() if m else ""


def carregar_docentes_rag(md_path: Path = RAG_BASE) -> list[dict[str, Any]]:
    """Extrai os 33 docentes exclusivamente de `base_de_conhecimento_rag_docentes_dcomp_ufs.md`."""
    texto = md_path.read_text(encoding="utf-8")
    docentes: list[dict[str, Any]] = []

    for bloco in re.split(r"\n---\n", texto):
        m_header = re.search(r"### Docente (\d+):\s*(.+)", bloco)
        if not m_header:
            continue

        loc = _extrair_campo(bloco, "Localização e Contatos")
        sala_m = re.search(r"Sala:\s*([^|]+)", loc)
        foto_m = re.search(r"Foto no SIGAA:\s*([^|]+)", loc)
        sala_txt = sala_m.group(1).strip() if sala_m else ""
        sala_num_m = re.search(r"(\d+)", sala_txt)

        form_norm = normalizar_texto(_extrair_campo(bloco, "Formação Acadêmica"))
        partes_form = [p.strip() for p in form_norm.split(";") if p.strip()]
        grad_txt = " ; ".join(
            p for p in partes_form if any(k in p for k in ("graduacao", "bacharelado", "licenciatura"))
        )
        pos_txt = " ; ".join(
            p for p in partes_form if any(k in p for k in ("mestrado", "doutorado", "pos-doutor"))
        )

        docentes.append(
            {
                "id": int(m_header.group(1)),
                "nome": m_header.group(2).strip(),
                "siape": _extrair_campo(bloco, "Matrícula SIAPE"),
                "sexo": normalizar_texto(_extrair_campo(bloco, "Sexo")),
                "categoria": normalizar_texto(_extrair_campo(bloco, "Categoria Funcional")),
                "sala_num": int(sala_num_m.group(1)) if sala_num_m else None,
                "foto": normalizar_texto(foto_m.group(1) if foto_m else ""),
                "formacao": form_norm,
                "grad": grad_txt,
                "pos": pos_txt,
                "areas": normalizar_texto(_extrair_campo(bloco, "Áreas de Interesse (Ensino/Pesquisa)")),
                "atuacao": normalizar_texto(_extrair_campo(bloco, "Atuação e Perfil Profissional")),
                "lattes": normalizar_texto(_extrair_campo(bloco, "Currículo Lattes")),
            }
        )

    return docentes


def _contem_algum(texto: str, termos: tuple[str, ...]) -> bool:
    return any(t in texto for t in termos)


def _avaliar_booleano(condicao: bool) -> str:
    return "positivo" if condicao else "negativo"


def _avaliar_campo_textual(texto: str, termos: tuple[str, ...]) -> str:
    if not texto or "nao cadastrad" in texto:
        return "negativo"
    return "positivo" if _contem_algum(texto, termos) else "negativo"


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
    avaliador: Callable[[dict[str, Any]], str]


def montar_catalogo_perguntas() -> list[RegraPergunta]:
    """Catálogo de 24 perguntas derivadas exclusivamente da base RAG do SIGAA."""
    return [
        # Perfil e localização (SIGAA)
        RegraPergunta(
            id="sexo_feminino",
            categoria="perfil",
            texto="Esse(a) docente é do sexo feminino?",
            prioridade_aluno=1.0,
            avaliador=lambda d: _avaliar_booleano(d["sexo"] == "feminino"),
        ),
        RegraPergunta(
            id="vinculo_substituto",
            categoria="perfil",
            texto="Esse(a) docente é Professor(a) Substituto(a)?",
            prioridade_aluno=1.0,
            avaliador=lambda d: _avaliar_booleano("substituto" in d["categoria"]),
        ),
        RegraPergunta(
            id="possui_foto_sigaa",
            categoria="perfil",
            texto="Esse(a) docente possui foto cadastrada no SIGAA?",
            prioridade_aluno=0.85,
            avaliador=lambda d: _avaliar_booleano(d["foto"] == "sim"),
        ),
        RegraPergunta(
            id="possui_lattes",
            categoria="perfil",
            texto="Esse(a) docente possui link de Currículo Lattes informado no SIGAA?",
            prioridade_aluno=0.85,
            avaliador=lambda d: _avaliar_booleano("http" in d["lattes"]),
        ),
        RegraPergunta(
            id="possui_sala",
            categoria="perfil",
            texto="Esse(a) docente tem número de sala informado no SIGAA?",
            prioridade_aluno=0.9,
            avaliador=lambda d: _avaliar_booleano(d["sala_num"] is not None),
        ),
        RegraPergunta(
            id="sala_ate_15",
            categoria="perfil",
            texto="Esse(a) docente tem sala cadastrada no SIGAA com número de 01 a 15?",
            prioridade_aluno=0.9,
            avaliador=lambda d: _avaliar_booleano(d["sala_num"] is not None and d["sala_num"] <= 15),
        ),
        RegraPergunta(
            id="perfil_completo_sigaa",
            categoria="perfil",
            texto="Esse(a) docente possui formação acadêmica cadastrada no SIGAA?",
            prioridade_aluno=0.9,
            avaliador=lambda d: _avaliar_booleano("nao cadastrada" not in d["formacao"]),
        ),
        # Formação acadêmica (SIGAA)
        RegraPergunta(
            id="graduacao_ufs",
            categoria="formacao",
            texto="Esse(a) docente fez graduação na UFS?",
            prioridade_aluno=1.0,
            avaliador=lambda d: _avaliar_booleano(bool(re.search(r"\b(ufs|sergipe)\b", d["grad"]))),
        ),
        RegraPergunta(
            id="graduacao_engenharia",
            categoria="formacao",
            texto="Esse(a) docente fez graduação em alguma Engenharia (Elétrica, Eletrônica, Civil, Computação ou Sistemas)?",
            prioridade_aluno=1.0,
            avaliador=lambda d: _avaliar_booleano("engenharia" in d["grad"]),
        ),
        RegraPergunta(
            id="pos_ufpe",
            categoria="formacao",
            texto="Esse(a) docente fez mestrado ou doutorado na UFPE?",
            prioridade_aluno=1.0,
            avaliador=lambda d: _avaliar_booleano(_contem_algum(d["pos"], ("ufpe", "pernambuco"))),
        ),
        RegraPergunta(
            id="pos_sudeste",
            categoria="formacao",
            texto="Esse(a) docente fez mestrado, doutorado ou pós-doutorado no Sudeste (USP, UNICAMP, UFSCAR, PUC-Rio ou UFU)?",
            prioridade_aluno=1.0,
            avaliador=lambda d: _avaliar_booleano(
                _contem_algum(d["pos"], ("usp", "unicamp", "ufscar", "puc-rio", "ufu"))
            ),
        ),
        RegraPergunta(
            id="pos_sul",
            categoria="formacao",
            texto="Esse(a) docente fez mestrado ou doutorado no Sul do Brasil (UFPR, UFRGS ou PUCPR)?",
            prioridade_aluno=1.0,
            avaliador=lambda d: _avaliar_booleano(
                _contem_algum(d["pos"], ("ufpr", "ufrgs", "pucpr", "parana", "rio grande do sul"))
            ),
        ),
        RegraPergunta(
            id="pos_exterior",
            categoria="formacao",
            texto="Esse(a) docente tem formação, doutorado sanduíche ou pós-doutorado no exterior?",
            prioridade_aluno=1.0,
            avaliador=lambda d: _avaliar_booleano(
                _contem_algum(
                    d["formacao"],
                    (
                        "alemanha",
                        "canada",
                        "suecia",
                        "colombia",
                        "argentina",
                        "eua",
                        "franca",
                        "holanda",
                        "portugal",
                        "espanha",
                        "paris",
                        "delft",
                        "aveiro",
                        "valencia",
                        "toronto",
                        "wisconsin",
                        "simon fraser",
                        "tubingen",
                    ),
                )
            ),
        ),
        RegraPergunta(
            id="pos_doutorado",
            categoria="formacao",
            texto="Esse(a) docente possui Pós-Doutorado cadastrado no SIGAA?",
            prioridade_aluno=1.0,
            avaliador=lambda d: _avaliar_booleano("pos-doutor" in d["pos"]),
        ),
        # Áreas de interesse (SIGAA)
        RegraPergunta(
            id="area_ling_prog",
            categoria="areas",
            texto="Esse(a) docente tem Linguagens de Programação ou Métodos Formais entre suas áreas no SIGAA?",
            prioridade_aluno=1.0,
            avaliador=lambda d: _avaliar_campo_textual(
                d["areas"], ("linguagens de programacao", "metodos formais")
            ),
        ),
        RegraPergunta(
            id="area_eng_software",
            categoria="areas",
            texto="Esse(a) docente atua com Engenharia de Software, Qualidade/Requisitos ou Sistemas de Informação?",
            prioridade_aluno=1.0,
            avaliador=lambda d: _avaliar_campo_textual(
                d["areas"],
                (
                    "engenharia de software",
                    "software engineering",
                    "sistemas de informacao",
                    "information systems",
                    "qualidade de software",
                ),
            ),
        ),
        RegraPergunta(
            id="area_ia_ml_dados",
            categoria="areas",
            texto="Esse(a) docente atua com IA, Aprendizado de Máquina, Mineração ou Ciência de Dados?",
            prioridade_aluno=1.0,
            avaliador=lambda d: _avaliar_campo_textual(
                d["areas"],
                (
                    "inteligencia artificial",
                    "aprendizado de maquina",
                    "aprendizagem de maquina",
                    "mineracao de dados",
                    "ciencia de dados",
                    "computacao inteligente",
                    "deep learning",
                    "redes neurais",
                ),
            ),
        ),
        RegraPergunta(
            id="area_redes_sd_iot",
            categoria="areas",
            texto="Esse(a) docente atua com Redes de Computadores, Sistemas Distribuídos, Nuvem ou IoT?",
            prioridade_aluno=1.0,
            avaliador=lambda d: _avaliar_campo_textual(
                d["areas"],
                (
                    "redes de computadores",
                    "sistemas distribuidos",
                    "nuvem",
                    "cloud",
                    "iot",
                    "internet das coisas",
                    "sensores",
                ),
            ),
        ),
        RegraPergunta(
            id="area_hardware_embarcados",
            categoria="areas",
            texto="Esse(a) docente atua com Hardware, Arquitetura de Computadores, Sistemas Embarcados ou FPGA?",
            prioridade_aluno=1.0,
            avaliador=lambda d: _avaliar_campo_textual(
                d["areas"],
                ("arquitetura de computadores", "sistemas embarcados", "microeletronica", "fpga", "hardware"),
            ),
        ),
        RegraPergunta(
            id="area_imagens_cg_visao",
            categoria="areas",
            texto="Esse(a) docente atua com Processamento de Imagens, Computação Gráfica ou Visão Computacional?",
            prioridade_aluno=1.0,
            avaliador=lambda d: _avaliar_campo_textual(
                d["areas"], ("processamento de imagens", "computacao grafica", "visao computacional")
            ),
        ),
        RegraPergunta(
            id="area_otimizacao_teoria",
            categoria="areas",
            texto="Esse(a) docente atua com Otimização, Teoria da Computação ou Complexidade?",
            prioridade_aluno=1.0,
            avaliador=lambda d: _avaliar_campo_textual(
                d["areas"], ("otimizacao", "teoria da computacao", "complexidade")
            ),
        ),
        RegraPergunta(
            id="area_educacao_jogos",
            categoria="areas",
            texto="Esse(a) docente atua com Informática na Educação, Jogos Educativos ou Ensino/Inclusão Digital?",
            prioridade_aluno=1.0,
            avaliador=lambda d: _avaliar_campo_textual(
                d["areas"],
                (
                    "educacao",
                    "educativa",
                    "jogos educativos",
                    "inclusao digital",
                    "tecnologias educacionais",
                    "ensino de programacao",
                ),
            ),
        ),
        RegraPergunta(
            id="area_saude_bio_cidades",
            categoria="areas",
            texto="Esse(a) docente trabalha com Saúde/Bioinformática/e-Health ou Cidades Inteligentes?",
            prioridade_aluno=1.0,
            avaliador=lambda d: _avaliar_campo_textual(
                d["areas"],
                (
                    "bioinformatica",
                    "saude",
                    "e-health",
                    "ehealth",
                    "elm-health",
                    "healthcare",
                    "informatica medica",
                    "biomedica",
                    "biossinais",
                    "cidades inteligentes",
                    "smart city",
                ),
            ),
        ),
        # Atuação e perfil profissional (SIGAA)
        RegraPergunta(
            id="atua_procc_gestao",
            categoria="atuacao",
            texto="O perfil no SIGAA menciona atuação no PROCC, liderança de grupo de pesquisa ou cargo de gestão?",
            prioridade_aluno=1.0,
            avaliador=lambda d: _avaliar_booleano(
                _contem_algum(d["atuacao"], ("procc", "lider", "chefe", "coordenador", "vice-diretor", "pro-reitor"))
            ),
        ),
    ]


def construir_matriz(md_path: Path = RAG_BASE) -> dict[str, Any]:
    """Gera o dicionário completo pronto para serializar em data/processed/perguntas_matriz.json."""
    docentes = carregar_docentes_rag(md_path)
    catalogo = montar_catalogo_perguntas()

    docentes_out = [
        {"id": d["id"], "siape": d["siape"], "nome": d["nome"]}
        for d in docentes
    ]

    perguntas_out: list[dict[str, Any]] = []
    for regra in catalogo:
        lik_sim: list[float] = []
        lik_nao: list[float] = []
        lik_nao_sei: list[float] = []

        for doc in docentes:
            estado = regra.avaliador(doc)
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
        f"e {len(matriz['perguntas'])} perguntas (base exclusiva: {RAG_BASE.name})."
    )


if __name__ == "__main__":
    main()
