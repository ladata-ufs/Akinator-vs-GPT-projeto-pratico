# Akinator Bayesiano vs. LLM — DCOMP/UFS (SEMAC 2026)

Comparação entre um **Akinator Bayesiano (Regra de Bayes + Índice de Gini)** e um **agente LLM com RAG (n8n + `gpt-5-mini`)** para adivinhar qualquer um dos **33 docentes do DCOMP/UFS**.

**[Jogar a versão Bayesiana online](https://ladata-ufs.github.io/Akinator-vs-GPT-projeto-pratico/)**

---

## Como funciona

### 1. Abordagem Bayesiana (`src/`, `scripts/`, `web/`)
O jogo começa com probabilidade uniforme `1/33` para cada docente. A cada resposta (`Sim`, `Não` ou `Não sei`), as probabilidades são atualizadas pela Regra de Bayes usando a matriz de verossimilhança `P(E | H)` (`data/processed/perguntas_matriz.json`, com 23 perguntas sobre disciplinas do SIGAA, áreas de pesquisa, pós-graduação e formação):

$$P(H_i \mid E) = \frac{P(H_i) \, P(E \mid H_i)}{\sum_{j=1}^{33} P(H_j) \, P(E \mid H_j)}$$

Para evitar que um erro humano elimine o docente correto de forma irreversível, a matriz usa probabilidades suavizadas (`0.89 / 0.05 / 0.06` para positivo, `0.05 / 0.89 / 0.06` para negativo, `0.62 / 0.24 / 0.14` para parcial e `0.34 / 0.33 / 0.33` quando o dado não consta no SIGAA). O `"Não sei"` mantém o mesmo peso para positivos e negativos, funcionando como resposta neutra.

A escolha da próxima pergunta minimiza a **impureza de Gini esperada** após a resposta:

$$\text{Gini}(p) = 1 - \sum_{i=1}^{33} p_i^2$$

Na interface web e no terminal, o motor sorteia entre as 3 melhores perguntas de Gini na 1ª rodada e desempata por proximidade (`margem = 0.002`) nas seguintes, variando as partidas sem perder acurácia. Nos benchmarks, o modo determinístico é mantido por padrão.

- **Modo 5 perguntas:** para em 5 perguntas (mesmo limite do LLM).
- **Modo livre:** para ao atingir `85%` de confiança ou `12` perguntas.

### 2. Abordagem LLM + RAG (`llm/`)
Implementada no n8n (`llm/workflow.json`), usa `gpt-5-mini`, embeddings da OpenAI, vector store em memória (`topK = 35`) e janela de memória de 6 turnos sobre `llm/base_de_conhecimento_rag_docentes_dcomp_ufs.md`. O jogador envia até 5 dicas livres em texto e o agente tenta cravar o docente até a 5ª dica.

---

## Comparativos e Resultados

### Desempenho nos 33 docentes (`scripts/simular_benchmark.py`)

| Estratégia (Bayes) | 5 perguntas (Top-1 / Top-3) | Modo livre ≤12 (Top-1 / Média) |
|---|---:|---:|
| Determinístico (Gini puro) | 27/33 (81,8%) / 33/33 (100,0%) | 33/33 (100,0%) / 5,97 perg. |
| Dinâmico (Top-3 na 1ª + margem 0,002) | 82,4% / 99,7% | 100,0% / 6,07 perg. |

### Bayes × LLM na amostra de teste (`n = 10`, até 5 interações)

Testamos 10 docentes em abas limpas no chat do n8n (os 6 que o Bayes não fecha em Top-1 com 5 perguntas + 4 aleatórios), enviando ao LLM as mesmas 5 informações das primeiras perguntas do Bayes em forma de dicas:

![Comparação Bayes vs LLM](results/comparacao.svg)

| Método | Top-1 | Top-3 |
|---|---:|---:|
| Bayes (5 perguntas) | 4/10 (40,0%) | 10/10 (100,0%) |
| Bayes (modo livre ≤12) | 10/10 (100,0%) | 10/10 (100,0%) |
| LLM (n8n + `gpt-5-mini` + RAG, 5 dicas) | 1/10 (10,0%) | — |

<details>
<summary>Resultado por docente (clique para expandir)</summary>

| Docente | Bayes (5 perg.) | Bayes Livre | LLM (5 dicas) | Palpite do LLM |
|---|---:|---:|---:|---|
| LEONARDO NOGUEIRA MATOS | ❌ (Top-3) | ✅ (7) | ❌ (5) | Giovanny Fernando Lucero Palma |
| MICHEL DOS SANTOS SOARES | ❌ (Top-3) | ✅ (8) | ❌ (5) | Bruno Otavio Piedade Prado |
| RICARDO JOSE PAIVA DE BRITTO SALGUEIRO | ❌ (Top-3) | ✅ (7) | ❌ (5) | Admilson de Ribamar Lima Ribeiro |
| ROGERIO PATRICIO CHAGAS DO NASCIMENTO | ❌ (Top-3) | ✅ (7) | ❌ (5) | Hendrik Teixeira Macedo |
| VICTOR FLAVIO DE ANDRADE ARAUJO | ❌ (Top-3) | ✅ (6) | ❌ (5) | Carlos Alberto Estombelo Montesco |
| GIANCARLO LIMA TORRES | ❌ (Top-3) | ✅ (8) | ❌ (5) | *Sem resposta (`{"output": ""}`)* |
| CARLOS ALBERTO ESTOMBELO MONTESCO | ✅ | ✅ (5) | ❌ (4) | Glauco de Figueiredo Carneiro |
| BEATRIZ TRINCHAO ANDRADE DE CARVALHO | ✅ | ✅ (5) | ❌ (2) | Alberto Costa Neto |
| KENIA KODEL COX | ✅ | ✅ (7) | ✅ (3) | **Kenia Kodel Cox** |
| CALEBE MICAEL DE OLIVEIRA CONCEICAO | ✅ | ✅ (5) | ❌ (3) | Kenia Kodel Cox |

</details>

**Por que o LLM errou 9 de 10 nesse teste:**
- O `.md` do RAG não tem a lista de disciplinas do SIGAA (apenas perfil, formação e áreas de pesquisa). Quando a dica citava disciplinas, o LLM aproximou por palavras-chave nas áreas e descartou o docente certo (Leonardo, Carlos Montesco, Beatriz e Calebe).
- Em *Michel*, a 5ª dica era afirmativa (*"costuma ministrar Programação Imperativa"*), mas o LLM leu como negativa e eliminou o próprio Michel.
- Em *Beatriz*, *Calebe* e *Carlos Montesco*, o agente parou antes da 5ª dica achando que só havia 1 candidato possível.
- Em *Giancarlo*, o webhook retornou `{ "output": "" }` vazio na 5ª dica.

> **Observação:** as dicas foram geradas a partir das perguntas do Bayes (que incluem disciplinas do SIGAA ausentes no RAG) e a amostra já incluía os 6 casos mais difíceis em 5 perguntas. Com dicas livres focadas em sexo, sala, formação e cargos, o RAG tende a acertar bem mais.

### Robustez do Bayes a ruído (`scripts/robustez.py`)

| Ruído simulado | 5 perguntas (Top-1 / Top-3) | Modo livre (Top-1 / Top-3 / Média) |
|---|---:|---:|
| 0% | 81,8% / 100,0% | 100,0% / 100,0% / 5,97 |
| 5% troca `Sim` ↔ `Não` | 64,5% / 83,7% | 89,1% / 95,9% / 6,66 |
| 10% troca `Sim` ↔ `Não` | 50,0% / 70,8% | 73,7% / 89,0% / 7,24 |
| 20% troca `Sim` ↔ `Não` | 30,5% / 47,3% | 44,4% / 62,3% / 7,87 |
| 10% `"Não sei"` | 65,5% / 95,3% | 99,7% / 100,0% / 6,85 |
| 30% `"Não sei"` | 36,6% / 71,8% | 96,0% / 99,6% / 8,92 |

### Resumo prático
- **Bayes + Gini:** roda offline em < 1 ms, custo zero, 100% explicável e tolera `"Não sei"`, mas só responde ao conjunto fixo de 23 perguntas.
- **LLM + RAG:** entende dicas livres em linguagem natural, porém depende do que está no texto do RAG, tem custo de API e pode errar filtros lógicos.

---

## Como rodar

Não precisa instalar bibliotecas externas — basta **Python 3.12+**:

```bash
# Jogar no terminal
python3 scripts/jogar_cli.py --modo 5
python3 scripts/jogar_cli.py --modo livre

# Regerar matriz (data/processed/perguntas_matriz.json e web/matriz.js)
python3 scripts/build_likelihood_matrix.py

# Rodar benchmarks e análises
python3 scripts/simular_benchmark.py
python3 scripts/robustez.py
python3 scripts/comparar.py
```

Para jogar na interface web localmente, abra [`web/index.html`](web/index.html) no navegador.  
Para rodar o agente LLM, importe [`llm/workflow.json`](llm/workflow.json) no n8n, configure sua chave da OpenAI e envie [`llm/base_de_conhecimento_rag_docentes_dcomp_ufs.md`](llm/base_de_conhecimento_rag_docentes_dcomp_ufs.md) pelo formulário inicial do fluxo.

## Estrutura

```
├── index.html          # Redirecionamento para web/ (GitHub Pages)
├── data/
│   ├── raw/            # CSVs de docentes e disciplinas (SIGAA/UFS)
│   └── processed/      # Matriz P(E | H) gerada (33 × 23)
├── src/                # Motor Bayesiano e cálculo de Gini
├── scripts/            # CLI, geração da matriz, benchmark, robustez e comparação
├── web/                # Interface HTML/JS estática
├── llm/                # Workflow n8n e base RAG em Markdown
└── results/            # CSVs, JSON das partidas no LLM e gráfico SVG
```
