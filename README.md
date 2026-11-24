# Akinator Bayesiano vs. LLM — DCOMP/UFS (SEMAC 2026)

Comparação entre um **Akinator Bayesiano (Regra de Bayes + Índice de Gini)** e um **agente LLM com RAG (n8n + `gpt-5-mini`)** para adivinhar qualquer um dos **33 docentes do DCOMP/UFS** usando **exatamente a mesma base de conhecimento** (`llm/base_de_conhecimento_rag_docentes_dcomp_ufs.md`).

**[Jogar a versão Bayesiana online](https://ladata-ufs.github.io/Akinator-vs-GPT-projeto-pratico/)**

---

## Como funciona

### 1. Abordagem Bayesiana (`src/`, `scripts/`, `web/`)
O jogo começa com probabilidade uniforme `1/33` para cada docente. A cada resposta (`Sim`, `Não` ou `Não sei`), as probabilidades são atualizadas pela Regra de Bayes usando a matriz de verossimilhança `P(E | H)` (`data/processed/perguntas_matriz.json`, gerada diretamente a partir do `.md` do RAG com **24 perguntas** sobre perfil/sala/foto/Lattes, formação acadêmica, áreas de interesse e atuação no SIGAA):

$$P(H_i \mid E) = \frac{P(H_i) \, P(E \mid H_i)}{\sum_{j=1}^{33} P(H_j) \, P(E \mid H_j)}$$

Para tolerar erros humanos sem eliminar o alvo de forma irreversível, a matriz usa probabilidades suavizadas (`0.89 / 0.05 / 0.06` para positivo e `0.05 / 0.89 / 0.06` para negativo). O `"Não sei"` mantém o mesmo peso para positivos e negativos, funcionando como resposta neutra.

A escolha da próxima pergunta minimiza a **impureza de Gini esperada** após a resposta:

$$\text{Gini}(p) = 1 - \sum_{i=1}^{33} p_i^2$$

Na interface web e no terminal, o motor sorteia entre as 3 melhores perguntas de Gini na 1ª rodada e desempata por proximidade (`margem = 0.002`) nas seguintes, variando as partidas. Nos benchmarks, o modo determinístico é mantido por padrão.

- **Modo 5 perguntas:** encerra após 5 perguntas (mesmo limite do LLM).
- **Modo livre:** encerra ao atingir `85%` de confiança ou `12` perguntas.

### 2. Abordagem LLM + RAG (`llm/`)
Implementada no n8n (`llm/workflow.json`), usa `gpt-5-mini`, embeddings da OpenAI, vector store em memória (`topK = 35`) e janela de memória de 6 turnos sobre `llm/base_de_conhecimento_rag_docentes_dcomp_ufs.md`. O jogador envia até 5 dicas livres em texto e o agente precisa cravar o docente até a 5ª dica.

---

## Comparativos e Resultados

### Desempenho do Bayes nos 33 docentes (`scripts/simular_benchmark.py`)

Com o motor Bayesiano restrito exclusivamente aos campos presentes no `.md` do RAG:

| Estratégia (Bayes) | 5 perguntas (Top-1 / Top-3) | Modo livre ≤12 (Top-1 / Média) |
|---|---:|---:|
| Determinístico (Gini puro) | 27/33 (81,8%) / 33/33 (100,0%) | 33/33 (100,0%) / 5,97 perg. |
| Dinâmico (Top-3 na 1ª + margem 0,002) | 77,6% / 98,9% | 100,0% / 6,23 perg. |

### Bayes × LLM na amostra de teste (`n = 10`, limite de 5 interações)

![Comparação Bayes vs LLM](results/comparacao.svg)

| Método | Top-1 | Top-3 |
|---|---:|---:|
| Bayes (5 perguntas, mesma base RAG) | 8/10 (80,0%) | 10/10 (100,0%) |
| Bayes (modo livre ≤12, mesma base RAG) | 10/10 (100,0%) | 10/10 (100,0%) |
| LLM (n8n + `gpt-5-mini` + RAG, 5 dicas) | 1/10 (10,0%) | — |

<details>
<summary>Resultado por docente (clique para expandir)</summary>

| Docente | Bayes (5 perg.) | Bayes Livre | LLM (5 dicas) | Palpite do LLM |
|---|---:|---:|---:|---|
| LEONARDO NOGUEIRA MATOS | ❌ (Top-3) | ✅ (7) | ❌ (5) | Giovanny Fernando Lucero Palma |
| MICHEL DOS SANTOS SOARES | ✅ | ✅ (6) | ❌ (5) | Bruno Otavio Piedade Prado |
| RICARDO JOSE PAIVA DE BRITTO SALGUEIRO | ✅ | ✅ (6) | ❌ (5) | Admilson de Ribamar Lima Ribeiro |
| ROGERIO PATRICIO CHAGAS DO NASCIMENTO | ✅ | ✅ (5) | ❌ (5) | Hendrik Teixeira Macedo |
| VICTOR FLAVIO DE ANDRADE ARAUJO | ❌ (Top-3) | ✅ (7) | ❌ (5) | Carlos Alberto Estombelo Montesco |
| GIANCARLO LIMA TORRES | ✅ | ✅ (5) | ❌ (5) | *Sem resposta (`{"output": ""}`)* |
| CARLOS ALBERTO ESTOMBELO MONTESCO | ✅ | ✅ (6) | ❌ (4) | Glauco de Figueiredo Carneiro |
| BEATRIZ TRINCHAO ANDRADE DE CARVALHO | ✅ | ✅ (7) | ❌ (2) | Alberto Costa Neto |
| KENIA KODEL COX | ✅ | ✅ (5) | ✅ (3) | **Kenia Kodel Cox** |
| CALEBE MICAEL DE OLIVEIRA CONCEICAO | ✅ | ✅ (4) | ❌ (3) | Kenia Kodel Cox |

</details>

**O que observamos nas partidas do LLM (`results/llm_resultados.json`):**
- Falhas de raciocínio sobre negação/afirmação (ex.: em *Michel*, uma dica afirmativa foi interpretada como negativa).
- Paradas prematuras com excesso de confiança antes de usar todas as 5 dicas (*Beatriz*, *Calebe*, *Carlos Montesco*).
- Instabilidade de execução no webhook (em *Giancarlo*, retornou `{ "output": "" }` vazio na 5ª dica).
- Parte das dicas gravadas originalmente em `results/llm_resultados.json` citava disciplinas de graduação ausentes no `.md` do RAG; mesmo após restringirmos o Bayes apenas ao `.md` do RAG, ele manteve 81,8% de acerto Top-1 em 5 perguntas nos 33 docentes (80% na amostra) e 100% no modo livre.

### Robustez do Bayes a ruído (`scripts/robustez.py`)

| Ruído simulado | 5 perguntas (Top-1 / Top-3) | Modo livre (Top-1 / Top-3 / Média) |
|---|---:|---:|
| 0% | 81,8% / 100,0% | 100,0% / 100,0% / 5,97 |
| 5% troca `Sim` ↔ `Não` | 63,7% / 84,9% | 88,6% / 97,6% / 6,83 |
| 10% troca `Sim` ↔ `Não` | 50,0% / 69,6% | 73,6% / 87,4% / 7,40 |
| 20% troca `Sim` ↔ `Não` | 30,4% / 49,5% | 39,9% / 60,3% / 7,78 |
| 10% `"Não sei"` | 64,8% / 96,0% | 99,2% / 100,0% / 6,87 |
| 30% `"Não sei"` | 39,0% / 76,2% | 93,5% / 99,6% / 8,76 |

### Resumo prático
- **Bayes + Gini:** roda offline em < 1 ms, custo zero, 100% explicável e tolera `"Não sei"`, mas trabalha sobre um conjunto fixo de 24 perguntas estruturadas.
- **LLM + RAG:** entende dicas livres em linguagem natural, porém depende da qualidade da recuperação textual, tem custo de API e pode errar filtros lógicos.

---

## Como rodar

Não precisa instalar bibliotecas externas — basta **Python 3.12+**:

```bash
# Jogar no terminal
python3 scripts/jogar_cli.py --modo 5
python3 scripts/jogar_cli.py --modo livre

# Regerar matriz a partir da base RAG (data/processed/perguntas_matriz.json e web/matriz.js)
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
├── llm/                # Base única (Markdown RAG) e workflow n8n
├── data/processed/     # Matriz P(E | H) gerada a partir do RAG (33 × 24)
├── src/                # Motor Bayesiano e cálculo de Gini
├── scripts/            # CLI, geração da matriz, benchmark, robustez e comparação
├── web/                # Interface HTML/JS estática
└── results/            # CSVs, JSON das partidas no LLM e gráfico SVG
```
