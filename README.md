# Akinator Bayesiano vs. LLM — Docentes do DCOMP/UFS (SEMAC 2026)

Implementação de um **Akinator Bayesiano** baseado em atualização de probabilidades pela Regra de Bayes e seleção dinâmica de perguntas via **Índice de Gini**, construído sobre os dados públicos dos **33 docentes do DCOMP/UFS** e comparado à abordagem com **LLM + RAG** (`origin/akinator-abordagem-gpt`).

## Resultados Principais

### Desempenho nos 33 docentes (`scripts/simular_benchmark.py`)
| Modo | Top-1 | Top-3 | Média de perguntas |
|---|---:|---:|---:|
| **5 perguntas** (mesmo limite da LLM) | 27/33 (81,8%) | 33/33 (100,0%) | 4,91 |
| **Convergência livre** (≤12 perguntas, limiar 85%) | 33/33 (100,0%) | 33/33 (100,0%) | 5,97 |

### Comparação Bayes × LLM na amostra de teste (`n = 10`)
- **Bayes (5 perguntas):** 4/10 Top-1 (40,0%) | 10/10 Top-3 (100,0%)
- **Bayes (livre ≤12):** 10/10 Top-1 (100,0%)
- **LLM (n8n + GPT-5-mini + RAG, 5 dicas):** 1/10 Top-1 (10,0%)

Detalhes em [`results/comparacao.md`](results/comparacao.md) e [`results/robustez.csv`](results/robustez.csv).

## Estrutura do Projeto
```
├── raw/            # Tabelas base de docentes e disciplinas (SIGAA/UFS)
├── processed/      # Matriz de verossimilhança P(E | H) (33 docentes × 23 perguntas)
├── src/            # Motor Bayesiano e seleção por Índice de Gini
├── scripts/        # Geração da matriz, CLI interativa, benchmarks e comparação
├── web/            # Interface web interativa (HTML/JS puro)
└── results/        # Resultados dos experimentos e gráficos
```

## Como Executar

Requer apenas **Python 3.12+** (biblioteca padrão).

- **Interface Web:** abra [`web/index.html`](web/index.html) no navegador.
- **Jogar no terminal:**
  ```bash
  python3 scripts/jogar_cli.py --modo 5
  python3 scripts/jogar_cli.py --modo livre
  ```
- **Reproduzir matriz, benchmarks e análises:**
  ```bash
  python3 scripts/build_likelihood_matrix.py
  python3 scripts/simular_benchmark.py
  python3 scripts/robustez.py
  python3 scripts/comparar.py
  ```
