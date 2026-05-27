# CLAUDE.md — Rinha de Backend 2026

Detecção de fraude via busca vetorial KNN. Python idiomático, sem extensões nativas.
Design completo em `docs/PRD.md`. Este arquivo concentra os findings que custaram tempo.

## Arquitetura

HAProxy (UDS, round-robin) → 2× Granian/Starlette → Faiss `IndexIVFFlat` por partição,
mmap-compartilhado. Payload → vetor 14-dim → chave de partição 8-bit → KNN k=5 → score.
Hot path: msgspec decode → vectorize → partition_key → faiss search → resposta
pré-renderizada (só 6 scores possíveis).

## Limites e scoring (decidem tudo)

- 1.0 CPU e 350 MB somando TODOS os serviços. Porta 9999. Network bridge. Repo MIT público.
- `(FP+FN+Err)/N > 15%` ⇒ detecção = -3000. p99 > 2000 ms ⇒ p99 = -3000.
- Erro ponderado: FP=1, FN=3, Err HTTP=5. **Estabilidade > latência.**
- Não usar payloads do teste como referência — só `references.json.gz`.

## Estado atual

**Score real (hardware da Rinha): ~738.** Prévia oficial: p99 **692 ms** (p99_score só 160),
detecção 578, failure 1.73%, 0 http_errors. **O gargalo é LATÊNCIA, não detecção.**

O k6 local numa CPU moderna dá p99 ~0.9 ms / final ~3571 — **enganoso**: o hardware da Rinha
é um Mac Mini 2014 (Haswell 2.6 GHz), bem mais lento. Sob 900 RPS as APIs (Faiss/Python)
ficam CPU-bound e a fila estoura o p99. Reproduzir local: `docker-compose.sim.yml` com
`SIM_API_CPUS=0.15 SIM_LB_CPUS=0.10` bate o p99 oficial dentro de 0.4%.

## Findings (não reaprender)

- **Medir no hardware certo.** Bench isolado mente (numpy brute-force parecia ok ~19 ms mas
  deu **-6000** sob carga). E o k6 local numa CPU rápida TAMBÉM mente: a quota de CPU do
  Docker limita fração de tempo, não a velocidade do core. Só a máquina lenta da Rinha (ou o
  `docker-compose.sim.yml` calibrado) revela o p99 real sob saturação.
- **Faiss IVFFlat + mmap** (`read_index(IO_FLAG_MMAP)`): ~0.3 ms/query, page cache
  compartilhado entre os 2 workers. Os vetores moram no índice (não guardar `.npy` separado).
- **faiss-cpu não tem wheel musllinux** → base Docker `python:3.14-slim`, NÃO alpine.
- **scipy cKDTree** converte para float64 interno (estoura RAM) e é incompatível com mmap.
  Descartado.
- **int16** exige `astype(int32)` por query → aloca no hot path → lento. float32 + BLAS vence.
- **nprobe=8 é o ótimo.** Subir dá +27 de detecção mas joga o p99 de ~0.9 ms para ~740 ms
  sob carga. As ~939 misclassifications são limite do KNN, não do IVF — nprobe não move.
- **Teto de detecção do KNN ≈ 600.** Para superar: modelo treinado offline (GBDT/MLP) sobre
  `references.json.gz`. Está no backlog (`docs/PRD.md` §9).
- **`ruff format` quebra `except (A, B):`** virando sintaxe inválida → usar
  `contextlib.suppress(...)`.
- **docker compose** roda imagem velha sem `--build`; o sweep de nprobe builda uma vez no início.
- **`umask 000`** no entrypoint: HAProxy roda non-root e precisa escrever no UDS compartilhado.
- **`storage_opt size`** do compose de teste precisa de xfs pquota — remover localmente.

## Dev

```
uv run python scripts/download_data.py   # baixa references/mcc_risk/normalization
uv run python scripts/build_index.py     # constrói data/index/ (idempotente)
task check                               # ruff + zuban + pytest
docker compose up --build                # stack completa em :9999
```

Sem `__init__.py` (PEP 420). Py 3.14: sem `from __future__ import annotations` (PEP 649).
ruff `select=ALL` com ignores em `ruff.toml`. Type check via zuban.
