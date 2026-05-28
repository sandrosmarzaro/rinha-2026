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

**Score real (hardware da Rinha): ~1697.** Prévia oficial #7063: p99 73 ms (p99_score 1137),
detecção 560 (rate 1523, penalty −963), failure 1.77%, 0 http_errors. Gargalo restante: latência
sob saturação no Haswell. Subir além daqui exige eliminar a busca (modelo treinado offline).

O k6 local numa CPU moderna dá p99 ~0.9 ms / final ~3571 — **enganoso**: o hardware da Rinha
é um Mac Mini 2014 (Haswell 2.6 GHz), bem mais lento. Sob 900 RPS as APIs (Faiss/Python)
ficam CPU-bound e a fila estoura o p99. Reproduzir local: `docker-compose.sim.yml` com
`SIM_API_CPUS=0.15 SIM_LB_CPUS=0.10` bate o p99 e o final dentro de ~1%.

Trajetória: 738 (nprobe=8) → 1528 (nprobe=1) → **1697** (+ fail-safe no handler).

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
- **nprobe=1 é o ótimo no hardware real** (baked em `build_index.py`). Cada lista IVF a mais
  varrida vira latência direta sob saturação no Haswell. nprobe=1 deu +790 vs nprobe=8 (738 →
  1528). Perdeu ~25 de detecção, ganhou ~930 de p99_score. Sweep e calibração no rig casam
  com o oficial dentro de ~1%.
- **Levers estruturais que falharam** (rig calibrado):
  - Partição mais fina (512 chaves, amount em 3 bits): topo do amount é denso pelo clamp em
    10000 → split de só 26% no max, mas vizinhos se separam entre partições → recall caiu
    mais que a latência ganhou. **Regressão.**
  - Quantizador grosso HNSW (`IndexHNSWFlat` no IVF): `IndexFlatL2` em 14-dim já é SIMD,
    o passo grosso não dominava. ~30 µs vs ~30 µs. **Sem ganho.**
- **Handler precisa ser fail-safe.** Qualquer exception não tratada (timestamp esquisito,
  payload malformado) vira 500 → http_errors pesa **5×** no E (vs FP=1, FN=3). Try/except
  no handler retornando o response de "fraude" (approved=False, score=1.0) custou +169 no
  real (1528 → 1697) — o sim local não pega porque o `test-data.json` não tem os mesmos
  edge cases do oficial.
- **Teto de detecção do KNN ≈ 600.** Para superar: modelo treinado offline (GBDT/MLP) sobre
  `references.json.gz`. Está no backlog (`docs/PRD.md` §10).
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
