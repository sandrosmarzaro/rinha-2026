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

**Score real (hardware da Rinha): ~2921.** Prévia oficial #7152: p99 87 ms (p99_score 1061),
detecção 1860 (rate 2541, penalty −681), failure 0.20%, 0 http_errors. TP/TN/FP/FN =
23914/30039/78/28. Gargalo restante: latência sob saturação no Haswell.

O k6 local numa CPU moderna dá p99 ~0.9 ms / final ~3571 — **enganoso**: o hardware da Rinha
é um Mac Mini 2014 (Haswell 2.6 GHz), bem mais lento. Sob 900 RPS as APIs (Faiss/Python)
ficam CPU-bound e a fila estoura o p99. Reproduzir local: `docker-compose.sim.yml` com
`SIM_API_CPUS=0.15 SIM_LB_CPUS=0.10` bate o oficial dentro de ~5%.

Trajetória: 738 (nprobe=8) → 1528 (nprobe=1) → 1697 (+ fail-safe) → 2512 (+ bbox-prune) →
**2921** (nprobe=2, recovers recall on cross-partition extras).

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
- **nprobe muda de ótimo conforme o resto do search evolui** (baked em `build_index.py`).
  Sob nprobe-only (sem bbox): nprobe=1 venceu (+790 vs nprobe=8: 738→1528) — cada lista IVF
  extra vira latência direta sob saturação. Depois do bbox-prune + unanimous-exit os fáceis
  saem rápido, então sobra orçamento pra recall maior nos boundaries: nprobe=2 com cap=8
  bate nprobe=1 cap=8 por +441 no rig (2480→2768; real 2512→2921). nprobe=4 piora (latência
  sobe mais que detecção sobe). Toda vez que mudar bbox/cap, re-sweepar nprobe no rig.
- **Levers estruturais que falharam** (rig calibrado):
  - Partição mais fina (512 chaves, amount em 3 bits): topo do amount é denso pelo clamp em
    10000 → split de só 26% no max, mas vizinhos se separam entre partições → recall caiu
    mais que a latência ganhou. **Regressão.**
  - Quantizador grosso HNSW (`IndexHNSWFlat` no IVF): `IndexFlatL2` em 14-dim já é SIMD,
    o passo grosso não dominava. ~30 µs vs ~30 µs. **Sem ganho.**
  - `IndexFlatL2` puro por partição (KNN exato dentro): satura o Haswell catastroficamente,
    93% timeout, **-6000**.
  - Feature weights por dim usando `|ρ|` (correlação com label): com métrica uniforme já
    100% acurada, qualquer perturbação só piora os k-vizinhos. **Regressão (-116).**
- **Handler precisa ser fail-safe.** Qualquer exception não tratada (timestamp esquisito,
  payload malformado) vira 500 → http_errors pesa **5×** no E (vs FP=1, FN=3). Try/except
  no handler retornando o response de "fraude" (approved=False, score=1.0) custou +169 no
  real (1528 → 1697) — o sim local não pega porque o `test-data.json` não tem os mesmos
  edge cases do oficial.
- **Env override em compose vence o bake silenciosamente.** O `submission/docker-compose.yml`
  tinha `RINHA_NPROBE:-1` quando bakei `IVF_NPROBE=2` no meta — runner usou 1 e mascarou o
  ganho. Regra: submission roda do bake direto, **sem env de tuning**; sim mantém envs pra
  sweeps; mudou bake → remover/atualizar a env em compose. Source of truth = `meta.json`.
- **KNN k=5 EXATO dá 100% de accuracy sobre as references** (verificado em 2000 queries do
  test-data via brute-force BLAS, 0 TP/TN/FP/FN). Não tem regra escondida, não tem teto do
  KNN: o teto era a APROXIMAÇÃO do IVF. A combinação nprobe=2 + bbox cross-partition +
  unanimous-exit recupera quase todo o recall perdido (1.77% → 0.20% failure).
- **Bounding-box pruning cross-partition** (em `search.py`): salva `min`/`max` por dim por
  partição em `meta.json`; em runtime, depois da primária, calcula `lb_sq = ||q − bbox||²`
  vetorizado e visita só partições com `lb_sq < worst-of-top-K`, com merge incremental
  dos top-K e cap `RINHA_MAX_EXTRA` (default 8). Sozinho custa ~127 pts de latência por
  query (`np.maximum`/`einsum`/`argsort`) → emparelhado com **early-exit por voto unânime**
  na primária para queries de alta confiança pularem o sweep. Combo deu **+815 no real**
  (1697 → 2512), detecção 560 → 1383, p99 quase inalterado.
- **Tag `:latest` no Docker Hub não força re-pull no runner da Rinha.** Sempre re-taggear
  cada imagem nova com o SHA curto do commit e fazer pin no `submission/docker-compose.yml`.
  Senão o runner re-usa imagem cacheada e o resultado idêntico ao anterior mascara o teste.
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
