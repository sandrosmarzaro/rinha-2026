# CLAUDE.md — Rinha de Backend 2026

Detecção de fraude via busca vetorial KNN. Python idiomático, sem extensões nativas.
Design completo em `docs/PRD.md`. **Histórico de evolução — todos os plateaus, tentativas
descartadas, e findings algorítmicos — em `docs/EVOLUTION.md`** (consultar antes de
re-tentar algo que parece novo). Este arquivo concentra **arquitetura corrente, estado
atual, e gotchas vivos**.

## Arquitetura

HAProxy (UDS, round-robin) → 2× Granian/RSGI → **pure numpy IVF** (nlist=2048, nprobe=12),
mmap-compartilhado. Faiss roda só no build pra treinar k-means; runtime carrega
`centroids.npy` + `vectors.npy` cluster-sorted + `vec_norms.npy` pré-computadas e calcula
distâncias via norm-expansion (`||a-b||² = ||a||² + ||b||² - 2·a·b`). Payload → 2-rule raw
fast-path (92.6%) OR vectorize 14-dim → partition_key 8-bit (homogeneous-exit) → numpy IVF
KNN k=5 → resposta pré-renderizada (só 6 scores possíveis).

## Limites e scoring (decidem tudo)

- 1.0 CPU e 350 MB somando TODOS os serviços. Porta 9999. Network bridge. Repo MIT público.
- `(FP+FN+Err)/N > 15%` ⇒ detecção = -3000. p99 > 2000 ms ⇒ p99 = -3000.
- Erro ponderado: FP=1, FN=3, Err HTTP=5. **Estabilidade > latência.**
- Não usar payloads do teste como referência — só `references.json.gz`.
- Repartição atual: APIs 0.40 cada, HAProxy 0.20 (validado plateau #12; HAProxy a 0.10 era
  CPU-starved → ERR variance).

## Estado atual

**Score real (hardware da Rinha): 4177 confirmado em prévia #7906.** Plateau #13 com
pure numpy IVF (drop Faiss do runtime, ele só treina k-means no build). Detection 2501
(FP=33, FN=4, ERR=0, E=45 — idêntica ao plateau #12 já que mesmas centroides + mesmo
nprobe = mesmo recall); p99 21.1ms; p99 score 1676. Gargalo restante = ainda p99 score
(1676/3000) — detection já saturada perto do teto.

**Como medir**: prévia oficial via issue `rinha/test smarzaro-python` no repo
`zanfranceschi/rinha-de-backend-2026`. Sim local em `docker-compose.sim.yml` calibrado
0.16/0.16/0.08 (≈ real 0.40/0.40/0.20 com fator throttle 2.5×). **Sim prediz bem
mudanças de parâmetros dentro do mesmo motor; engine MIXING (Faiss+numpy) diverge real
mas single-engine numpy reproduz sim direção (Lever 5 mixto regrediu real, plateau #13
single-engine ganhou).** Validar mudanças de substrato com bench bare-metal
`taskset -c 0` antes de queimar prévia.

## Findings vivos (regras de engajamento)

### Medição

- **Bench isolado e k6 local mentem; só a prévia oficial conta.** Numpy brute-force
  parecia ok isolado (~19 ms) e deu -6000 sob carga real. Quota de CPU no Docker não
  modela o core lento do Haswell linearmente.
- **Mesmo a prévia oficial varia ±10-15%**. Vimos 3045 e 2633 com a MESMA imagem por
  causa de http_errors do tail. Pra avaliar uma mudança: 2 prévias do baseline + 2 da
  nova versão pra confirmar variância e direção. Nunca confiar num número único.
- **Quota oficial: 10 prévias por dia** (apesar do doc dizer "ilimitadas"). 11ª retorna
  "Limite de submissões por dia atingido (10/10). Tente novamente amanhã." Orçar bem.
- **Gargalo real é CPU throttle do cgroup, não o handler.** Profile interno
  (`fraud_api/profile.py`, opt-in via `RINHA_PROFILE=1`) mediu p99 dentro do worker
  = 1.5ms vs k6 p99 = 52ms (plateau 11 era). O gap são os ~50ms de espera por slot CPU
  quando o cgroup CFS quota se esgota no janelão de 100ms. Implicação: otimizar
  microsegundos do handler não move o score; tem que reduzir CPU/request a ponto de
  caber em uma janela.

### Constraints técnicos

- **faiss-cpu não tem wheel musllinux** → base Docker `python:3.14-slim`, NÃO alpine.
- **`umask 000` no entrypoint**: HAProxy roda non-root e precisa escrever no UDS
  compartilhado.
- **Tag `:latest` no Docker Hub não força re-pull no runner da Rinha.** Sempre
  re-taggear cada imagem nova com o SHA curto do commit e fazer pin no
  `submission/docker-compose.yml`. Senão o runner re-usa imagem cacheada e o resultado
  idêntico ao anterior mascara o teste.
- **Env override em compose vence o bake silenciosamente.** Submission roda do bake
  direto, **sem env de tuning**. Source of truth = `meta.json` do índice + constantes
  em `search.py`.

### Design invariantes

- **Handler precisa ser fail-safe.** Qualquer exception não tratada (timestamp esquisito,
  payload malformado) vira 500 → http_errors pesa **5×** no E (vs FP=1, FN=3). Try/except
  no handler retornando o response de "fraude" (approved=False, score=1.0) custou +169
  real (1528 → 1697). Bench/teste local não pega porque o `test-data.json` da Rinha não
  cobre os mesmos edge cases do oficial.
- **KNN k=5 EXATO dá 100% de accuracy sobre as references** (verificado via brute-force
  BLAS). Não tem regra escondida; o teto é a APROXIMAÇÃO do IVF, não o classificador.
  Levers que reduzem recall (cell-mean fast-path, tree pre-classifier, int8 SQ) regridem
  detection irrecuperavelmente nas queries hard pós-2-rule.
- **2-rule raw fast-path** (`amount/avg ≤ 0.971` → legit; `amount > 2996` → fraud) cobre
  92.6% com 99.99% pureza, **sem vectorize** — é a peça que mantém custo médio baixo.
  Substituir por tree treinada em vetores regrediu (tree generaliza pior do espaço raw).

### Gotchas

- **`ruff format` quebra `except (A, B):`** virando sintaxe inválida → usar
  `contextlib.suppress(...)`.
- **`partition_key` 8-bit é vestigial pós single-IVF** — sobrevive só pra o
  `homogeneous_score` early-exit (18 partições puras catch 9.5% dos boundary).
  K-means coarse quantizer faz o resto via L2 real.

## Fora do constraint do projeto

SIMD AVX2 escrito à mão, custom HTTP via io_uring, FD passing entre processos e busy-poll
em NAPI renderiam muito, mas violam "Python idiomático sem extensões nativas" e estão fora
de escopo. Linguagens compiladas em geral conseguem voar com KNN exato e cabem tudo em ~1ms
no Haswell, fora do nosso teto.

## Dev

```
uv run python scripts/download_data.py   # baixa references/mcc_risk/normalization
uv run python scripts/build_index.py     # constrói data/index/ (idempotente)
task check                               # ruff + zuban + pytest
docker compose up --build                # stack completa em :9999
```

Sem `__init__.py` (PEP 420). Py 3.14: sem `from __future__ import annotations` (PEP 649).
ruff `select=ALL` com ignores em `ruff.toml`. Type check via zuban.

Estrutura:

- `fraud_api/` — código de produção (RSGI app, Faiss + numpy KNN, vectorize, partition).
- `scripts/` — build_index, download_data (offline).
- `tests/` — pytest (parity oracle).
- `bench/` — k6 load testing local (gitignored, espelha o setup do runner da Rinha).
- `docs/PRD.md` — design original.
- `docs/EVOLUTION.md` — histórico completo (plateaus, descartes, findings).
