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

**Score real (hardware da Rinha): ~3233.** Prévia oficial #7218: p99 65 ms (p99_score 1190),
detecção 2043, failure 0.15%, 0 http_errors. TP/TN/FP/FN = 23924/30054/63/18. Gargalo
restante: latência (p99_score 1190/3000) — detecção já no patamar prático.

**Como medir**: prévia oficial via issue `rinha/test smarzaro-python` no repo `zanfranceschi/
rinha-de-backend-2026`. Ilimitada por design da Rinha. **Não rodar sim local** — quota de
CPU no Docker limita fração de tempo, não a velocidade do core; bench rápido aqui não
prevê o Haswell saturado lá. Mesmo a prévia oficial varia ±10-15% entre runs (vimos 3045
e 2633 com a MESMA imagem por causa de http_errors do tail). Pra avaliar uma mudança:
duas prévias com o mesmo binário pra ter banda de variância, comparar contra duas prévias
da versão anterior.

Trajetória: 738 (nprobe=8) → 1528 (nprobe=1) → 1697 (+ fail-safe) → 2512 (+ bbox-prune) →
2921 (nprobe=2) → 3045 (asymmetric extras_nprobe=3) → **3233** (fast-path 2 regras).

## Findings (não reaprender)

- **Medir no hardware certo.** Bench isolado e k6 local mentem; só a prévia oficial conta.
  Numpy brute-force parecia ok isolado (~19 ms) e deu -6000 sob carga real. Quota de CPU
  no Docker não modela o core lento do Haswell. E mesmo a prévia tem variância: 3045 e
  2633 vieram com a MESMA imagem por causa de http_errors da cauda; nunca confiar num
  número único.
- **Faiss IVFFlat + mmap** (`read_index(IO_FLAG_MMAP)`): ~0.3 ms/query, page cache
  compartilhado entre os 2 workers. Os vetores moram no índice (não guardar `.npy` separado).
- **faiss-cpu não tem wheel musllinux** → base Docker `python:3.14-slim`, NÃO alpine.
- **scipy cKDTree** converte para float64 interno (estoura RAM) e é incompatível com mmap.
  Descartado.
- **int16** exige `astype(int32)` por query → aloca no hot path → lento. float32 + BLAS vence.
- **nprobe muda de ótimo conforme o resto do search evolui** (baked em `build_index.py`).
  Sob nprobe-only (sem bbox): nprobe=1 venceu (+790 vs nprobe=8: 738→1528) — cada lista IVF
  extra vira latência direta sob saturação. Depois do bbox-prune + unanimous-exit os fáceis
  saem rápido, então sobra orçamento pra recall maior nos boundaries: nprobe=2 venceu
  nprobe=1 (2921 vs 2512). nprobe=4 já piora (latência custa mais que a detecção rende).
- **Asymmetric nprobe entre primary e extras**: a primária paga a vasta maioria das queries
  via unanimous-exit, então custo dela domina o p99 (mantém nprobe=2). Os extras só rodam em
  queries ambíguas após filtro do bbox — vale a pena cavar mais fundo neles. `EXTRAS_NPROBE=3`
  é o ótimo (+124 real: 2921→3045). Subir a 4 ou 5 destrói o p99 sem ganho. **Levers que
  pareciam óbvios e não moveram nada**: padding 14→16 dim (SIMD alinhado), pruning tightness
  factor (< ou > 1.0). Tradeoff dominante é recall-vs-latência nos extras.
- **Levers estruturais que falharam**:
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
  no handler retornando o response de "fraude" (approved=False, score=1.0) custou +169 real
  (1528 → 1697). Bench/teste local não pega porque o `test-data.json` da Rinha não cobre
  os mesmos edge cases do oficial.
- **Env override em compose vence o bake silenciosamente.** `submission/docker-compose.yml`
  tinha `RINHA_NPROBE:-1` quando bakei `IVF_NPROBE=2` no meta — runner usou 1 e mascarou o
  ganho. Regra: submission roda do bake direto, **sem env de tuning**. Source of truth =
  `meta.json` do índice + constantes em `search.py`.
- **KNN k=5 EXATO dá 100% de accuracy sobre as references** (verificado em 2000 queries do
  test-data via brute-force BLAS, 0 TP/TN/FP/FN). Não tem regra escondida, não tem teto do
  KNN: o teto era a APROXIMAÇÃO do IVF. A combinação nprobe=2 + bbox cross-partition +
  unanimous-exit recupera quase todo o recall perdido (1.77% → 0.20% failure).
- **Bounding-box pruning cross-partition** (em `search.py`): salva `min`/`max` por dim por
  partição em `meta.json`; em runtime, depois da primária, calcula `lb_sq = ||q − bbox||²`
  vetorizado e visita só partições com `lb_sq < worst-of-top-K`, com merge incremental dos
  top-K e cap `MAX_EXTRA_PARTITIONS=8`. Sozinho custa ~127 pts de latência por query
  (`np.maximum`/`einsum`/`argsort`) → emparelhado com **early-exit por voto unânime** na
  primária para queries de alta confiança pularem o sweep. Combo deu **+815 no real**
  (1697 → 2512), detecção 560 → 1383, p99 quase inalterado.
- **Verificar unanime via bbox global tem teto mas custa caro demais.** Sempre computar bbox
  (mesmo nas unanimes) recupera ~36 wrong-unanime-exits — detecção sobe ~360. Mas o numpy
  per-query sob saturação no Haswell joga p99 de 100ms pra 560-735ms (mesmo com neighbor list
  pré-computada por partição). Net negativo. A imprecisão das unanimes é o preço a pagar.
- **Fast-path determinístico antes do KNN** (em `handlers.py`): 2 thresholds hand-derivados
  offline (`amount/avg ≤ 0.971` → legit; `amount > 2996` → fraud) cobrem ~92.6% das queries
  com 99.99% accuracy em references. Pula vectorize + partition_key + Faiss inteiro pra
  essas queries — só ~7.4% boundary cai no KNN. Custou +188 real (3045→3233) cortando
  p99 de 100ms→65ms, **sem perder detecção** (mesmos 2043). A latência saved reduz queue
  na cauda → menos timeouts → score mais estável entre prévias.
- **Tag `:latest` no Docker Hub não força re-pull no runner da Rinha.** Sempre re-taggear
  cada imagem nova com o SHA curto do commit e fazer pin no `submission/docker-compose.yml`.
  Senão o runner re-usa imagem cacheada e o resultado idêntico ao anterior mascara o teste.
- **Quota oficial: 10 prévias por dia** (apesar do doc dizer "ilimitadas"). 11ª retorna
  "Limite de submissões por dia atingido (10/10). Tente novamente amanhã." Orçar bem: pra
  uma mudança, idealmente 2 prévias do baseline + 2 da nova versão pra confirmar variância
  e direção. Variância vista ~10-15% entre runs com mesma imagem.
- **`ruff format` quebra `except (A, B):`** virando sintaxe inválida → usar
  `contextlib.suppress(...)`.
- **`umask 000`** no entrypoint: HAProxy roda non-root e precisa escrever no UDS compartilhado.

## Dev

```
uv run python scripts/download_data.py   # baixa references/mcc_risk/normalization
uv run python scripts/build_index.py     # constrói data/index/ (idempotente)
task check                               # ruff + zuban + pytest
docker compose up --build                # stack completa em :9999
```

Sem `__init__.py` (PEP 420). Py 3.14: sem `from __future__ import annotations` (PEP 649).
ruff `select=ALL` com ignores em `ruff.toml`. Type check via zuban.
