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

**Score real (hardware da Rinha): banda 3190-3233.** Duas prévias na mesma imagem
`:c4d12ec` deram 3233 (#7218) e 3190 (#7232) — jitter ~1.3% só de p99 (64ms vs 72ms),
detecção cravada em 2043-2048. Failure ~0.15%. **O algoritmo é determinístico**: TP/TN/FP/FN
em ~23900-23925/30030-30055/62-63/18. Variância maior (vimos -400 no #7213 com 26 http_errors)
é caso atípico de cauda do runner, não a banda normal. Gargalo: latência (p99_score 1190/3000)
— detecção já no patamar prático.

**Como medir**: prévia oficial via issue `rinha/test smarzaro-python` no repo `zanfranceschi/
rinha-de-backend-2026`. Ilimitada por design da Rinha. **Não rodar sim local** — quota de
CPU no Docker limita fração de tempo, não a velocidade do core; bench rápido aqui não
prevê o Haswell saturado lá. Mesmo a prévia oficial varia ±10-15% entre runs (vimos 3045
e 2633 com a MESMA imagem por causa de http_errors do tail). Pra avaliar uma mudança:
duas prévias com o mesmo binário pra ter banda de variância, comparar contra duas prévias
da versão anterior.

## Trajetória completa

Cada nova plateau atingida é registrada aqui na hora — sem precisar instruir, pra
preservar contra autocompactação. As linhas abaixo são incrementos validados (sim
calibrado e/ou prévia oficial); cada uma é um commit no repo.

### Mudanças que ficaram

| # | Score | Técnica | Δ | Comprovação |
|---|---:|---|---:|---|
| 0 | −6000 | numpy brute-force partitioned (M1-M5 PRD original) | — | k6 oficial: ~15ms/query satura 1 CPU @ 900 RPS → fila → 80% timeout |
| 1 | 738 | Faiss `IndexIVFFlat` + `mmap` (IO_FLAG_MMAP) por partição, `nprobe=8` | **+6738** | prévia oficial (M6) |
| 2 | 1528 | `nprobe 8 → 1` (baked em `build_index`) | **+790** | prévia: cada lista IVF a mais era latência pura sob saturação |
| 3 | 1697 | fail-safe `try/except` no `fraud_score` handler, default = fraud | **+169** | prévia: crashes em payloads edge eliminados (http_errors pesa 5× no E) |
| 4 | 2512 | bbox `lb_sq` cross-partition + unanime-exit + cap=8 | **+815** | prévia: `min`/`max` por dim em `meta.json`; pula sweep se 5/5 mesma label |
| 5 | 2921 | `nprobe 1 → 2` (baked) | **+409** | prévia: com bbox+unanime, dá pra cavar mais fundo na primária |
| 6 | 3045 | asymmetric `EXTRAS_NPROBE=3` (primary fica em 2) | **+124** | prévia: extras só rodam após bbox filter, vale recall maior |
| 7 | **3190-3233** | fast-path determinístico ANTES do KNN (2 regras hand-derivadas) | **+145-188** | prévias #7218/#7232. `amount/avg ≤ 0.971` → legit; `amount > 2996` → fraud. 92.6% cov 99.99% pureza |

### Descartados (cada um testado, com motivo)

| Tentativa | Resultado | Onde quebrou |
|---|---:|---|
| `scipy.spatial.cKDTree` | drop | float64 interno → estoura RAM; incompatível com mmap |
| int16 quantization (numpy hot path) | -48ms p99 | `astype(int32)` por query aloca 28 MB no hot path |
| Partição mais fina (512 chaves, amount 3-bit) | -43 sim | topo do amount denso (clamp em 10000) → split só 26%; vizinhos se separam → recall cai mais que latência ganha |
| HNSW coarse quantizer no IVF (`IndexHNSWFlat`) | 0 | `IndexFlatL2` em 14-dim já é SIMD-vetorizado; coarse step não dominava |
| `IndexFlatL2` puro por partição (exato dentro) | -6000 sim | KNN exato em ~283k vetores médios não cabe no Haswell saturado, 93% timeout |
| Feature weights `\|ρ\|` por dim | -116 sim | métrica uniforme já 100% acurada na references → perturbação só piora |
| Padding 14→16 dim (SIMD alignment) | 0 sim | Faiss alinha internamente; tail handling não era gargalo |
| Tightness factor no bbox prune (< 1.0 ou > 1.0) | 0/-1000 sim | bbox natural já calibrado; tight<1.0 corta partições críticas, tight>1.0 só adiciona custo |
| Always-bbox compute + min-lb verified unanime-exit | -460 sim | recupera +360 detecção (36 wrong-unanimes) mas numpy/query sob saturação joga p99 de 100ms pra 560ms |
| Smart unanime-exit com `homogeneous_score` filter | -390 sim | mesmo problema: bbox-compute por query mata p99 |
| Neighbor list pré-computada (K=32 por partição) | -510 sim | fancy indexing `bbox_max[neighbors]` ainda custa caro; não baixa bbox-cost |
| 6-rule fast-path (km_home/installments/km_last/tx24h adicionais) | 0 sim | queries extras eram fáceis pro KNN; saving ~1µs/query, neutro |
| Decision tree depth=5 sklearn como fast-path | proj. neutro | 2 leaves ≥99.99% pureza cobrem 96.5% (vs 92.6% com 2 regras), adiciona ~36 erros — perfil parecido com 6-rule |
| `IndexIVFScalarQuantizer` int8 (nprobe=2 e nprobe=4) | -250 sim | quantização introduz erro de boundary irrecuperável; +probes não compensa |
| `IndexIVFScalarQuantizer` fp16 | 0 sim | half memory (147→84MB) mas budget já tinha folga; detection cravada |
| `IndexRefineFlat` sobre IVFSQ8 (int8 wide + float32 rerank, k_factor=4) | **-6000 sim** | refine vectors carregam em heap (não mmap), 178MB+ por worker estoura budget → OOM → 51k timeouts |

### Fora do constraint do projeto

SIMD AVX2 escrito à mão, custom HTTP via io_uring, FD passing entre processos e busy-poll
em NAPI renderiam muito, mas violam "Python idiomático sem extensões nativas" e estão fora
de escopo. Linguagens compiladas em geral conseguem voar com KNN exato e cabem tudo em ~1ms
no Haswell, fora do nosso teto.

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
  p99 de 100ms→65ms, **sem perder detecção** (mesmos 2043).
- **Mais regras de fast-path têm retorno marginal** (testado em sim). Adicionar 4 regras
  fraude extras (km_home, installments, km_last, tx24h thresholds) leva cobertura pra 96.5%
  com mesma pureza, mas a fatia extra é facil pro KNN também — só economiza ~1.2µs avg por
  query no agregado. Sim mostrou neutro. Decision tree depth=5 (sklearn offline) acharia
  rules similares — projetado igual.
- **`IndexIVFScalarQuantizer` int8 perde recall irrecuperável**. Testado em sim: nprobe=2
  caiu de 3232→2969 (-263), nprobe=4 não compensou (2983). A quantização introduz erro de
  boundary que mais probes não recupera. Index size 147→53MB seria bom mas detection cara
  demais. Stay float32 IVFFlat.
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
