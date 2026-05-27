# PRD — Rinha de Backend 2026

Documento de design. Descreve o sistema como construído e o racional de cada decisão.

---

## 1. Contexto e objetivo

A Rinha de Backend 2026 propõe uma API de detecção de fraude transacional via **busca
vetorial**. Cada requisição traz dados de uma transação de cartão; a API responde se
aprovou e qual o score de fraude (0..1).

Cada payload vira um vetor de 14 dimensões normalizadas, comparado contra ~3M vetores
rotulados de referência. A label dos 5 vizinhos mais próximos determina o score
(`fraudes / 5`). Approved se `score < 0.6`.

**Objetivo:** submissão em **Python idiomático**, sem extensões nativas custom nem
truques de baixo nível (LB próprio, SCM_RIGHTS, SIMD manual). Tirar o máximo de Python:
Starlette + msgspec + Granian no hot path, Faiss + mmap na busca.

---

## 2. Restrições

| Item | Limite | Origem |
|---|---|---|
| CPU total (todos serviços) | **1.0 CPU** | ARQUITETURA.md |
| RAM total (todos serviços) | **350 MB** | ARQUITETURA.md |
| Porta exposta | **9999** (LB) | API.md |
| Network mode | **bridge** (host/privileged proibidos) | ARQUITETURA.md |
| Topologia mínima | 1 LB + 2 APIs em round-robin | ARQUITETURA.md |
| Imagens Docker | públicas, linux/amd64 | FAQ.md |
| Hardware do teste | Mac Mini 2014, 2.6 GHz Haswell, 8 GB | SUBMISSAO.md |

**Proibições explícitas:**

- LB não pode inspecionar payload, ter lógica condicional ou responder antes de repassar.
- Não usar payloads do teste como referência ou lookup.
- Repositório público com licença MIT.

**Penalidades duras (score):**

- `(FP + FN + Err) / N > 15%` ⇒ score de detecção = -3000 (independe da latência).
- p99 > 2000 ms ⇒ score p99 = -3000.
- Erro ponderado: FP=1, FN=3, Err (HTTP) = 5.

**Implicação operacional:** estabilidade > latência. Um bug que cause HTTP 500 em 16% das
requisições zera o score de detecção. Logging silencioso e fallback obrigatórios.

---

## 3. Arquitetura

```
                 :9999
                   │
                   ▼
        ┌──────────────────┐
        │     HAProxy      │  0.10 CPU / 15 MB
        │   (round-robin)  │
        └────┬─────────┬───┘
             │ UDS     │ UDS
             ▼         ▼
       ┌──────────┐ ┌──────────┐
       │ Granian  │ │ Granian  │  0.45 CPU / 110 MB cada
       │  + api1  │ │  + api2  │
       └────┬─────┘ └────┬─────┘
            │            │
            └─────┬──────┘
                  │ faiss.read_index(IO_FLAG_MMAP)
                  ▼
            ┌──────────────┐
            │ data/index/  │  ~141 MB (page cache compartilhado)
            │  (read-only) │
            └──────────────┘
```

| Serviço | CPU | RAM (limit) | Justificativa |
|---|---|---|---|
| HAProxy | 0.10 | 15 MB | Single-process, hot path mínimo |
| api1 (Granian + Starlette) | 0.45 | 110 MB | Granian + numpy/Faiss + working set |
| api2 (Granian + Starlette) | 0.45 | 110 MB | Idem |
| **Total** | **1.00** | **~235 MB** | margem confortável |

O page cache do mmap é compartilhado entre os 2 workers — cada um carrega só as páginas
que toca. RSS medido fica em ~71 MB por worker.

**Comunicação:**

- HAProxy ↔ APIs: **UDS** em `/tmp/sockets/{api1,api2}.sock`, volume tmpfs compartilhado
  pelos 3 containers. Evita o overhead de TCP localhost.
- APIs ↔ disco: `data/index/` montado read-only, acessado via mmap.

---

## 4. Pipeline de dados

### 4.1. Build-time (`scripts/build_index.py`)

Roda uma vez durante o `docker build`. Etapas:

1. Carregar `references.json.gz` (~3M registros, vetor + label).
2. Computar a **chave de partição 8-bit** (ver 5.1) para cada registro.
3. Ordenar por chave → registros da mesma partição ficam contíguos.
4. Para cada partição não-vazia e não-homogênea, construir um índice Faiss:
   - `n < max(40, 8·nlist)` ⇒ `IndexFlatL2` (brute force exato, SIMD via BLAS).
   - caso contrário ⇒ `IndexIVFFlat` (k-means + listas invertidas), `nlist = n // 400`
     (cap 1024), `nprobe = 8`.
5. Serializar em `data/index/`:
   - `labels.npy` — uint8 `(N,)`, labels reordenadas por chave de partição.
   - `meta.json` — `boundaries`, `fallbacks`, `homogeneous_score`, `ivf_nprobe`.
   - `faiss/<key>.faiss` — um índice por partição (os vetores vivem dentro do índice).

Idempotente: pula o rebuild se os arquivos de saída são mais novos que a fonte.

### 4.2. Runtime (startup, `fraud_api/index.py`)

1. `np.load(labels.npy, mmap_mode='r')` + `madvise(MADV_HUGEPAGE | MADV_WILLNEED)`.
2. `faiss.read_index(path, IO_FLAG_MMAP)` por partição — as páginas ficam no page cache
   compartilhado.
3. Warmup no lifespan: uma query dummy em cada índice para faultar as páginas e primar
   os internals antes de o LB liberar tráfego.
4. `/ready` só responde 200 após o lifespan completar.

### 4.3. Por que Faiss float32 + mmap

- `IndexIVFFlat` faz ~0.3 ms por query: 1 CPU sustenta 900 RPS com folga.
- mmap compartilha o page cache entre os 2 workers — a maior parte dos 141 MB não é
  duplicada por processo.
- float32 usa o SIMD do BLAS nativamente. Quantização int16 foi descartada: o cast
  `astype(int32)` necessário por query aloca memória no hot path e destrói a latência.

---

## 5. Algoritmo

### 5.1. Chave de partição (8 bits)

Derivada do vetor normalizado em O(1):

| Bits | Campo | Valores |
|---|---|---|
| 0 | `is_online` | 0/1 |
| 1 | `card_present` | 0/1 |
| 2 | `unknown_merchant` | 0/1 |
| 3-4 | bucket de `amount` | 0..3 (cuts em 0.005, 0.020, 0.100) |
| 5-7 | bucket de risco do MCC | 0..7 (cuts 0.1..0.8) |

256 partições possíveis, distribuição enviesada. Partições vazias recebem fallback (5.3).

### 5.2. Query path (`fraud_api/search.py`)

```
1. Parse body            (msgspec)
2. Vetorizar → float32[14]
3. partition_key 8-bit
4. resolver fallback (partição vazia → vizinha por Hamming)
5. early-exit se a partição é homogênea (todos fraud ou todos legit)
6. idx.search(q, k=5)    (Faiss IVF/Flat)
7. score = labels[neighbors].sum() / 5
8. resposta pré-renderizada por score (K+1 respostas possíveis)
```

### 5.3. Fallback para partição vazia

`compute_fallbacks` mapeia cada chave vazia para a partição não-vazia mais próxima por
distância de Hamming (empate resolvido pela maior partição, melhor cobertura de KNN).

### 5.4. Early-exit homogêneo

Partições onde todos os vetores têm a mesma label retornam o score constante sem busca.
Pega uma fração relevante das queries a custo zero.

---

## 6. Stack web

### 6.1. Estrutura

```
fraud_api/             # PEP 420 namespace package, sem __init__.py
  app.py              # Starlette + lifespan (build_app_data + warmup) + routes
  handlers.py         # /ready, /fraud-score + respostas pré-renderizadas
  schemas.py          # msgspec.Struct request/response
  state.py            # AppData + build_app_data() (disco ou sintético)
  index.py            # loader mmap de labels + índices Faiss
  search.py           # query path KNN + oracle brute-force (paridade)
  vectorize.py        # payload → vetor 14-dim
  partition.py        # chave 8-bit + fallback Hamming
scripts/
  download_data.py    # baixa references/mcc_risk/normalization
  build_index.py      # pré-processamento build-time
  tune_nprobe.sh      # varredura de nprobe via k6
tests/
  test_contract.py
  test_parity.py
docs/
  PRD.md
```

### 6.2. msgspec em Starlette

Starlette não integra msgspec nativamente. Glue minimalista: `Decoder`/`Encoder`
cacheados em escopo de módulo. Como `fraud_score` só pode ser `count/5` com `count` em
`0..5`, há apenas 6 respostas possíveis — pré-renderizadas uma vez no import; o hot path
só indexa nelas.

### 6.3. Granian

`granian --interface asgi --uds <sock> --workers 1 --runtime-mode st --log-level warning`

- `--interface asgi`: protocolo padrão suportado por Starlette.
- `--uds`: Unix socket em vez de TCP localhost.
- `--workers 1` + `--runtime-mode st`: 1 worker single-threaded por container; em 1 CPU,
  paralelismo extra vira contenção.

### 6.4. Starlette app

`AppData` é um `@dataclass(slots=True, frozen=True)` carregado uma vez no lifespan e
acessado via `request.app.state.data`. O lifespan garante que requests só chegam depois do
startup — `/ready` apenas retorna 200.

---

## 7. Load balancer (HAProxy)

`nbthread 1`, round-robin sobre os dois UDS, `http-check send meth GET uri /ready` para só
liberar tráfego depois que a API responde 2xx, `option http-keep-alive` para reusar a
conexão LB↔backend. `umask 000` no entrypoint para o HAProxy (non-root) conseguir escrever
no socket compartilhado.

---

## 8. Testes

- **Contract** (`test_contract.py`): payload de exemplo → resposta com shape válido
  (`approved: bool`, `0 ≤ fraud_score ≤ 1`); `/ready` → 200.
- **Paridade** (`test_parity.py`): oracle brute-force float32 numpy vs implementação
  particionada Faiss sobre 1000 queries amostradas. Decisão final diverge em < 1%.
  É o teste que paga: um bug em chave de partição, fallback ou busca quebra paridade.
  Skipa quando o dataset completo não está montado.

---

## 9. Riscos e tradeoffs

| Risco | Mitigação |
|---|---|
| Python tem overhead no hot path | handler mínimo, msgspec direto, respostas pré-renderizadas |
| GIL serializa Python puro | 1 worker/container × 2; Faiss libera o GIL na busca |
| mmap não conta no limit do Docker | limit por API com margem; medir `docker stats` sob carga |
| Partição muito enviesada | fallback Hamming para partições adjacentes |
| Startup lento | índices pré-construídos no build, mmap + warmup no lifespan |
| Bug silencioso na vetorização | teste de paridade obrigatório |

### Alternativas descartadas

- **scipy cKDTree**: converte os vetores para float64 internamente (cópia que estoura o
  budget de RAM) e é incompatível com mmap. Substituído por Faiss.
- **Quantização int16**: cast por query aloca no hot path; float32 + BLAS é mais rápido.
- **HNSW**: memória do grafo estoura o budget; recall não compensa em 14 dims.
- **LB custom / SCM_RIGHTS / SIMD manual**: fora da meta de Python idiomático.
- **Litestar + RSGI**: RSGI nunca chegou ao master (issue #3423); Litestar expõe só ASGI.
  Trocado por Starlette + glue msgspec.
- **Modelo treinado offline (GBDT/MLP)**: viável sobre `references.json.gz` (não payloads
  de teste). Backlog — caminho para superar o teto de detecção do KNN.
