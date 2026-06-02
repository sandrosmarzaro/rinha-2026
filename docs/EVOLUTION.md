# Evolução do projeto

Histórico completo: cada plateau de score atingido, cada tentativa descartada, e os
findings algorítmicos que custaram tempo. O `CLAUDE.md` aponta pra cá pra contexto
profundo; este arquivo é o registro auditável da pesquisa.

## Mudanças que ficaram (plateaus)

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
| 8 | **3236** | `IndexIVFScalarQuantizer` fp16 storage | 0, mem -43% | prévia #7242. Mesma banda (3190-3233) mas index 147→84 MB, working set menor — folga pro tail sob pressão. Faiss decomprime pra float32 pra SIMD distance |
| 9 | **3279** | RSGI nativo (drop Starlette/ASGI) + `PYTHONOPTIMIZE=2` + `--http 1 --no-ws` | **+43** | prévia #7292 no `1bc7158`. p99 65→57 ms (-12%). Detection neutra (FP+1). Diff: -279/+24 LOC, removeu starlette dep. Body da issue precisa ser `rinha/test smarzaro-python` (não só o título — runner parsa o body) |
| 10 | **3691** | Single global IVF (nlist=2048, nprobe=12) + drop bbox sweep + Python partition_key (cleanup) | **+412** | prévia #7450 no `d4af075`. Detection 2038→2480 (FP 64→32 -50%, FN 18→7 -60%, failure_rate 0.15%→0.07% abaixo do floor). `rate_component=3000` no CAP máximo (error rate < MIN_EPSILON 0.001). **K-means coarse quantizer agrupa por L2 real** vs hand-crafted partition_key arbitrária — não precisa mais do bbox sweep pra compensar desalinhamento. 1 faiss.search/query (vs 1-9 antes). Profile revelou throttle externo do cgroup ser o gargalo k6 — isso libera CPU budget pra recall melhor (faiss_search avg 469→963µs interno mas k6 score subiu). Sim projetou +155, Haswell real entregou +412 (mais CPU = throttle menos severo). Diff -89 LOC (removeu bbox, fallback partition índices, extras_nprobe, MAX_EXTRA_PARTITIONS) |
| 11 | **3722** | K-means training `cp.niter=25 cp.nredo=4` (vs default 10/1) | **+31** | prévia #7619 no `2b0559e`. Detection 2480→2490 (FN 7→5 -29%, FP 32→34 +6%, net E 53→49 -7.5%). p99 63.7→58.5 ms (-8%). Sim avg 3707→3705 (dentro de ruído de 12 pts) mas detection determinística — 3 réplicas idênticas FP=34 FN=5. Real Haswell carregou a melhoria de detecção exatamente como previsto + ganho de p99 não modelado pelo sim. Custo: +8s no build, runtime 0 |
| 12 | **4091** | IVFFlat fp32 (drop SQ_fp16) + HAProxy CPU 0.10→0.20 (APIs 0.45→0.40 cada) | **+368** | prévia #7825 no `3bc922a`. Diagnose com `taskset -c 0` revelou que IndexIVFScalarQuantizer fp16 leva 798µs/query vs IVFFlat fp32 102µs (7.8×) — em 14-dim a decompressão fp16 domina. HAProxy a 0.10 estava CPU-starved → 1-5 ERR por sim run; a 0.20 ERR=0 sistemático. p99 58.5→25.7 ms (−56%); detection 2490→2501 (FP 34→33, FN 5→4). Index 84→184MB; runtime resident set 87/165MB no sim (mmap evict de cold pages mantém working set baixo). 5 sim replicates 3766-3802 (band 36 pts, ERR=0 todas), detection idêntica FP=33 FN=4 E=45 — sim previu real com precisão excepcional. Real Haswell entregou p99 ainda melhor que sim modelou (25.7ms vs sim 53ms) por ter mais CPU absoluto |
| 13 | **4177** | Pure numpy IVF (drop Faiss do runtime — só treina k-means no build) | **+86** | prévia #7906 no `e59c6e1`. Substrato totalmente em numpy: centroides + cluster-sorted vectors + norms pré-computadas; distância via norm-expansion (`||a-b||² = ||a||² + ||b||² - 2·a·b`), elimina alloc de `(block - q)` e crossing Faiss↔Python por query. Internal search_avg 957µs Faiss → 443µs numpy no sim (-54%); total p99 interno 1633→1205µs. Detection IDÊNTICA ao plateau #12 (FP=33 FN=4 E=45) — mesmas centroides + mesmo nprobe = mesmo recall. p99 real 25.7→21.1ms (-18%). **Hipótese da Lever 5 (numpy híbrido com Faiss centroid) confirmada às avessas**: aquele fracasso era do MIXING das duas engines sob CPU regimes diferentes; single-engine numpy reproduz sim→real direção. 5 sim replicates 3805-3844, identical detection. Levers já testadas em cima do Faiss (`Adaptive nprobe`, `Cell-mean`, `Tree pre-classifier`) precisam ser re-avaliadas no novo motor — comportamento sob numpy puro pode ser diferente |

## Tentativas em cima do single-IVF (todas descartadas)

| Tentativa | Resultado | Por quê |
|---|---:|---|
| Per-cluster purity skip (`quantizer.search(q,1)` + cluster_majority lookup; 1825/2048 clusters puros @ 99.9%, 1786 @ 1.0) | -57 sim | 62.5% das boundary cortaram faiss (478→963µs cut), MAS detection regrediu (FP +5, FN +1). Pure cluster ≠ correct K-NN: a true query pode estar mais perto de neighbors em OUTRO cluster que o IVF sweep encontraria. **Lossy fundamental** |
| nlist tuning (1024, 4096) | 0 sim | (2048, nprobe=12) já saturada. (4096, 24) mesmo total scan: tied. (1024, 6) coarser: tied/pior |
| `--runtime-mode mt --runtime-threads 2` no single-IVF | -7 sim, p99 +6ms | Threads overhead > paralelismo. GIL + faiss interno C++ não compõem bem |
| `MAP_POPULATE` no `global.faiss` (single mmap contígua) | 0 sim | Page cache aquece organicamente nos 2min de ramp do k6 |
| vectorize internals (manual ISO parse) | -217 ns/req | microbench Py 3.14: `fromisoformat` 416ns < slice manual 633ns. Datetime tem C-impl otimizado, manual python perde. Mesmo se ganhasse, seria 0.05µs avg agregado em 10% boundary |
| Pre-allocated `np.empty(14)` buffer (singleton st-mode) | 0 sim (3709 vs banda 3701-3713) | Alloc custa ~272ns/call mas em 10% boundary = 0.03µs avg agregado, dentro de ruído estatístico |
| nprobe sweep fino (8/10/14 vs 12) com 2 réplicas cada | 0 sim | nprobe=10 +13 avg (banda 30 pts run-to-run), nprobe=14 melhor detection (FN=5) mas p99 +7ms cancela. Pico ainda em 10-12 |
| CPU pinning (`cpuset: '0'`/`'1'` por container) | 0 sim | Sim host tem cores sobrando (sem contenção L1/L2 pra reduzir); real Rinha é single-core onde cpuset é no-op anyway |
| Cluster homogeneous skip @ 1.0 floor (replicate com niter=25) | descartado por prior data | 1784/2048 clusters pure @ 1.0 com niter=25 (vs 1786 com niter=10) — k-means mais firme não muda estrutura de purity. Ceiling estrutural: boundary query pode ter true K-NN em cluster vizinho mesmo com cluster próprio puro |
| Decision tree pre-classifier (depth=10, min_leaf=50, sample=500k, conf=0.95) APÓS as 2 regras | -100 a -150 sim | Tree treina em references e cobre 53% do boundary com 99.6% accuracy em test-data, mas sob k6 saturado FP saltou 34→43-49 e ERR variou 1-5 (vs baseline 0-3). Detection 2382→2257 avg. Faiss KNN sobre 5 vizinhos é estritamente melhor que tree no boundary porque o que sobrou após 2-rule é exatamente a fatia ambígua onde recall importa |
| Adaptive nprobe (LOW=6 primário, HIGH=16 quando fc∈{2,3}) | 0 sim, faiss avg +18% | Profile revelou que ~51% das queries boundary triggam HIGH pass — boundary é intrinsecamente ambíguo após 2-rule já cortar o fácil. avg faiss subiu 1082→1272µs em vez de cair. Variantes (LOW=4/8, HIGH=12, ambig={1,2,3,4}) todas pioraram. Lever errado pro nosso shape de boundary |
| Cell-mean fast-path sobre IVF cluster (margem 0.01) | -120 sim | Mesmo problema do tree: o 7.4% que sobra após 2-rule é o conjunto onde majority-vote do cluster diverge do KNN-exato. FN +5 estrutural (frauds que vivem em clusters predominantemente legit), FP +4. Detection 2382→2216 avg |
| Tree 100% (substitui as 2 regras): vectorize → tree → Faiss | 0-100 sim, FP=42-57 | Tree em todas as queries pega 95.4% (vs 92.6% do 2-rule), corta Faiss frequency 54% (2715→1231). MAS FP saltou 34→49-57 consistente (conf=0.95 ou 0.99). Tree é fundamentalmente menos preciso que 2-rule no espaço amount/avg + amount: 2-rule usa thresholds hand-derivados num espaço raw que generaliza melhor pro test que árvore treinada em references vetorizadas. Projeção real: det 2410-2360 vs 2490, -82 a -130 |
| Numpy IVF híbrido (Faiss só pro quantizer, cluster scan em numpy fp32 mmap) | **−74 real, +34 sim** | prévia #7873 no `524e322`: 4017 vs plateau #12 4091. Sim mostrou ganho ilusório (avg 3810 vs 3776, faiss_avg interno 957→731µs no sim), mas Haswell real entregou p99 30.5 vs 25.7ms (+5ms). Faiss IVFFlat tem AVX2 interno que se beneficia mais com quota CPU real (0.40) vs sim throttle (0.16). Bench bare-metal já dizia numpy seria mais lento (281µs vs 102µs); sim throttled inverteu por causa de overhead Python/Faiss diferente sob CPU starvation, mas real Haswell restaura a hierarquia. **Lição**: sim → real diverge quando muda o substrato algorítmico (não só parâmetros). Sim só prevê confiável dentro do mesmo motor. Revertido em `8fddef6` |
| `IndexIVFFlat` fp32 wide pre-plateau-12 (revert plateau 8 isoladamente, antes do HA bump) | 0 sim, +100MB | Mesma detection que fp16 (não-lossy a 14 dims) mas index 84→184MB. Sozinho não pagava — só quando combinado com HA 0.10→0.20 (plateau #12) virou win. **Levers compõem mesmo quando isolados parecem neutros** |
| Smart ε-dedup das refs (eps=0.05, 3M→1.27M, mesma-label balls collapsed) | **−36 real, +17 sim** | prévia #7921 no `1d4fe1d`: 4141 vs plateau #13 4177. Sim mostrou ganho marginal (avg 3850 vs 3833) com detection IMPROVED (FN 4→0, FP 33→41, E 45→41, det 2501→2513). Mas em real Haswell p99 PIOROU (21.1→23.6ms), gerando p99_score -48 que come o detection +12. Hipótese: dataset menor altera padrão de acesso mmap de forma que a working set fica mais fria; ou variância natural prévia-a-prévia (~10-15%). Random subsample 50% testado como controle: catastrófico (E saltou 45→1271), refs NÃO são globalmente redundantes — só locais em ε-balls puras. eps=0.02 (76% kept) e eps=0.10 (42% kept) ambos piores. Revertido em `e49b1ef` |

## Descartados pré-single-IVF (cada um testado, com motivo)

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
| `IndexRefineFlat` sobre IVFSQ8 (int8 wide + float32 rerank, k_factor=4) | **-6000 sim** | refine vectors carregam em heap (não mmap), 178MB+ por worker estoura budget → OOM → 51k timeouts |
| Custom rerank em Python sobre `vectors.npy` mmap'd (IVFSQ8 wide + numpy rerank, k_factor=2/4) | -10 sim, +37MB | funciona (sem -6000 do RefineFlat porque mmap), detection +10 (rerank exato recupera int8). Mas p99 +5ms e index 84→221MB. Net negativo vs fp16. **Concept-proven mas pior que baseline.** |
| Per-cluster bbox via Faiss `invlists`/`search_preassigned` | descartado sem implementar | requer compute por query de bbox lb pra ≤1024 clusters em Python (~5µs/query). Mesmo padrão dos experimentos always-bbox/smart-exit/neighbor-list que perderam: trabalho Python per-query sob saturação no Haswell ampliar p99 mais do que pruning algorítmico salva. Marginal mesmo se ganhasse algo. |
| Granian `--backpressure 256 --http1-buffer-size 8192` | 0 sim | Concorrência real ~22 in-flight, backpressure 256 nunca aciona; buffer menor para responses de 50B não pressiona |
| `MAP_POPULATE` em todos os `.faiss` no startup (single syscall, alternativa limpa ao pre-touch byte-a-byte) | 0 sim | Os 2 min de ramp-up do k6 já aquecem page cache organicamente; pre-fault não muda tail |
| msgspec `gc=False` em todas as Structs + drop unused `FraudRequest.id` | 0 sim | gc=False é hint sem mudança semântica; struct decode já era ~3µs/req — saving sub-µs invisível |
| `B+C+D` empilhados | **-41 sim** (regrediu) | Stacking não compõe; run-to-run variance amplifica quando tudo é ruído individual. Não há sinal a tirar |
| `mlock` no índice via `ctypes.mlock` + `ulimits.memlock:-1` (~80 MB locked, 90/165 MB no cgroup) | 0 sim | Funciona técnicamente, mas sim sem competição por page cache não modela o cenário onde mlock paga. No real Haswell (8 GB RAM, sem outras workloads), também provavelmente neutro — page cache já não evita pages do índice |
| `FRAUD_THRESHOLD 0.6 → 0.4` (reclassifica score 0.4 = 2/5 vizinhos fraude → fraude) | **-725 sim** | FP saltou 63→373, FN caiu 17→14. Score 0.4 é majoritariamente real-legit (8% fraude); custo FP×1 (+310) >> ganho FN×3 (+9) |
| `K_NEIGHBORS 5 → 3` | **-810 sim** | Granularidade pior (scores 0, 0.33, 0.67, 1.0): FP+115, FN+85 simultâneo. Menos vizinhos = boundary mais ruidoso em ambas direções. Latência ganho mínimo (~1ms) |
| Grid search das 2 regras de fast-path (200 thresholds × 14 dims, piso pureza 99.99%) | 0 (sem alternativa melhor) | Pair atual já near-optimal: joint cov 92.57%/99.99% pureza. Best grid alternative `amount/avg≤0.5` + `amount>3061`: 92.29% (-0.28%). `hour≤5h→fraud` interessante mas pior joint. Lever saturado |
| K-means fast-path (replace 2 rules) | descartado sem testar | Forçaria `vectorize` (~3-5µs) ANTES da decisão pra 92.6% das queries, vs 0.5µs da 2-rule atual em raw payload. Como segunda camada após 2-rule só atinge 7.4% (boundary hard) onde K-means não bate KNN. Análise neutra/negativa antes de implementar |
| `IndexHNSWSQ` fp16 M=8 efSearch=16 por partição | **-422 sim** | Index 84→247 MB blowing budget (api-1 97% mem), p99 spikes 51→101ms entre runs. Detection regride -224 (FP+20, FN+12) por M=8 sparso. HNSW + 14-dim + 84 partições é regime errado pra HNSW; IVF vence |
| PCA 14→8 dims (project pré-search) | **-1216 sim** | Detection colapsa 2043→835, erros saltam 81→490. Features são todas discriminativas (sem redundância); 14→8 joga signal fora. Index cai 84→57 MB e p99 fica igual, mas detection é catastrofica. Confirma que 14-dim é mínimo |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 FAISS_NUM_THREADS=1` | 0 sim | faiss/numpy não estavam multithreading em ops de 14-dim de qualquer jeito |
| `LD_PRELOAD=libmimalloc.so.3` | -9 sim (ruido) | Alloc patterns em Python são dominados por interior structures (objects, dicts); allocator swap não move |
| `madvise(MADV_RANDOM)` em faiss mmap | inviável | Faiss owns o mmap e não expõe pointer pra advise externo; precisaria patch no Faiss |
| Recompilar faiss-cpu com `-march=haswell` | desnecessário | Wheel oficial `faiss-cpu 1.14.2` já contém `OPTIMIZE DD AVX2` (Dispatcher Dinâmico), runtime já usa AVX2 path em Haswell |
| HAProxy `maxconn 64` por backend | 0 sim | Concorrência real ~22 in-flight, threshold 64 nunca aciona, sem 503s, sem efeito no tail |
| `MAP_HUGETLB` explícito (2MB pages) | inviável | Host (e provavelmente runner) sem `nr_hugepages` reservados; pedido falha com ENOMEM. THP via `madvise(MADV_HUGEPAGE)` já aplicado |
| `faiss.omp_set_num_threads(1)` no init | 0 sim | Mesmo efeito de `OMP_NUM_THREADS=1`: faiss/numpy não multithreading em 14-dim |
| `partition_key` direto do payload raw (pulando vectorize na boundary) | descartado análise | Homogeneous partitions são extremos (todos labels iguais); boundary queries são por definição ambíguas → raramente caem em homogeneous. Adiciona ~2µs em 79% do boundary, economiza ~3µs em 21% → net negativo |
| Granian `--runtime-mode mt --runtime-threads 2` | 0 sim | Profile revelou que gap k6 vs interno (50ms) é **CPU throttle do cgroup**, não serialização do event loop. MT compartilha mesma quota → não move |
| Granian `--workers 2` por container (4 total) | 0 sim, memória 90% | Mesma quota CPU compartilhada → sem ganho. Memória aperta (150/165 MB) e ainda é risco de OOM |
| `MAX_EXTRA_PARTITIONS 8 → 4` | -261 sim | Profile mostrou faiss_search interno caiu 30% (1093→768µs p99), MAS k6 p99 IGUAL (51ms). **Comprova: gargalo é CPU throttle externo, não trabalho do handler.** Detection colapsou (FP+23, FN+15) pq cortar bbox cap mata recall |
| Granian ASGI raw (sem Starlette) | -34 sim | RSGI nativo é mais leve que ASGI mesmo dentro do Granian (não foi só dropar Starlette) — protocolo mais compacto, menos alloc por request |
| Uvicorn ASGI raw (sem Starlette) | -32 sim | ≈ Granian ASGI raw. Não bate Granian RSGI; Rust core do Granian + protocolo RSGI compactam mais |
| Hypercorn ASGI raw (sem Starlette) | -192 sim | Server Python-puro lento; p99 80ms vs 51ms do Granian RSGI |

## Findings algorítmicos da evolução

Findings que documentam decisões já tomadas. Pra regras vivas (gotchas, restrições de
infra, padrões de medição), ver `CLAUDE.md`.

- **Faiss IVFFlat + mmap** (`read_index(IO_FLAG_MMAP)`): ~0.3 ms/query, page cache
  compartilhado entre os 2 workers. Os vetores moram no índice (não guardar `.npy` separado).
- **scipy cKDTree** converte para float64 interno (estoura RAM) e é incompatível com mmap.
  Descartado.
- **int16** exige `astype(int32)` por query → aloca no hot path → lento. float32 + BLAS vence.
- **nprobe muda de ótimo conforme o resto do search evolui** (baked em `build_index.py`).
  Sob nprobe-only (sem bbox): nprobe=1 venceu (+790 vs nprobe=8: 738→1528) — cada lista IVF
  extra vira latência direta sob saturação. Depois do bbox-prune + unanimous-exit os fáceis
  saem rápido, então sobra orçamento pra recall maior nos boundaries: nprobe=2 venceu
  nprobe=1 (2921 vs 2512). nprobe=4 já piora (latência custa mais que a detecção rende).
  Após single-IVF (plateau 10+): nprobe=12 é o ótimo.
- **Asymmetric nprobe entre primary e extras** (era arquitetura per-partition, pré single-IVF):
  a primária paga a vasta maioria das queries via unanimous-exit, então custo dela domina o p99
  (mantém nprobe=2). Os extras só rodam em queries ambíguas após filtro do bbox — vale a pena
  cavar mais fundo neles. `EXTRAS_NPROBE=3` é o ótimo (+124 real: 2921→3045). Subir a 4 ou 5
  destrói o p99 sem ganho. Tradeoff dominante é recall-vs-latência nos extras.
- **Levers estruturais que falharam** (pré single-IVF):
  - Partição mais fina (512 chaves, amount em 3 bits): topo do amount é denso pelo clamp em
    10000 → split de só 26% no max, mas vizinhos se separam entre partições → recall caiu
    mais que a latência ganhou. **Regressão.**
  - Quantizador grosso HNSW (`IndexHNSWFlat` no IVF): `IndexFlatL2` em 14-dim já é SIMD,
    o passo grosso não dominava. ~30 µs vs ~30 µs. **Sem ganho.**
  - `IndexFlatL2` puro por partição (KNN exato dentro): satura o Haswell catastroficamente,
    93% timeout, **-6000**.
  - Feature weights por dim usando `|ρ|` (correlação com label): com métrica uniforme já
    100% acurada, qualquer perturbação só piora os k-vizinhos. **Regressão (-116).**
- **KNN k=5 EXATO dá 100% de accuracy sobre as references** (verificado em 2000 queries do
  test-data via brute-force BLAS, 0 TP/TN/FP/FN). Não tem regra escondida, não tem teto do
  KNN: o teto era a APROXIMAÇÃO do IVF. A combinação nprobe=2 + bbox cross-partition +
  unanimous-exit recupera quase todo o recall perdido (1.77% → 0.20% failure).
- **Bounding-box pruning cross-partition** (era em `search.py` pré-single-IVF; removido em
  plateau 10): salva `min`/`max` por dim por partição em `meta.json`; em runtime, depois da
  primária, calcula `lb_sq = ||q − bbox||²` vetorizado e visita só partições com
  `lb_sq < worst-of-top-K`, com merge incremental dos top-K e cap `MAX_EXTRA_PARTITIONS=8`.
  Sozinho custa ~127 pts de latência por query (`np.maximum`/`einsum`/`argsort`) → emparelhado
  com **early-exit por voto unânime** na primária para queries de alta confiança pularem o
  sweep. Combo deu **+815 no real** (1697 → 2512), detecção 560 → 1383, p99 quase inalterado.
- **Verificar unanime via bbox global tem teto mas custa caro demais** (pré-single-IVF).
  Sempre computar bbox (mesmo nas unanimes) recupera ~36 wrong-unanime-exits — detecção sobe
  ~360. Mas o numpy per-query sob saturação no Haswell joga p99 de 100ms pra 560-735ms (mesmo
  com neighbor list pré-computada por partição). Net negativo. A imprecisão das unanimes é o
  preço a pagar.
- **Fast-path determinístico antes do KNN** (2 thresholds hand-derivados em `rsgi_app.py`):
  `amount/avg ≤ 0.971` → legit; `amount > 2996` → fraud. Cobrem ~92.6% das queries com 99.99%
  accuracy em references. Pula vectorize + partition_key + Faiss inteiro pra essas queries —
  só ~7.4% boundary cai no KNN. Custou +188 real (3045→3233) cortando p99 de 100ms→65ms,
  **sem perder detecção** (mesmos 2043).
- **Mais regras de fast-path têm retorno marginal** (testado em sim). Adicionar 4 regras
  fraude extras (km_home, installments, km_last, tx24h thresholds) leva cobertura pra 96.5%
  com mesma pureza, mas a fatia extra é facil pro KNN também — só economiza ~1.2µs avg por
  query no agregado. Sim mostrou neutro. Decision tree depth=5 (sklearn offline) acharia
  rules similares — projetado igual.
- **`IndexIVFScalarQuantizer` int8 perde recall irrecuperável**. Testado em sim: nprobe=2
  caiu de 3232→2969 (-263), nprobe=4 não compensou (2983). A quantização introduz erro de
  boundary que mais probes não recupera. Index size 147→53MB seria bom mas detection cara
  demais. Stay fp32 IVFFlat (plateau 12 confirmou fp16 também perde — em 14-dim qualquer SQ
  decompression custa mais do que a economia de memória).
- **fp16 vs fp32 em 14-dim** (descoberto no diagnose pre-plateau-12): `IndexIVFScalarQuantizer`
  fp16 leva 798µs/query bare-metal vs `IndexIVFFlat` fp32 102µs (7.8× mais lento). Em alta
  dimensionalidade (100+) a decompressão se amortiza sobre mais SIMD work, mas em 14-dim ela
  domina. Memória 84→184MB foi compensada por mmap evict de cold pages (resident set ficou
  estável em ~87MB no sim).
- **Sim throttled prediz real só dentro do mesmo motor**. Mudanças de parâmetro (niter, nprobe,
  CPU split, fp16↔fp32 dentro de Faiss) traduzem bem. Mudanças de substrato (Faiss → numpy
  custom) divergem: numpy IVF híbrido mostrou +34 sim e −74 real porque sim throttle (0.16 CPU)
  amplifica Python overhead diferentemente do real Haswell (0.40 CPU). Bench bare-metal
  (`taskset -c 0`) costuma estar mais alinhada ao real que sim throttled pra essas escolhas.
