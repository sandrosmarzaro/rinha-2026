//! IVF KNN scan kernel for the Rinha de Backend 2026 fraud-detection API.
//!
//! Compute path (per query):
//!   1. Receive selected probe cells (NPROBE clusters).
//!   2. For every ref in those clusters, compute ||q - v||² using the
//!      norm-expansion identity over int16-quantized vectors:
//!         dist = q_norm + v_norm − 2·(q · v)
//!   3. Maintain a top-5 buffer by distance.
//!   4. Return the fraud-count (0..5) of the top-5 labels.
//!
//! This is the scalar/portable implementation. The AVX2 SIMD path is gated
//! behind a runtime check in `is_x86_feature_detected!("avx2")`.

use numpy::IntoPyArray;
use numpy::PyArray1;
use numpy::PyReadonlyArray1;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

const PADDED_DIM: usize = 16;
const QUANT_SCALE: f32 = 10_000.0;
const MAX_AMOUNT: f32 = 10_000.0;
const MAX_INSTALLMENTS: f32 = 12.0;
const AMOUNT_VS_AVG_RATIO: f32 = 10.0;
const MAX_MINUTES: f32 = 1440.0;
const MAX_KM: f32 = 1000.0;
const MAX_TX_COUNT_24H: f32 = 20.0;
const MAX_MERCHANT_AVG_AMOUNT: f32 = 10_000.0;
const HOURS_DIVISOR: f32 = 23.0;
const WEEKDAY_DIVISOR: f32 = 6.0;
const MISSING_SENTINEL_SCALED: i16 = -10_000;

#[inline(always)]
fn parse_u32_n(bytes: &[u8]) -> u32 {
    let mut v = 0u32;
    for &b in bytes {
        v = v * 10 + (b - b'0') as u32;
    }
    v
}

#[inline]
fn iso_to_epoch_seconds(s: &[u8]) -> i64 {
    let year = parse_u32_n(&s[0..4]) as i32;
    let month = parse_u32_n(&s[5..7]);
    let day = parse_u32_n(&s[8..10]);
    let hour = parse_u32_n(&s[11..13]);
    let minute = parse_u32_n(&s[14..16]);
    let second = parse_u32_n(&s[17..19]);
    let y = year - if month <= 2 { 1 } else { 0 };
    let era = if y >= 0 { y / 400 } else { (y - 399) / 400 };
    let yoe = (y - era * 400) as u32;
    let doy = (153 * (if month > 2 { month - 3 } else { month + 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era as i64 * 146097 + doe as i64 - 719468;
    days * 86400 + hour as i64 * 3600 + minute as i64 * 60 + second as i64
}

#[inline]
fn weekday_mon0(s: &[u8]) -> u32 {
    let year = parse_u32_n(&s[0..4]) as i32;
    let month = parse_u32_n(&s[5..7]);
    let day = parse_u32_n(&s[8..10]);
    let y = year - if month <= 2 { 1 } else { 0 };
    let era = if y >= 0 { y / 400 } else { (y - 399) / 400 };
    let yoe = (y - era * 400) as u32;
    let doy = (153 * (if month > 2 { month - 3 } else { month + 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era as i64 * 146097 + doe as i64 - 719468;
    ((days + 3).rem_euclid(7)) as u32
}

#[inline(always)]
fn q_round_clamp01(x: f32) -> i16 {
    let c = x.clamp(0.0, 1.0);
    (c * QUANT_SCALE).round() as i16
}

#[inline(always)]
fn q_round_signed(x: f32) -> i16 {
    let c = x.clamp(-1.0, 1.0);
    (c * QUANT_SCALE).round() as i16
}

#[inline]
fn mcc_risk(mcc: &[u8]) -> f32 {
    match mcc {
        b"4511" => 0.35,
        b"5311" => 0.25,
        b"5411" => 0.15,
        b"5812" => 0.30,
        b"5912" => 0.20,
        b"5944" => 0.45,
        b"5999" => 0.50,
        b"7801" => 0.80,
        b"7802" => 0.75,
        b"7995" => 0.85,
        _ => 0.50,
    }
}

/// Insertion-sort-style top-5 buffer keyed by distance.
#[derive(Copy, Clone)]
struct Top5 {
    dists: [i64; 5],
    labels: [u8; 5],
    filled: usize,
}

impl Top5 {
    #[inline(always)]
    fn new() -> Self {
        Top5 {
            dists: [i64::MAX; 5],
            labels: [0; 5],
            filled: 0,
        }
    }

    /// Insert (dist, label) maintaining sorted order ascending by dist.
    /// O(5) per insertion.
    #[inline(always)]
    fn push(&mut self, dist: i64, label: u8) {
        if self.filled < 5 {
            // Insert preserving order.
            let mut i = self.filled;
            while i > 0 && self.dists[i - 1] > dist {
                self.dists[i] = self.dists[i - 1];
                self.labels[i] = self.labels[i - 1];
                i -= 1;
            }
            self.dists[i] = dist;
            self.labels[i] = label;
            self.filled += 1;
        } else if dist < self.dists[4] {
            // Replace last and bubble up.
            let mut i = 4;
            while i > 0 && self.dists[i - 1] > dist {
                self.dists[i] = self.dists[i - 1];
                self.labels[i] = self.labels[i - 1];
                i -= 1;
            }
            self.dists[i] = dist;
            self.labels[i] = label;
        }
    }

    #[inline(always)]
    fn worst(&self) -> i64 {
        if self.filled < 5 {
            i64::MAX
        } else {
            self.dists[4]
        }
    }

    #[inline(always)]
    fn fraud_count(&self) -> i32 {
        let mut c = 0;
        for i in 0..self.filled {
            c += self.labels[i] as i32;
        }
        c
    }
}

/// Scalar i32-accumulator distance: ||q - v||² over PADDED_DIM lanes (last
/// padded ones must be zero in both q and v).
#[inline(always)]
fn dist_scalar(q: &[i16], v: &[i16]) -> i64 {
    let mut acc: i64 = 0;
    let n = q.len();
    for i in 0..n {
        let d = q[i] as i32 - v[i] as i32;
        acc += (d * d) as i64;
    }
    acc
}

/// AVX2-vectorized distance over 16 i16 lanes per vector.
/// Returns ||q - v||² as i64. Padded zero lanes in both inputs cancel out.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dist_avx2_pair(q_ptr: *const i16, v_ptr: *const i16) -> i64 {
    use std::arch::x86_64::*;
    let q = _mm256_loadu_si256(q_ptr as *const __m256i);
    let v = _mm256_loadu_si256(v_ptr as *const __m256i);
    let diff = _mm256_sub_epi16(q, v);
    let mads = _mm256_madd_epi16(diff, diff);
    let lo = _mm256_castsi256_si128(mads);
    let hi = _mm256_extracti128_si256(mads, 1);
    let sum128 = _mm_add_epi32(lo, hi);
    let s1 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b1110));
    let s2 = _mm_add_epi32(s1, _mm_shuffle_epi32(s1, 0b0001));
    _mm_cvtsi128_si32(s2) as i64
}

/// AVX2 lower-bound squared distance from `q` to bbox `[min, max]`:
///     lb² = Σ_d (q[d] - clamp(q[d], min[d], max[d]))²
/// Padded zero lanes (14, 15) contribute 0 because min=max=0 there.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn bbox_lb_avx2(q_ptr: *const i16, min_ptr: *const i16, max_ptr: *const i16) -> i64 {
    use std::arch::x86_64::*;
    let q = _mm256_loadu_si256(q_ptr as *const __m256i);
    let mn = _mm256_loadu_si256(min_ptr as *const __m256i);
    let mx = _mm256_loadu_si256(max_ptr as *const __m256i);
    let clipped = _mm256_max_epi16(_mm256_min_epi16(q, mx), mn);
    let diff = _mm256_sub_epi16(q, clipped);
    let mads = _mm256_madd_epi16(diff, diff);
    let lo = _mm256_castsi256_si128(mads);
    let hi = _mm256_extracti128_si256(mads, 1);
    let sum128 = _mm_add_epi32(lo, hi);
    let s1 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b1110));
    let s2 = _mm_add_epi32(s1, _mm_shuffle_epi32(s1, 0b0001));
    _mm_cvtsi128_si32(s2) as i64
}

/// Scalar fallback for bbox lower bound (non-AVX2 hosts).
#[inline(always)]
fn bbox_lb_scalar(q: &[i16], mn: &[i16], mx: &[i16]) -> i64 {
    let mut acc = 0i64;
    for d in 0..PADDED_DIM {
        let qd = q[d] as i32;
        let lo = mn[d] as i32;
        let hi = mx[d] as i32;
        let clipped = qd.clamp(lo, hi);
        let diff = (qd - clipped) as i64;
        acc += diff * diff;
    }
    acc
}

/// Core hot loop. Walks the probed cell ranges and updates top-5.
fn scan_clusters_scalar(
    query: &[i16],          // length 16 (14 + 2 padded zero)
    vectors: &[i16],        // (N*16,) flat row-major
    cluster_labels: &[u8],  // (N,)
    cluster_offsets: &[i64],
    cells: &[i64],
    top5: &mut Top5,
) {
    for &cell in cells {
        let s = cluster_offsets[cell as usize] as usize;
        let e = cluster_offsets[cell as usize + 1] as usize;
        for i in s..e {
            let v = &vectors[i * 16..i * 16 + 16];
            let d = dist_scalar(query, v);
            if d < top5.worst() {
                top5.push(d, cluster_labels[i]);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_clusters_avx2(
    query: &[i16],
    vectors: &[i16],
    cluster_labels: &[u8],
    cluster_offsets: &[i64],
    cells: &[i64],
    top5: &mut Top5,
) {
    let q_ptr = query.as_ptr();
    for &cell in cells {
        let s = cluster_offsets[cell as usize] as usize;
        let e = cluster_offsets[cell as usize + 1] as usize;
        let base = vectors.as_ptr();
        for i in s..e {
            let v_ptr = base.add(i * 16);
            let d = dist_avx2_pair(q_ptr, v_ptr);
            if d < top5.worst() {
                top5.push(d, cluster_labels[i]);
            }
        }
    }
}

/// Iterative KD-tree branch-and-bound DFS, scalar inner kernel.
#[allow(clippy::too_many_arguments)]
fn kdtree_scan_scalar(
    query: &[i16],
    vectors: &[i16],
    labels: &[u8],
    nodes_min: &[i16],
    nodes_max: &[i16],
    nodes_left: &[i32],
    nodes_right: &[i32],
    nodes_start: &[u32],
    nodes_len: &[u32],
    top5: &mut Top5,
) {
    const STACK_CAP: usize = 128;
    let mut stack_idx = [0i32; STACK_CAP];
    let mut stack_lb = [0i64; STACK_CAP];
    let mut sp = 0usize;
    let mut cur: i32 = 0;
    let mut cur_lb: i64 = 0;
    loop {
        if cur_lb >= top5.worst() {
            if sp == 0 {
                return;
            }
            sp -= 1;
            cur = stack_idx[sp];
            cur_lb = stack_lb[sp];
            continue;
        }
        let ci = cur as usize;
        let len = nodes_len[ci];
        if len > 0 {
            let s = nodes_start[ci] as usize;
            let e = s + len as usize;
            for i in s..e {
                let d = dist_scalar(query, &vectors[i * PADDED_DIM..i * PADDED_DIM + PADDED_DIM]);
                if d < top5.worst() {
                    top5.push(d, labels[i]);
                }
            }
            if sp == 0 {
                return;
            }
            sp -= 1;
            cur = stack_idx[sp];
            cur_lb = stack_lb[sp];
            continue;
        }
        let left = nodes_left[ci];
        let right = nodes_right[ci];
        let lb_l = bbox_lb_scalar(
            query,
            &nodes_min[left as usize * PADDED_DIM..left as usize * PADDED_DIM + PADDED_DIM],
            &nodes_max[left as usize * PADDED_DIM..left as usize * PADDED_DIM + PADDED_DIM],
        );
        let lb_r = bbox_lb_scalar(
            query,
            &nodes_min[right as usize * PADDED_DIM..right as usize * PADDED_DIM + PADDED_DIM],
            &nodes_max[right as usize * PADDED_DIM..right as usize * PADDED_DIM + PADDED_DIM],
        );
        let (near, near_lb, far, far_lb) = if lb_l <= lb_r {
            (left, lb_l, right, lb_r)
        } else {
            (right, lb_r, left, lb_l)
        };
        if far_lb < top5.worst() && sp < STACK_CAP {
            stack_idx[sp] = far;
            stack_lb[sp] = far_lb;
            sp += 1;
        }
        cur = near;
        cur_lb = near_lb;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn kdtree_scan_avx2(
    query: &[i16],
    vectors: &[i16],
    labels: &[u8],
    nodes_min: &[i16],
    nodes_max: &[i16],
    nodes_left: &[i32],
    nodes_right: &[i32],
    nodes_start: &[u32],
    nodes_len: &[u32],
    top5: &mut Top5,
) {
    const STACK_CAP: usize = 128;
    let mut stack_idx = [0i32; STACK_CAP];
    let mut stack_lb = [0i64; STACK_CAP];
    let mut sp = 0usize;
    let mut cur: i32 = 0;
    let mut cur_lb: i64 = 0;
    let q_ptr = query.as_ptr();
    let vec_base = vectors.as_ptr();
    let lbl_base = labels.as_ptr();
    let min_base = nodes_min.as_ptr();
    let max_base = nodes_max.as_ptr();
    loop {
        if cur_lb >= top5.worst() {
            if sp == 0 {
                return;
            }
            sp -= 1;
            cur = stack_idx[sp];
            cur_lb = stack_lb[sp];
            continue;
        }
        let ci = cur as usize;
        let len = nodes_len[ci];
        if len > 0 {
            let s = nodes_start[ci] as usize;
            let e = s + len as usize;
            for i in s..e {
                let d = dist_avx2_pair(q_ptr, vec_base.add(i * PADDED_DIM));
                if d < top5.worst() {
                    top5.push(d, *lbl_base.add(i));
                }
            }
            if sp == 0 {
                return;
            }
            sp -= 1;
            cur = stack_idx[sp];
            cur_lb = stack_lb[sp];
            continue;
        }
        let left = nodes_left[ci];
        let right = nodes_right[ci];
        let lb_l = bbox_lb_avx2(
            q_ptr,
            min_base.add(left as usize * PADDED_DIM),
            max_base.add(left as usize * PADDED_DIM),
        );
        let lb_r = bbox_lb_avx2(
            q_ptr,
            min_base.add(right as usize * PADDED_DIM),
            max_base.add(right as usize * PADDED_DIM),
        );
        let (near, near_lb, far, far_lb) = if lb_l <= lb_r {
            (left, lb_l, right, lb_r)
        } else {
            (right, lb_r, left, lb_l)
        };
        if far_lb < top5.worst() && sp < STACK_CAP {
            stack_idx[sp] = far;
            stack_lb[sp] = far_lb;
            sp += 1;
        }
        cur = near;
        cur_lb = near_lb;
    }
}

/// Python-facing exact KNN over a KD-tree (post-IVF).
///
/// Returns fraud-count (0..5) of top-5 nearest reference labels.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn knn_top5_kdtree(
    py: Python<'_>,
    query: PyReadonlyArray1<i16>,
    vectors: PyReadonlyArray2<i16>,
    labels: PyReadonlyArray1<u8>,
    nodes_min: PyReadonlyArray2<i16>,
    nodes_max: PyReadonlyArray2<i16>,
    nodes_left: PyReadonlyArray1<i32>,
    nodes_right: PyReadonlyArray1<i32>,
    nodes_start: PyReadonlyArray1<u32>,
    nodes_len: PyReadonlyArray1<u32>,
) -> PyResult<i32> {
    let q = query.as_slice()?;
    let v = vectors.as_slice()?;
    let lbl = labels.as_slice()?;
    let nmn = nodes_min.as_slice()?;
    let nmx = nodes_max.as_slice()?;
    let nl = nodes_left.as_slice()?;
    let nr = nodes_right.as_slice()?;
    let ns = nodes_start.as_slice()?;
    let nlen = nodes_len.as_slice()?;
    let mut top5 = Top5::new();
    py.allow_threads(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    kdtree_scan_avx2(q, v, lbl, nmn, nmx, nl, nr, ns, nlen, &mut top5);
                }
                return;
            }
        }
        kdtree_scan_scalar(q, v, lbl, nmn, nmx, nl, nr, ns, nlen, &mut top5);
    });
    Ok(top5.fraud_count())
}

/// Python-facing IVF kernel.
///
/// Arguments
/// ---------
/// query : np.ndarray[i16]  shape (16,)  — query, padded so dim-14 and 15 are 0
/// vectors : np.ndarray[i16]  shape (N, 16)  — refs, padded
/// cluster_labels : np.ndarray[u8]  shape (N,)
/// cluster_offsets : np.ndarray[i64]  shape (nlist+1,)
/// cells : np.ndarray[i64]  shape (nprobe,)  — selected cell ids
///
/// Returns
/// -------
/// fraud_count : int  — number of fraud labels in top-5 (0..5)
#[pyfunction]
fn knn_top5_fraud_count(
    py: Python<'_>,
    query: PyReadonlyArray1<i16>,
    vectors: PyReadonlyArray2<i16>,
    cluster_labels: PyReadonlyArray1<u8>,
    cluster_offsets: PyReadonlyArray1<i64>,
    cells: PyReadonlyArray1<i64>,
) -> PyResult<i32> {
    let q = query.as_slice()?;
    let v = vectors.as_slice()?;
    let labels = cluster_labels.as_slice()?;
    let offsets = cluster_offsets.as_slice()?;
    let cell_arr = cells.as_slice()?;

    let mut top5 = Top5::new();
    py.allow_threads(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    scan_clusters_avx2(q, v, labels, offsets, cell_arr, &mut top5);
                }
                return;
            }
        }
        scan_clusters_scalar(q, v, labels, offsets, cell_arr, &mut top5);
    });

    Ok(top5.fraud_count())
}

const AMOUNT_CUTS: [i16; 3] = [50, 200, 1000];
const MCC_CUTS: [i16; 7] = [1000, 2000, 3000, 4000, 5000, 6000, 8000];

/// Bucket index = count of cuts that `q` is >= to (mirrors `np.searchsorted(side='right')`).
#[inline(always)]
fn bucket(cuts: &[i16], q: i16) -> u32 {
    cuts.iter().filter(|&&c| q >= c).count() as u32
}

/// Vectorize a payload into a 16-lane i16 buffer AND compute its 8-bit
/// partition key in one pass. ISO timestamps are parsed by byte slicing —
/// no datetime overhead. Sentinel dims 5/6 are `-10000` when `last_ts` is None.
///
/// Returns `(np.ndarray[i16] shape (16,), partition_key: int)`.
#[pyfunction]
#[pyo3(signature = (
    amount, installments, requested_at, avg_amount, tx_count_24h,
    merchant_unknown, mcc, merchant_avg_amount,
    is_online, card_present, km_from_home,
    last_ts=None, last_km=0.0,
))]
#[allow(clippy::too_many_arguments)]
fn vectorize_to_i16<'py>(
    py: Python<'py>,
    amount: f32,
    installments: f32,
    requested_at: &str,
    avg_amount: f32,
    tx_count_24h: f32,
    merchant_unknown: bool,
    mcc: &str,
    merchant_avg_amount: f32,
    is_online: bool,
    card_present: bool,
    km_from_home: f32,
    last_ts: Option<&str>,
    last_km: f32,
) -> PyResult<(Bound<'py, PyArray1<i16>>, u32)> {
    let req_bytes = requested_at.as_bytes();
    let hour = parse_u32_n(&req_bytes[11..13]).min(23) as f32;
    let weekday = weekday_mon0(req_bytes) as f32;

    let avg_safe = if avg_amount > 0.0 { avg_amount } else { 1.0 };

    let (dim5, dim6) = match last_ts {
        None => (MISSING_SENTINEL_SCALED, MISSING_SENTINEL_SCALED),
        Some(ts) => {
            let req_secs = iso_to_epoch_seconds(req_bytes);
            let last_secs = iso_to_epoch_seconds(ts.as_bytes());
            let mins = (req_secs - last_secs).unsigned_abs() as f32 / 60.0;
            (
                q_round_clamp01(mins / MAX_MINUTES),
                q_round_clamp01(last_km / MAX_KM),
            )
        }
    };

    let mut out = [0i16; PADDED_DIM];
    out[0] = q_round_clamp01(amount / MAX_AMOUNT);
    out[1] = q_round_clamp01(installments / MAX_INSTALLMENTS);
    out[2] = q_round_clamp01((amount / avg_safe) / AMOUNT_VS_AVG_RATIO);
    out[3] = q_round_signed(hour / HOURS_DIVISOR);
    out[4] = q_round_signed(weekday / WEEKDAY_DIVISOR);
    out[5] = dim5;
    out[6] = dim6;
    out[7] = q_round_clamp01(km_from_home / MAX_KM);
    out[8] = q_round_clamp01(tx_count_24h / MAX_TX_COUNT_24H);
    out[9] = if is_online { QUANT_SCALE as i16 } else { 0 };
    out[10] = if card_present { QUANT_SCALE as i16 } else { 0 };
    out[11] = if merchant_unknown { QUANT_SCALE as i16 } else { 0 };
    out[12] = (mcc_risk(mcc.as_bytes()) * QUANT_SCALE).round() as i16;
    out[13] = q_round_clamp01(merchant_avg_amount / MAX_MERCHANT_AVG_AMOUNT);
    // dims 14, 15 stay 0 (AVX2 pad)

    let key = u32::from(is_online)
        | (u32::from(card_present) << 1)
        | (u32::from(merchant_unknown) << 2)
        | (bucket(&AMOUNT_CUTS, out[0]) << 3)
        | (bucket(&MCC_CUTS, out[12]) << 5);

    Ok((out.to_vec().into_pyarray(py), key))
}

#[pymodule]
fn knn_simd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(knn_top5_fraud_count, m)?)?;
    m.add_function(wrap_pyfunction!(knn_top5_kdtree, m)?)?;
    m.add_function(wrap_pyfunction!(vectorize_to_i16, m)?)?;
    Ok(())
}
