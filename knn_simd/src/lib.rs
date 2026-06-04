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

use numpy::PyReadonlyArray1;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

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
    // _mm256_madd_epi16: pairs of int16 → 8 lanes of int32, each = a0*b0 + a1*b1.
    let mads = _mm256_madd_epi16(diff, diff);
    // Horizontal sum the 8 i32 lanes into a single i64.
    // First, hadd within 128-bit halves.
    let lo = _mm256_castsi256_si128(mads);
    let hi = _mm256_extracti128_si256(mads, 1);
    let sum128 = _mm_add_epi32(lo, hi);
    // Reduce 4 i32 → 1 i32. Use shuffles.
    let s1 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b1110));
    let s2 = _mm_add_epi32(s1, _mm_shuffle_epi32(s1, 0b0001));
    _mm_cvtsi128_si32(s2) as i64
}

/// Per-cluster bbox lower bound on ||q - v||² for any v in the cluster.
/// `bbox_ptr` points to 32 i16 = [min_dim0..min_dim15, max_dim0..max_dim15].
/// Padded lanes contribute 0 since both min and max are 0 there.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn cluster_lb_sq_avx2(q_ptr: *const i16, bbox_ptr: *const i16) -> i64 {
    use std::arch::x86_64::*;
    let q = _mm256_loadu_si256(q_ptr as *const __m256i);
    let min_v = _mm256_loadu_si256(bbox_ptr as *const __m256i);
    let max_v = _mm256_loadu_si256(bbox_ptr.add(16) as *const __m256i);
    let above = _mm256_subs_epi16(q, max_v); // saturating; positive if q > max
    let below = _mm256_subs_epi16(min_v, q); // saturating; positive if q < min
    let zero = _mm256_setzero_si256();
    let pos_above = _mm256_max_epi16(above, zero);
    let pos_below = _mm256_max_epi16(below, zero);
    // At most one of (above, below) is positive per dim → sum is the gap.
    let lb_per_dim = _mm256_add_epi16(pos_above, pos_below);
    let mads = _mm256_madd_epi16(lb_per_dim, lb_per_dim);
    let lo = _mm256_castsi256_si128(mads);
    let hi = _mm256_extracti128_si256(mads, 1);
    let sum128 = _mm_add_epi32(lo, hi);
    let s1 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b1110));
    let s2 = _mm_add_epi32(s1, _mm_shuffle_epi32(s1, 0b0001));
    _mm_cvtsi128_si32(s2) as i64
}

#[inline(always)]
fn cluster_lb_sq_scalar(query: &[i16], bbox: &[i16]) -> i64 {
    // bbox = [min×16, max×16]
    let mut acc: i64 = 0;
    for d in 0..16 {
        let q = query[d] as i32;
        let lo = bbox[d] as i32;
        let hi = bbox[16 + d] as i32;
        let gap = if q > hi {
            q - hi
        } else if q < lo {
            lo - q
        } else {
            0
        };
        acc += (gap * gap) as i64;
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

#[inline(always)]
fn scan_one_cluster_scalar(
    query: &[i16],
    vectors: &[i16],
    cluster_labels: &[u8],
    s: usize,
    e: usize,
    top5: &mut Top5,
) {
    for i in s..e {
        let v = &vectors[i * 16..i * 16 + 16];
        let d = dist_scalar(query, v);
        if d < top5.worst() {
            top5.push(d, cluster_labels[i]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_one_cluster_avx2(
    q_ptr: *const i16,
    base: *const i16,
    cluster_labels: &[u8],
    s: usize,
    e: usize,
    top5: &mut Top5,
) {
    for i in s..e {
        let v_ptr = base.add(i * 16);
        let d = dist_avx2_pair(q_ptr, v_ptr);
        if d < top5.worst() {
            top5.push(d, cluster_labels[i]);
        }
    }
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

/// Cluster-bbox-pruned IVF kernel: skip whole clusters whose bounding-box
/// lower bound on ||q - v||² already exceeds the current top-5 worst.
///
/// `cluster_bbox` is (nlist, 2, 16) flattened to (nlist * 32) i16: per cluster,
/// 16 i16 mins followed by 16 i16 maxs.
#[pyfunction]
fn knn_top5_bbox_pruned(
    py: Python<'_>,
    query: PyReadonlyArray1<i16>,
    vectors: PyReadonlyArray2<i16>,
    cluster_labels: PyReadonlyArray1<u8>,
    cluster_offsets: PyReadonlyArray1<i64>,
    cluster_bbox: PyReadonlyArray1<i16>,
    cells: PyReadonlyArray1<i64>,
) -> PyResult<i32> {
    let q = query.as_slice()?;
    let v = vectors.as_slice()?;
    let labels = cluster_labels.as_slice()?;
    let offsets = cluster_offsets.as_slice()?;
    let bbox = cluster_bbox.as_slice()?;
    let cell_arr = cells.as_slice()?;

    let mut top5 = Top5::new();
    py.allow_threads(|| {
        let use_avx2 = is_x86_feature_detected!("avx2");
        let q_ptr = q.as_ptr();
        let base = v.as_ptr();
        let bbox_base = bbox.as_ptr();

        for &cell in cell_arr {
            let cluster_idx = cell as usize;
            // Each cluster's bbox occupies 32 i16 (16 min + 16 max).
            let bbox_ptr = unsafe { bbox_base.add(cluster_idx * 32) };
            let lb_sq = if use_avx2 {
                unsafe { cluster_lb_sq_avx2(q_ptr, bbox_ptr) }
            } else {
                cluster_lb_sq_scalar(q, &bbox[cluster_idx * 32..cluster_idx * 32 + 32])
            };
            if lb_sq >= top5.worst() {
                continue; // prune entire cluster
            }
            let s = offsets[cluster_idx] as usize;
            let e = offsets[cluster_idx + 1] as usize;
            if use_avx2 {
                unsafe { scan_one_cluster_avx2(q_ptr, base, labels, s, e, &mut top5) }
            } else {
                scan_one_cluster_scalar(q, v, labels, s, e, &mut top5);
            }
        }
    });

    Ok(top5.fraud_count())
}

#[pymodule]
fn knn_simd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(knn_top5_fraud_count, m)?)?;
    m.add_function(wrap_pyfunction!(knn_top5_bbox_pruned, m)?)?;
    Ok(())
}
