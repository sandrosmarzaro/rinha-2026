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

#[pymodule]
fn knn_simd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(knn_top5_fraud_count, m)?)?;
    Ok(())
}
