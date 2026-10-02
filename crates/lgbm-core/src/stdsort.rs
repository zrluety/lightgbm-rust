//! libstdc++'s `std::sort` (introsort), element order included.
//!
//! Upstream sorts with `std::sort` (via `Common::ParallelSort`, which calls it
//! directly below 1024 elements or with one thread). It is not stable, and
//! some callers accumulate floating-point sums in the sorted order of tied
//! elements, so reproducing their results bit for bit needs the same order.
//!
//! upstream: libstdc++ bits/stl_algo.h `__sort`, `__introsort_loop`,
//! `__final_insertion_sort`; bits/stl_heap.h `__make_heap`, `__adjust_heap`.

const S_THRESHOLD: usize = 16;

/// Sort `v` with the strict-weak-order predicate `less`, as `std::sort` does.
pub fn sort_by<T: Copy>(v: &mut [T], less: impl Fn(&T, &T) -> bool) {
    let n = v.len();
    if n > 1 {
        let depth_limit = 2 * (usize::BITS - 1 - n.leading_zeros()) as usize;
        introsort_loop(v, 0, n, depth_limit, &less);
        final_insertion_sort(v, &less);
    }
}

/// upstream: utils/common.h `Common::ParallelSort`: `std::sort` per chunk of
/// at least 1024 elements, then pairwise `std::merge` passes.
pub fn parallel_sort_by<T: Copy>(v: &mut [T], less: impl Fn(&T, &T) -> bool, num_threads: usize) {
    const MIN_INNER_LEN: usize = 1024;
    let len = v.len();
    if len <= MIN_INNER_LEN || num_threads <= 1 {
        sort_by(v, less);
        return;
    }
    let inner = len.div_ceil(num_threads).max(MIN_INNER_LEN);
    for chunk in v.chunks_mut(inner) {
        sort_by(chunk, &less);
    }
    let mut buf = v.to_vec();
    let mut s = inner;
    while s < len {
        let mut left = 0;
        while left < len {
            let mid = left + s;
            let right = (mid + s).min(len);
            if mid < right {
                buf[left..mid].copy_from_slice(&v[left..mid]);
                let (mut i, mut j, mut o) = (left, mid, left);
                while i < mid && j < right {
                    if less(&v[j], &buf[i]) {
                        v[o] = v[j];
                        j += 1;
                    } else {
                        v[o] = buf[i];
                        i += 1;
                    }
                    o += 1;
                }
                while i < mid {
                    v[o] = buf[i];
                    i += 1;
                    o += 1;
                }
            }
            left += 2 * s;
        }
        s *= 2;
    }
}

fn introsort_loop<T: Copy>(v: &mut [T], first: usize, mut last: usize, mut depth: usize, less: &impl Fn(&T, &T) -> bool) {
    while last - first > S_THRESHOLD {
        if depth == 0 {
            heap_sort(&mut v[first..last], less);
            return;
        }
        depth -= 1;
        let cut = unguarded_partition_pivot(v, first, last, less);
        introsort_loop(v, cut, last, depth, less);
        last = cut;
    }
}

fn unguarded_partition_pivot<T: Copy>(v: &mut [T], first: usize, last: usize, less: &impl Fn(&T, &T) -> bool) -> usize {
    let mid = first + (last - first) / 2;
    move_median_to_first(v, first, first + 1, mid, last - 1, less);
    unguarded_partition(v, first + 1, last, first, less)
}

fn move_median_to_first<T: Copy>(
    v: &mut [T],
    result: usize,
    a: usize,
    b: usize,
    c: usize,
    less: &impl Fn(&T, &T) -> bool,
) {
    let pick = if less(&v[a], &v[b]) {
        if less(&v[b], &v[c]) {
            b
        } else if less(&v[a], &v[c]) {
            c
        } else {
            a
        }
    } else if less(&v[a], &v[c]) {
        a
    } else if less(&v[b], &v[c]) {
        c
    } else {
        b
    };
    v.swap(result, pick);
}

fn unguarded_partition<T: Copy>(
    v: &mut [T],
    mut first: usize,
    mut last: usize,
    pivot: usize,
    less: &impl Fn(&T, &T) -> bool,
) -> usize {
    loop {
        // the bounds checks only matter for predicates that are not strict weak orders
        while first < v.len() - 1 && less(&v[first], &v[pivot]) {
            first += 1;
        }
        last -= 1;
        while last > pivot && less(&v[pivot], &v[last]) {
            last -= 1;
        }
        if first >= last {
            return first;
        }
        v.swap(first, last);
        first += 1;
    }
}

fn final_insertion_sort<T: Copy>(v: &mut [T], less: &impl Fn(&T, &T) -> bool) {
    if v.len() > S_THRESHOLD {
        insertion_sort(&mut v[..S_THRESHOLD], less);
        for i in S_THRESHOLD..v.len() {
            unguarded_linear_insert(v, i, less);
        }
    } else {
        insertion_sort(v, less);
    }
}

fn insertion_sort<T: Copy>(v: &mut [T], less: &impl Fn(&T, &T) -> bool) {
    for i in 1..v.len() {
        if less(&v[i], &v[0]) {
            let val = v[i];
            v.copy_within(0..i, 1);
            v[0] = val;
        } else {
            unguarded_linear_insert(v, i, less);
        }
    }
}

fn unguarded_linear_insert<T: Copy>(v: &mut [T], mut last: usize, less: &impl Fn(&T, &T) -> bool) {
    let val = v[last];
    while last > 0 && less(&val, &v[last - 1]) {
        v[last] = v[last - 1];
        last -= 1;
    }
    v[last] = val;
}

/// `std::__partial_sort(first, last, last)`: `make_heap` then `sort_heap`.
fn heap_sort<T: Copy>(v: &mut [T], less: &impl Fn(&T, &T) -> bool) {
    let len = v.len();
    if len >= 2 {
        let mut parent = (len - 2) / 2;
        loop {
            let value = v[parent];
            adjust_heap(v, parent, len, value, less);
            if parent == 0 {
                break;
            }
            parent -= 1;
        }
    }
    let mut last = len;
    while last > 1 {
        last -= 1;
        let value = v[last];
        v[last] = v[0];
        adjust_heap(v, 0, last, value, less);
    }
}

fn adjust_heap<T: Copy>(v: &mut [T], mut hole: usize, len: usize, value: T, less: &impl Fn(&T, &T) -> bool) {
    let top = hole;
    let mut second = hole;
    while len >= 1 && second < (len - 1) / 2 {
        second = 2 * (second + 1);
        if less(&v[second], &v[second - 1]) {
            second -= 1;
        }
        v[hole] = v[second];
        hole = second;
    }
    if len & 1 == 0 && len >= 2 && second == (len - 2) / 2 {
        second = 2 * (second + 1);
        v[hole] = v[second - 1];
        hole = second - 1;
    }
    // __push_heap
    let mut parent = hole.wrapping_sub(1) / 2;
    while hole > top && less(&v[parent], &value) {
        v[hole] = v[parent];
        hole = parent;
        parent = hole.wrapping_sub(1) / 2;
    }
    v[hole] = value;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::Random;

    #[test]
    fn sorts_like_a_sort() {
        let mut rng = Random::new(7);
        for n in [0usize, 1, 2, 5, 16, 17, 100, 1000, 5000] {
            let mut v: Vec<i32> = (0..n).map(|_| rng.next_int(0, 50)).collect();
            let mut want = v.clone();
            want.sort();
            sort_by(&mut v, |a, b| a < b);
            assert_eq!(v, want);
        }
    }

    #[test]
    fn heap_fallback_sorts() {
        let mut v: Vec<i32> = (0..300).map(|i| (i * 7919) % 301).collect();
        let mut want = v.clone();
        want.sort();
        heap_sort(&mut v, &|a: &i32, b: &i32| a < b);
        assert_eq!(v, want);
    }

    #[test]
    fn unstable_like_libstdcxx() {
        // 17 equal keys: the median-of-three swap moves the middle element to the front
        let mut v: Vec<(i32, usize)> = (0..17).map(|i| (0, i)).collect();
        sort_by(&mut v, |a, b| a.0 < b.0);
        assert_ne!(v.iter().map(|x| x.1).collect::<Vec<_>>(), (0..17).collect::<Vec<_>>());
    }
}
