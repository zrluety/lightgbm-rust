//! upstream: include/LightGBM/utils/array_args.h (selection helpers used by
//! top-k style sampling such as GOSS).

/// Three-way partition around `arr[end-1]` in descending order.
/// Returns `(l, r)`: `arr[start..=l] > v`, `arr[l+1..r] == v`, `arr[r..end] < v`.
pub fn partition<T: PartialOrd + Copy>(arr: &mut [T], start: i32, end: i32) -> (i32, i32) {
    let mut i = start - 1;
    let mut j = end - 1;
    let mut p = i;
    let mut q = j;
    if start >= end - 1 {
        return (start - 1, end);
    }
    let v = arr[(end - 1) as usize];
    loop {
        loop {
            i += 1;
            if !(arr[i as usize] > v) {
                break;
            }
        }
        loop {
            j -= 1;
            if !(v > arr[j as usize]) {
                break;
            }
            if j == start {
                break;
            }
        }
        if i >= j {
            break;
        }
        arr.swap(i as usize, j as usize);
        if arr[i as usize] == v {
            p += 1;
            arr.swap(p as usize, i as usize);
        }
        if v == arr[j as usize] {
            q -= 1;
            arr.swap(j as usize, q as usize);
        }
    }
    arr.swap(i as usize, (end - 1) as usize);
    j = i - 1;
    i += 1;
    let mut k = start;
    while k <= p {
        arr.swap(k as usize, j as usize);
        k += 1;
        j -= 1;
    }
    let mut k = end - 2;
    while k >= q {
        arr.swap(i as usize, k as usize);
        k -= 1;
        i += 1;
    }
    (j, i)
}

/// upstream `ArgMaxAtK`: reorders `arr[start..end]` so that position `k`
/// holds the value it would have if the range were sorted descending.
/// The tail recursion is written as a loop.
pub fn arg_max_at_k<T: PartialOrd + Copy>(arr: &mut [T], mut start: i32, mut end: i32, k: i32) -> i32 {
    loop {
        if start >= end - 1 {
            return start;
        }
        let (l, r) = partition(arr, start, end);
        if (k > l && k < r) || (l == start - 1 && r == end - 1) {
            return k;
        } else if k <= l {
            end = l + 1;
        } else {
            start = r;
        }
    }
}

/// Index of the first maximum (upstream `ArgMax`, sequential branch).
pub fn arg_max<T: PartialOrd>(arr: &[T]) -> usize {
    let mut m = 0;
    for i in 1..arr.len() {
        if arr[i] > arr[m] {
            m = i;
        }
    }
    m
}

#[cfg(test)]
mod tests {
    //! Ported from upstream tests/cpp_tests/test_array_args.cpp.
    use super::*;

    // upstream: tests/cpp_tests/test_array_args.cpp TEST(Partition, JustWorks)
    #[test]
    fn partition_just_works() {
        let mut g = vec![0.5f32, 5.0, 1.0, 2.0, 2.0];
        let n = g.len() as i32;
        let (mb, me) = partition(&mut g, 0, n);
        assert_eq!(g[(mb + 1) as usize], g[(me - 1) as usize]);
        assert!(g[0] > g[(mb + 1) as usize]);
        assert!(g[(mb + 1) as usize] > *g.last().unwrap());
    }

    // upstream: tests/cpp_tests/test_array_args.cpp TEST(Partition, PartitionOneElement)
    #[test]
    fn partition_one_element() {
        let mut g = vec![0.5f32];
        let (mb, me) = partition(&mut g, 0, 1);
        assert_eq!(g[(mb + 1) as usize], g[(me - 1) as usize]);
    }

    // upstream: tests/cpp_tests/test_array_args.cpp TEST(Partition, Empty)
    #[test]
    fn partition_empty() {
        let mut g: Vec<f32> = vec![];
        let (mb, me) = partition(&mut g, 0, 0);
        assert_eq!(mb, -1);
        assert_eq!(me, 0);
    }

    // upstream: tests/cpp_tests/test_array_args.cpp TEST(Partition, AllEqual)
    #[test]
    fn partition_all_equal() {
        let mut g = vec![0.5f32, 0.5, 0.5];
        let (mb, me) = partition(&mut g, 0, 3);
        assert_eq!(g[(mb + 1) as usize], g[(me - 1) as usize]);
        assert_eq!(mb, -1);
        assert_eq!(me, 3);
    }

    #[test]
    fn arg_max_at_k_selects_kth_largest() {
        let base = [3.0f32, 9.0, 1.0, 9.0, 4.0, 7.0, 7.0, 0.5, 2.0, 7.0];
        let mut sorted = base.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        for k in 0..base.len() as i32 {
            let mut v = base.to_vec();
            let len = v.len() as i32;
            arg_max_at_k(&mut v, 0, len, k);
            assert_eq!(v[k as usize], sorted[k as usize], "k={k}");
        }
        let mut asc: Vec<f32> = (0..5000).map(|i| i as f32).collect();
        arg_max_at_k(&mut asc, 0, 5000, 10);
        assert_eq!(asc[10], 4989.0);
    }

    #[test]
    fn arg_max_first_wins() {
        assert_eq!(arg_max(&[1.0, 3.0, 3.0, 2.0]), 1);
    }
}
