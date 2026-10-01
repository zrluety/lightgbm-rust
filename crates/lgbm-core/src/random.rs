//! Port of upstream's linear congruential generator.
//!
//! upstream: include/LightGBM/utils/random.h (`class Random`). Bit-exact
//! reproduction matters because bin-construction sampling and seed derivation
//! depend on it.

use std::collections::BTreeSet;

#[derive(Debug, Clone)]
pub struct Random {
    x: u32,
}

impl Random {
    pub fn new(seed: i32) -> Self {
        Self { x: seed as u32 }
    }

    fn rand_int16(&mut self) -> i32 {
        self.x = self.x.wrapping_mul(214013).wrapping_add(2531011);
        ((self.x >> 16) & 0x7FFF) as i32
    }

    fn rand_int32(&mut self) -> i32 {
        self.x = self.x.wrapping_mul(214013).wrapping_add(2531011);
        (self.x & 0x7FFF_FFFF) as i32
    }

    /// Integer in `[lower, upper)` from the 16-bit stream.
    pub fn next_short(&mut self, lower: i32, upper: i32) -> i32 {
        self.rand_int16() % (upper - lower) + lower
    }

    /// Integer in `[lower, upper)` from the 31-bit stream.
    pub fn next_int(&mut self, lower: i32, upper: i32) -> i32 {
        self.rand_int32() % (upper - lower) + lower
    }

    /// Float in `[0, 1)`.
    pub fn next_float(&mut self) -> f32 {
        self.rand_int16() as f32 / 32768.0f32
    }

    /// Sample `k` sorted indices from `0..n`.
    pub fn sample(&mut self, n: i32, k: i32) -> Vec<i32> {
        let mut ret = Vec::with_capacity(k.max(0) as usize);
        if k > n || k <= 0 {
            return ret;
        }
        if k == n {
            ret.extend(0..n);
        } else if k > 1 && (k as f64) > (n as f64 / (k as f64).log2()) {
            for i in 0..n {
                let prob = (k as usize - ret.len()) as f64 / (n - i) as f64;
                if (self.next_float() as f64) < prob {
                    ret.push(i);
                }
            }
        } else {
            let mut set = BTreeSet::new();
            for r in (n - k)..n {
                let v = self.next_int(0, r + 1);
                if !set.insert(v) {
                    set.insert(r);
                }
            }
            ret.extend(set);
        }
        ret
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_all_when_k_equals_n() {
        let mut r = Random::new(1);
        assert_eq!(r.sample(5, 5), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn sample_is_sorted_unique_and_sized() {
        for (n, k) in [(1000, 10), (1000, 900), (50_000, 20_000)] {
            let mut r = Random::new(7);
            let s = r.sample(n, k);
            assert!(s.windows(2).all(|w| w[0] < w[1]));
            if k <= 10 {
                assert_eq!(s.len(), k as usize);
            }
            assert!(s.iter().all(|&v| v >= 0 && v < n));
        }
    }

    #[test]
    fn lcg_matches_reference_sequence() {
        // x_{n+1} = 214013 x_n + 2531011 (mod 2^32), starting at the seed.
        let mut r = Random::new(1);
        let mut x: u64 = 1;
        for _ in 0..5 {
            x = (x * 214013 + 2531011) % (1u64 << 32);
            assert_eq!(r.rand_int16() as u64, (x >> 16) & 0x7FFF);
        }
    }
}
