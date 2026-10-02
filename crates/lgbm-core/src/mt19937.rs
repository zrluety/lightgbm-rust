//! `std::mt19937` and the libstdc++ distributions upstream's gradient
//! discretizer draws from.
//!
//! upstream: src/treelearner/gradient_discretizer.cpp (`std::mt19937`,
//! `std::uniform_real_distribution<double>`, `std::uniform_int_distribution<int>`).
//! The distributions follow libstdc++, which builds the Linux wheels:
//! `generate_canonical<double, 53>` takes two 32-bit draws, and
//! `uniform_int_distribution` downscales with Lemire's nearly divisionless
//! method (`_S_nd`).

const N: usize = 624;
const M: usize = 397;

#[derive(Debug, Clone)]
pub struct Mt19937 {
    state: [u32; N],
    idx: usize,
}

impl Mt19937 {
    /// `std::mt19937(seed)`: the seed is taken modulo 2^32.
    pub fn new(seed: u32) -> Self {
        let mut state = [0u32; N];
        state[0] = seed;
        for i in 1..N {
            let prev = state[i - 1];
            state[i] = 1_812_433_253u32.wrapping_mul(prev ^ (prev >> 30)).wrapping_add(i as u32);
        }
        Self { state, idx: N }
    }

    fn twist(&mut self) {
        const UPPER: u32 = 0x8000_0000;
        const LOWER: u32 = 0x7fff_ffff;
        const MATRIX_A: u32 = 0x9908_b0df;
        for k in 0..N {
            let y = (self.state[k] & UPPER) | (self.state[(k + 1) % N] & LOWER);
            let mag = if y & 1 != 0 { MATRIX_A } else { 0 };
            self.state[k] = self.state[(k + M) % N] ^ (y >> 1) ^ mag;
        }
        self.idx = 0;
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.idx >= N {
            self.twist();
        }
        let mut z = self.state[self.idx];
        self.idx += 1;
        z ^= z >> 11;
        z ^= (z << 7) & 0x9d2c_5680;
        z ^= (z << 15) & 0xefc6_0000;
        z ^= z >> 18;
        z
    }

    /// `std::generate_canonical<double, 53>` (libstdc++): two draws,
    /// `(g1 + g2 * 2^32) / 2^64`, kept below 1.
    pub fn canonical(&mut self) -> f64 {
        const R: f64 = 4_294_967_296.0;
        let mut sum = self.next_u32() as f64;
        sum += self.next_u32() as f64 * R;
        let ret = sum / (R * R);
        // nextafter(1.0, 0.0)
        if ret >= 1.0 { f64::from_bits(0x3FEF_FFFF_FFFF_FFFF) } else { ret }
    }

    /// `std::uniform_real_distribution<double>(0, 1)`.
    pub fn uniform01(&mut self) -> f64 {
        self.canonical() * (1.0 - 0.0) + 0.0
    }

    /// `std::uniform_int_distribution<int>(a, b)` with `b - a < 2^32 - 1`
    /// (libstdc++ `_S_nd<uint64_t>`).
    pub fn uniform_int(&mut self, a: i32, b: i32) -> i32 {
        let range = (b as i64 - a as i64 + 1) as u32;
        let mut product = self.next_u32() as u64 * range as u64;
        let mut low = product as u32;
        if low < range {
            let threshold = range.wrapping_neg() % range;
            while low < threshold {
                product = self.next_u32() as u64 * range as u64;
                low = product as u32;
            }
        }
        ((product >> 32) as i64 + a as i64) as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_standard_sequence() {
        // the C++ standard requires the 10000th draw of a default-seeded mt19937 to be 4123659995
        let mut g = Mt19937::new(5489);
        let mut v = 0;
        for _ in 0..10000 {
            v = g.next_u32();
        }
        assert_eq!(v, 4_123_659_995);
    }

    #[test]
    fn distributions_stay_in_range() {
        let mut g = Mt19937::new(1);
        for _ in 0..1000 {
            let x = g.uniform01();
            assert!((0.0..1.0).contains(&x));
            let k = g.uniform_int(0, 7);
            assert!((0..=7).contains(&k));
        }
    }
}
