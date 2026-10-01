//! C/`{fmt}`-compatible `%g` formatting used by the model text format.
//!
//! upstream: include/LightGBM/utils/common.h `__TToStringHelper` (`{:g}` and
//! `{:.17g}`) and default `std::stringstream` output (6 significant digits).

/// Format like C `printf("%.{precision}g", v)`.
pub fn fmt_g(v: f64, precision: usize) -> String {
    if v.is_nan() {
        return if v.is_sign_negative() { "-nan".into() } else { "nan".into() };
    }
    if v.is_infinite() {
        return if v < 0.0 { "-inf".into() } else { "inf".into() };
    }
    let p = precision.max(1);
    if v == 0.0 {
        return if v.is_sign_negative() { "-0".into() } else { "0".into() };
    }
    // Scientific rendering gives the correctly rounded decimal exponent.
    let sci = format!("{:.*e}", p - 1, v);
    let (mantissa, exp) = sci.split_once('e').expect("scientific format");
    let exp: i32 = exp.parse().expect("exponent");
    if exp < -4 || exp >= p as i32 {
        let m = strip_trailing_zeros(mantissa);
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{m}e{sign}{:02}", exp.abs())
    } else {
        let decimals = (p as i32 - 1 - exp).max(0) as usize;
        strip_trailing_zeros(&format!("{:.*}", decimals, v)).to_string()
    }
}

fn strip_trailing_zeros(s: &str) -> &str {
    if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.') } else { s }
}

/// `%.17g`: round-trippable double output.
pub fn fmt_g17(v: f64) -> String {
    fmt_g(v, 17)
}

/// `%g` / default stream precision (6 significant digits).
pub fn fmt_g6(v: f64) -> String {
    fmt_g(v, 6)
}

/// Parse a number written by upstream (accepts `inf`, `-inf`, `nan`, `-nan`).
pub fn parse_f64(s: &str) -> Option<f64> {
    let s = s.trim();
    match s {
        // MSVC renders NaN this way
        "nan(ind)" | "-nan(ind)" => Some(f64::NAN),
        _ => s.parse::<f64>().ok(),
    }
}

/// upstream: utils/common.h `Pow(T base, int power)` (its multiplication order
/// determines the rounding of [`atof_legacy`]).
fn pow_legacy(base: f64, power: i32) -> f64 {
    if power < 0 {
        1.0 / pow_legacy(base, -power)
    } else if power == 0 {
        1.0
    } else if power % 2 == 0 {
        pow_legacy(base * base, power / 2)
    } else if power % 3 == 0 {
        pow_legacy(base * base * base, power / 3)
    } else {
        base * pow_legacy(base, power - 1)
    }
}

/// upstream: utils/common.h `Common::Atof`, the legacy, not correctly rounded
/// parser that `StringToArrayFast` uses for some model fields (`split_gain`,
/// `internal_value`, `internal_weight`, `shrinkage`). Kept bit-for-bit so
/// loaded models hold the same doubles as upstream's.
pub fn atof_legacy(token: &str) -> Option<f64> {
    let b = token.trim_matches(' ').as_bytes();
    let mut p = 0;
    let mut sign = 1.0;
    if p < b.len() && b[p] == b'-' {
        sign = -1.0;
        p += 1;
    } else if p < b.len() && b[p] == b'+' {
        p += 1;
    }
    let is_num = p < b.len() && (b[p].is_ascii_digit() || matches!(b[p], b'.' | b'e' | b'E'));
    if !is_num {
        return match std::str::from_utf8(&b[p..]).ok()?.to_ascii_lowercase().as_str() {
            "na" | "nan" | "null" => Some(f64::NAN),
            "inf" | "infinity" => Some(sign * 1e308),
            _ => None,
        };
    }
    let mut value = 0.0f64;
    while p < b.len() && b[p].is_ascii_digit() {
        value = value * 10.0 + (b[p] - b'0') as f64;
        p += 1;
    }
    if p < b.len() && b[p] == b'.' {
        let mut right = 0.0f64;
        let mut nn = 0;
        p += 1;
        while p < b.len() && b[p].is_ascii_digit() {
            right = (b[p] - b'0') as f64 + right * 10.0;
            nn += 1;
            p += 1;
        }
        value += right / pow_legacy(10.0, nn);
    }
    let mut frac = false;
    let mut scale = 1.0f64;
    if p < b.len() && matches!(b[p], b'e' | b'E') {
        p += 1;
        if p < b.len() && b[p] == b'-' {
            frac = true;
            p += 1;
        } else if p < b.len() && b[p] == b'+' {
            p += 1;
        }
        let mut expon: u32 = 0;
        while p < b.len() && b[p].is_ascii_digit() {
            expon = expon.wrapping_mul(10).wrapping_add((b[p] - b'0') as u32);
            p += 1;
        }
        expon = expon.min(308);
        while expon >= 50 {
            scale *= 1e50;
            expon -= 50;
        }
        while expon >= 8 {
            scale *= 1e8;
            expon -= 8;
        }
        while expon > 0 {
            scale *= 10.0;
            expon -= 1;
        }
    }
    if p != b.len() {
        return None;
    }
    Some(sign * if frac { value / scale } else { value * scale })
}

#[cfg(test)]
mod atof_tests {
    //! Ported from upstream tests/cpp_tests/test_common.cpp (`AtofPreciseTest`).
    use super::parse_f64;

    // upstream: tests/cpp_tests/test_common.cpp TEST_F(AtofPreciseTest, Basic)
    #[test]
    fn basic() {
        let cases: &[(&str, f64)] = &[
            ("0", 0.0),
            ("0E0", 0.0),
            ("-0E0", 0.0),
            ("-0", -0.0),
            ("1", 1.0),
            ("1E0", 1.0),
            ("-1", -1.0),
            ("-1E0", -1.0),
            ("123456.0", 123456.0),
            ("432E1", 432E1),
            ("1.2345678", 1.2345678),
            ("2.4414062E-4", 2.4414062E-4),
            ("3.0540412E5", 3.0540412E5),
            ("3.355445E7", 3.355445E7),
            ("1.1754944E-38", 1.1754944E-38),
        ];
        for (s, e) in cases {
            assert_eq!(parse_f64(s), Some(*e), "{s}");
        }
    }

    // upstream: tests/cpp_tests/test_common.cpp TEST_F(AtofPreciseTest, CornerCases)
    #[test]
    fn corner_cases() {
        let cases: &[(&str, f64)] = &[
            ("1e-400", 0.0),
            ("2.4703282292062326e-324", 0.0),
            ("4.9406564584124654e-324", f64::from_bits(0x0000000000000001)),
            ("8.44291197326099e-309", f64::from_bits(0x0006123400000001)),
            ("3.40282346638528859811704183484516925440e38", f32::MAX as f64),
            (
                "1.1754943508222875079687365372222456778186655567720875215087517062784172594547271728515625e-38",
                f32::MIN_POSITIVE as f64,
            ),
            (
                "17976931348623157081452742373170435679807056752584499659891747680315\
                 72607800285387605895586327668781715404589535143824642343213268894641\
                 82768467546703537516986049910576551282076245490090389328944075868508\
                 45513394230458323690322294816580855933212334827479782620414472316873\
                 8177180919299881250404026184124858368",
                f64::MAX,
            ),
            ("1.7976931348623158e+308", f64::MAX),
            (
                "179769313486231580793728971405303415079934132710037826936173778980444\
                 968292764750946649017977587207096330286416692887910946555547851940402\
                 630657488671505820681908902000708383676273854845817711531764475730270\
                 069855571366959622842914819860834936475292719074168444365510704342711\
                 559699508093042880177904174497792",
                f64::INFINITY,
            ),
            ("2.2250738585072009e-308", f64::from_bits(0x000fffffffffffff)),
            ("2.2250738585072012e-308", f64::MIN_POSITIVE),
            ("2.2250738585072014e-308", f64::MIN_POSITIVE),
        ];
        for (s, e) in cases {
            assert_eq!(parse_f64(s).map(f64::to_bits), Some(e.to_bits()), "{s}");
        }
    }

    // upstream: tests/cpp_tests/test_common.cpp TEST_F(AtofPreciseTest, ErrorInput)
    // Adapted: upstream throws; the Rust API returns None.
    #[test]
    fn error_input() {
        assert_eq!(parse_f64("x1"), None);
    }

    // upstream: tests/cpp_tests/test_common.cpp TEST_F(AtofPreciseTest, NaN)
    #[test]
    fn nan() {
        for s in ["nan", "NaN", "NAN", "-nan", "-NaN", "-NAN"] {
            let v = parse_f64(s).unwrap_or_else(|| panic!("failed to parse {s}"));
            assert!(v.is_nan(), "{s}");
            if !s.starts_with('-') {
                assert_eq!(v.to_bits(), f64::NAN.to_bits(), "{s}");
            }
        }
    }

    // upstream: tests/cpp_tests/test_common.cpp TEST_F(AtofPreciseTest, Inf)
    #[test]
    fn inf() {
        for (s, e) in [
            ("inf", f64::INFINITY),
            ("Inf", f64::INFINITY),
            ("INF", f64::INFINITY),
            ("-inf", f64::NEG_INFINITY),
            ("-Inf", f64::NEG_INFINITY),
            ("-INF", f64::NEG_INFINITY),
        ] {
            assert_eq!(parse_f64(s).map(f64::to_bits), Some(e.to_bits()), "{s}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_printf_g() {
        assert_eq!(fmt_g6(0.1), "0.1");
        assert_eq!(fmt_g6(1.0), "1");
        assert_eq!(fmt_g6(100000.0), "100000");
        assert_eq!(fmt_g6(1000000.0), "1e+06");
        assert_eq!(fmt_g6(0.0001), "0.0001");
        assert_eq!(fmt_g6(0.00001), "1e-05");
        assert_eq!(fmt_g6(123.456789), "123.457");
        assert_eq!(fmt_g6(-2.5e-10), "-2.5e-10");
        assert_eq!(fmt_g17(0.1), "0.10000000000000001");
        assert_eq!(fmt_g17(1e-35_f32 as f64), "1.0000000180025095e-35");
        assert_eq!(fmt_g17(f64::INFINITY), "inf");
    }

    #[test]
    fn legacy_atof_matches_upstream_rounding() {
        // values observed from upstream 4.7.0 after loading a model text
        assert_eq!(atof_legacy("9.53084e-05"), Some(9.530839999999999e-05));
        assert_eq!(atof_legacy("-1.39281"), Some(-1.3928099999999999));
        assert_eq!(atof_legacy("0.5"), Some(0.5));
        assert_eq!(atof_legacy("-inf"), Some(-1e308));
        assert!(atof_legacy("nan").unwrap().is_nan());
        assert_eq!(atof_legacy("1x"), None);
    }

    #[test]
    fn g17_round_trips() {
        for v in [0.1, 1.0 / 3.0, -12345.678e-30, 6.02214076e23, f64::MIN_POSITIVE] {
            assert_eq!(fmt_g17(v).parse::<f64>().unwrap(), v);
        }
    }
}
