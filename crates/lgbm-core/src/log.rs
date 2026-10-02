//! Engine log messages.
//!
//! upstream: include/LightGBM/utils/log.h (`Log`) and `Config::SetVerbosity`.
//!
//! As upstream, the minimum level and the destination are per thread. A
//! message is written as `[LightGBM] [<Level>] <message>`: to the calling
//! thread's capture buffer when [`start_capture`] was called on it (the
//! Python package does this on import, like upstream's
//! `LGBM_RegisterLogCallback`), otherwise to stdout. Drained messages are
//! cut to 511 bytes, as upstream's callback buffer cuts them.

use std::cell::{Cell, RefCell};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Fatal = -1,
    Warning = 0,
    Info = 1,
    Debug = 2,
}

thread_local! {
    static LEVEL: Cell<LogLevel> = const { Cell::new(LogLevel::Info) };
    static CAPTURED: RefCell<Option<Vec<String>>> = const { RefCell::new(None) };
}

/// upstream: `Log::ResetLogLevel`.
pub fn reset_log_level(level: LogLevel) {
    LEVEL.with(|l| l.set(level));
}

pub fn log_level() -> LogLevel {
    LEVEL.with(Cell::get)
}

/// The level upstream's `Config::SetVerbosity` sets for a `verbosity` value.
pub fn level_for_verbosity(verbosity: i32) -> LogLevel {
    match verbosity {
        v if v < 0 => LogLevel::Fatal,
        0 => LogLevel::Warning,
        1 => LogLevel::Info,
        _ => LogLevel::Debug,
    }
}

/// Whether a message at `level` would be written on this thread.
pub fn enabled(level: LogLevel) -> bool {
    level <= log_level()
}

/// Send this thread's messages to a buffer read by [`drain`] instead of stdout.
pub fn start_capture() {
    CAPTURED.with(|c| {
        c.borrow_mut().get_or_insert_with(Vec::new);
    });
}

/// Take the messages captured on this thread so far, oldest first.
pub fn drain() -> Vec<String> {
    let lines = CAPTURED.with(|c| c.borrow_mut().as_mut().map(std::mem::take).unwrap_or_default());
    lines.into_iter().map(cut_message).collect()
}

/// upstream's callback buffer holds 511 bytes of the message after the prefix.
fn cut_message(mut line: String) -> String {
    let body = line.find("] ").and_then(|i| line[i + 2..].find("] ").map(|j| i + 2 + j + 2)).unwrap_or(0);
    let mut end = line.len().min(body + 511);
    while !line.is_char_boundary(end) {
        end -= 1;
    }
    line.truncate(end);
    line
}

/// Run `f` and return what it logged on this thread, without the
/// `[LightGBM] [<Level>] ` prefixes; the previous destination is restored.
pub fn capture<R>(f: impl FnOnce() -> R) -> (R, Vec<String>) {
    let saved = CAPTURED.with(|c| c.borrow_mut().replace(Vec::new()));
    let r = f();
    let got = CAPTURED.with(|c| std::mem::replace(&mut *c.borrow_mut(), saved)).unwrap_or_default();
    let strip = |m: String| match m.find("] ").and_then(|i| m[i + 2..].find("] ").map(|j| i + 2 + j + 2)) {
        Some(k) => m[k..].to_string(),
        None => m,
    };
    (r, got.into_iter().map(strip).collect())
}

/// Messages held back by [`defer`].
#[must_use]
pub struct Deferred(Vec<String>);

impl Deferred {
    /// Write the held messages to this thread's destination, in order.
    pub fn emit(self) {
        for line in self.0 {
            emit(line);
        }
    }
}

/// Run `f` and hold back what it logs on this thread, for code that computes
/// earlier than the upstream step whose messages it reproduces.
pub fn defer<R>(f: impl FnOnce() -> R) -> (R, Deferred) {
    let saved = CAPTURED.with(|c| c.borrow_mut().replace(Vec::new()));
    let r = f();
    let got = CAPTURED.with(|c| std::mem::replace(&mut *c.borrow_mut(), saved)).unwrap_or_default();
    (r, Deferred(got))
}

/// `pool.install(f)` with the calling thread's level, and with the messages
/// `f` logs on the pool thread sent to the calling thread's destination.
/// Messages logged from inside parallel iterators stay on their worker
/// threads, so sequential code logs and parallel code returns its messages.
pub fn install<R: Send>(pool: &rayon::ThreadPool, f: impl FnOnce() -> R + Send) -> R {
    let level = log_level();
    let (r, got) = pool.install(move || {
        let saved_level = log_level();
        reset_log_level(level);
        let saved = CAPTURED.with(|c| c.borrow_mut().replace(Vec::new()));
        let r = f();
        let got = CAPTURED.with(|c| std::mem::replace(&mut *c.borrow_mut(), saved)).unwrap_or_default();
        reset_log_level(saved_level);
        (r, got)
    });
    for line in got {
        emit(line);
    }
    r
}

pub fn debug(msg: &str) {
    write(LogLevel::Debug, "Debug", msg);
}

pub fn info(msg: &str) {
    write(LogLevel::Info, "Info", msg);
}

pub fn warning(msg: &str) {
    write(LogLevel::Warning, "Warning", msg);
}

/// [`warning`] for each message, in order.
pub fn warnings<S: AsRef<str>>(msgs: impl IntoIterator<Item = S>) {
    for m in msgs {
        warning(m.as_ref());
    }
}

fn write(level: LogLevel, tag: &str, msg: &str) {
    if !enabled(level) {
        return;
    }
    emit(format!("[LightGBM] [{tag}] {msg}"));
}

/// Send a formatted line to this thread's destination.
fn emit(line: String) {
    let line = CAPTURED.with(|c| match c.borrow_mut().as_mut() {
        Some(buf) => {
            buf.push(line);
            None
        }
        None => Some(line),
    });
    if let Some(line) = line {
        println!("{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_filter_and_capture_is_per_thread() {
        start_capture();
        reset_log_level(LogLevel::Warning);
        info("hidden");
        warning("shown");
        reset_log_level(level_for_verbosity(2));
        debug(&"x".repeat(600));
        let got = drain();
        assert_eq!(got[0], "[LightGBM] [Warning] shown");
        assert_eq!(got[1].len(), "[LightGBM] [Debug] ".len() + 511);
        assert!(drain().is_empty());
        let other = std::thread::spawn(|| (log_level(), drain().len())).join().unwrap();
        assert_eq!(other, (LogLevel::Info, 0));
        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        install(&pool, || debug("from the pool"));
        assert_eq!(drain(), vec!["[LightGBM] [Debug] from the pool".to_string()]);
        let ((), held) = defer(|| info("later"));
        info("first");
        held.emit();
        assert_eq!(drain(), vec!["[LightGBM] [Info] first".to_string(), "[LightGBM] [Info] later".to_string()]);
        reset_log_level(LogLevel::Info);
    }
}
