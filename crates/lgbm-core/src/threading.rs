//! A fixed team of threads for short fork/join regions.
//!
//! The tree learner runs several short parallel regions per leaf. Waking
//! parked work-stealing threads for each region costs more than the region's
//! work on small leaves, so the learner uses a team whose workers spin
//! briefly between regions (like libgomp's `GOMP_SPINCOUNT`) before parking.
//! Tasks are claimed dynamically from a shared counter, so a descheduled
//! worker delays only the task it holds. Every task writes its own output,
//! so which thread runs which task never affects results.

use std::cell::{Cell, UnsafeCell};
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// How long an idle worker (or a waiting caller) spins before parking (or yielding).
const SPIN: Duration = Duration::from_micros(200);

/// Spin until `done()` or until [`SPIN`] has elapsed; returns `done()`.
#[inline]
fn spin_until(mut done: impl FnMut() -> bool) -> bool {
    let t0 = Instant::now();
    loop {
        for _ in 0..64 {
            if done() {
                return true;
            }
            std::hint::spin_loop();
        }
        if t0.elapsed() >= SPIN {
            return done();
        }
    }
}

fn wait_until(mut done: impl FnMut() -> bool) {
    while !spin_until(&mut done) {
        std::thread::yield_now();
    }
}

/// Regions with less estimated work than this run on the calling thread.
pub const MIN_PAR_WORK: usize = 1 << 14;

type Job = &'static (dyn Fn(usize) + Sync);

struct Shared {
    /// Incremented when a job is published.
    epoch: AtomicUsize,
    /// The last epoch whose job no worker may start any more.
    closed: AtomicUsize,
    /// Workers currently inside a job.
    active: AtomicUsize,
    job: UnsafeCell<Option<Job>>,
    ntasks: AtomicUsize,
    next: AtomicUsize,
    done: AtomicUsize,
    sleepers: AtomicUsize,
    lock: Mutex<()>,
    cv: Condvar,
    stop: AtomicBool,
    panic: Mutex<Option<Box<dyn std::any::Any + Send>>>,
}

// SAFETY: `job` is written only by `ThreadTeam::for_each`, before its epoch
// is published or after the epoch is closed and no worker is active.
unsafe impl Sync for Shared {}

impl Shared {
    /// Claim and run tasks of the current job until none remain.
    fn work(&self, f: Job) {
        let ntasks = self.ntasks.load(Ordering::Relaxed);
        loop {
            let i = self.next.fetch_add(1, Ordering::Relaxed);
            if i >= ntasks {
                return;
            }
            if let Err(p) = catch_unwind(AssertUnwindSafe(|| f(i))) {
                self.panic.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert(p);
            }
            self.done.fetch_add(1, Ordering::Release);
        }
    }
}

thread_local! {
    static IN_TEAM: Cell<bool> = const { Cell::new(false) };
}

pub struct ThreadTeam {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    /// Logical thread count (what `num_threads` asked for).
    logical: usize,
    /// OS threads actually used (at most one per logical CPU).
    n: usize,
    run_lock: Mutex<()>,
}

impl std::fmt::Debug for ThreadTeam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThreadTeam").field("logical", &self.logical).field("n", &self.n).finish()
    }
}

/// Resolve a `num_threads` parameter (<= 0 means all logical CPUs).
pub fn resolve_num_threads(num_threads: i32) -> usize {
    if num_threads > 0 {
        num_threads as usize
    } else {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    }
}

impl ThreadTeam {
    /// A team for `n` logical threads (the caller is one of them). At most
    /// one OS thread per logical CPU is started; callers that partition work
    /// by [`num_threads`](Self::num_threads) get the same partition (and
    /// results) either way.
    pub fn new(n: usize) -> Self {
        let logical = n.max(1);
        let n = logical.min(resolve_num_threads(0));
        let shared = Arc::new(Shared {
            epoch: AtomicUsize::new(0),
            closed: AtomicUsize::new(0),
            active: AtomicUsize::new(0),
            job: UnsafeCell::new(None),
            ntasks: AtomicUsize::new(0),
            next: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            sleepers: AtomicUsize::new(0),
            lock: Mutex::new(()),
            cv: Condvar::new(),
            stop: AtomicBool::new(false),
            panic: Mutex::new(None),
        });
        let workers = (1..n)
            .map(|tid| {
                let s = shared.clone();
                std::thread::Builder::new()
                    .name(format!("lgbm-team-{tid}"))
                    .spawn(move || worker(&s))
                    .expect("spawn team thread")
            })
            .collect();
        Self { shared, workers, logical, n, run_lock: Mutex::new(()) }
    }

    /// Logical thread count, used to partition work.
    pub fn num_threads(&self) -> usize {
        self.logical
    }

    /// Run `f(i)` for every `i` in `0..ntasks` and wait for all calls.
    /// `work` estimates the total cost in elementary operations; regions
    /// below [`MIN_PAR_WORK`], and nested calls, run on the calling thread.
    pub fn for_each(&self, ntasks: usize, work: usize, f: impl Fn(usize) + Sync) {
        if ntasks <= 1 || work < MIN_PAR_WORK || self.n == 1 || IN_TEAM.with(|c| c.get()) {
            (0..ntasks).for_each(f);
            return;
        }
        let _guard = self.run_lock.lock().unwrap_or_else(|e| e.into_inner());
        let s = &*self.shared;
        let f: &(dyn Fn(usize) + Sync) = &f;
        // SAFETY: the lifetime is erased only for the duration of this call;
        // the epoch is closed and no worker is active before we return.
        let job: Job = unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), Job>(f) };
        unsafe { *s.job.get() = Some(job) };
        s.ntasks.store(ntasks, Ordering::Relaxed);
        s.next.store(0, Ordering::Relaxed);
        s.done.store(0, Ordering::Relaxed);
        let epoch = s.epoch.fetch_add(1, Ordering::SeqCst) + 1;
        if s.sleepers.load(Ordering::SeqCst) > 0 {
            let _l = s.lock.lock().unwrap_or_else(|e| e.into_inner());
            s.cv.notify_all();
        }
        IN_TEAM.with(|c| c.set(true));
        s.work(job);
        IN_TEAM.with(|c| c.set(false));
        wait_until(|| s.done.load(Ordering::Acquire) == ntasks);
        s.closed.store(epoch, Ordering::SeqCst);
        wait_until(|| s.active.load(Ordering::SeqCst) == 0);
        unsafe { *s.job.get() = None };
        if let Some(p) = s.panic.lock().unwrap_or_else(|e| e.into_inner()).take() {
            resume_unwind(p);
        }
    }

    /// Like [`for_each`](Self::for_each) but collects one result per task.
    pub fn map<R: Send>(&self, ntasks: usize, work: usize, f: impl Fn(usize) -> R + Sync) -> Vec<R> {
        let mut out: Vec<Option<R>> = (0..ntasks).map(|_| None).collect();
        let slots = SharedMut::new(&mut out);
        // SAFETY: each task index is visited exactly once.
        self.for_each(ntasks, work, |i| unsafe { *slots.get(i) = Some(f(i)) });
        out.into_iter().map(|r| r.expect("task result")).collect()
    }
}

fn worker(s: &Shared) {
    IN_TEAM.with(|c| c.set(true));
    let mut seen = 0usize;
    loop {
        let changed = || s.epoch.load(Ordering::SeqCst) != seen || s.stop.load(Ordering::Acquire);
        if !spin_until(changed) {
            let mut g = s.lock.lock().unwrap_or_else(|e| e.into_inner());
            s.sleepers.fetch_add(1, Ordering::SeqCst);
            while s.epoch.load(Ordering::SeqCst) == seen && !s.stop.load(Ordering::SeqCst) {
                g = s.cv.wait(g).unwrap_or_else(|e| e.into_inner());
            }
            s.sleepers.fetch_sub(1, Ordering::SeqCst);
        }
        if s.stop.load(Ordering::Acquire) {
            return;
        }
        seen = s.epoch.load(Ordering::SeqCst);
        s.active.fetch_add(1, Ordering::SeqCst);
        // A worker that registers after the caller closed `seen` must not
        // touch its job: the caller may already have returned.
        if s.closed.load(Ordering::SeqCst) != seen {
            // SAFETY: published before epoch `seen`, which is not closed, and
            // the caller waits for `active == 0` before clearing it.
            if let Some(f) = unsafe { *s.job.get() } {
                s.work(f);
            }
        }
        s.active.fetch_sub(1, Ordering::SeqCst);
    }
}

impl Drop for ThreadTeam {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        {
            let _l = self.shared.lock.lock().unwrap_or_else(|e| e.into_inner());
            self.shared.cv.notify_all();
        }
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

/// Shared mutable access to a slice for tasks that touch disjoint elements.
pub struct SharedMut<T> {
    ptr: *mut T,
    len: usize,
}

// SAFETY: callers guarantee disjoint access (see `get` / `slice`).
unsafe impl<T: Send> Send for SharedMut<T> {}
unsafe impl<T: Send> Sync for SharedMut<T> {}

impl<T> SharedMut<T> {
    pub fn new(s: &mut [T]) -> Self {
        Self { ptr: s.as_mut_ptr(), len: s.len() }
    }

    /// # Safety
    /// No two live references may cover the same element.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn get(&self, i: usize) -> &mut T {
        assert!(i < self.len);
        unsafe { &mut *self.ptr.add(i) }
    }

    /// # Safety
    /// No two live references may overlap.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn slice(&self, start: usize, len: usize) -> &mut [T] {
        assert!(start + len <= self.len);
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(start), len) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_every_task_once() {
        let team = ThreadTeam::new(4);
        for _ in 0..200 {
            for ntasks in [0usize, 1, 3, 4, 17, 100] {
                for work in [0, usize::MAX] {
                    let v = team.map(ntasks, work, |i| i * 2);
                    assert_eq!(v, (0..ntasks).map(|i| i * 2).collect::<Vec<_>>());
                }
            }
        }
        let hits = AtomicUsize::new(0);
        team.for_each(5, usize::MAX, |_| {
            team.for_each(7, usize::MAX, |_| {
                hits.fetch_add(1, Ordering::Relaxed);
            });
        });
        assert_eq!(hits.load(Ordering::Relaxed), 35);
    }

    #[test]
    fn propagates_panics() {
        let team = ThreadTeam::new(3);
        let r = catch_unwind(AssertUnwindSafe(|| team.for_each(3, usize::MAX, |i| assert!(i != 2, "boom"))));
        assert!(r.is_err());
        assert_eq!(team.map(3, usize::MAX, |i| i), vec![0, 1, 2]);
    }
}
