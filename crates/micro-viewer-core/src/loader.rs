//! Tiles read in the background: worker threads take requests from a queue in priority
//! order and send the tiles back; the drawing thread uploads them at its next frame.
//! Every frame replaces the queue with what the view needs now, so tiles that left the
//! view are never read (cancelled); a tile already being read finishes and is cached.
//! Tiles are cut out of decoded storage blocks kept between reads ([`crate::blocks`]).

use std::collections::{HashSet, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use dyna_zarr_arrays::SourceRef;

use crate::blocks::Blocks;
use crate::tiles::TileKey;

/// One tile or brick to read: the level's array and the region.
pub struct Request {
    pub key: TileKey,
    /// The image's generation (set by the loader).
    pub gen: u64,
    pub array: SourceRef,
    pub start: Vec<u64>,
    pub stop: Vec<u64>,
}

impl Request {
    fn spatial(&self) -> usize {
        if self.key.brick {
            3
        } else {
            2
        }
    }
}

/// A tile read (or why it could not be).
pub struct Loaded {
    pub key: TileKey,
    gen: u64,
    /// Rows and columns of the region (edge tiles are smaller).
    pub w: u32,
    pub h: u32,
    pub data: Result<Vec<u8>, String>,
    /// Seconds the read took (the block's decode included when this tile started it).
    pub took: f64,
    /// A brick's lowest and highest value (to skip empty bricks).
    pub range: Option<(f32, f32)>,
}

/// The host's wake-up, shared with the workers.
type Wakeup = Arc<Mutex<Option<Box<dyn Fn() + Send>>>>;

struct Queue {
    items: VecDeque<Request>,
    stop: bool,
}

struct Shared {
    queue: Mutex<Queue>,
    ready: Condvar,
    /// Tiles sent and not yet taken by the drawing thread.
    waiting: AtomicUsize,
    blocks: Blocks,
}

/// The worker threads and their queue.
pub struct Loader {
    shared: Arc<Shared>,
    results: Receiver<Loaded>,
    workers: Vec<JoinHandle<()>>,
    /// Requested and not yet back (queued or being read).
    in_flight: HashSet<TileKey>,
    /// Bumped for every new image: tiles of an older one are dropped when they arrive.
    gen: u64,
    /// Called from a worker after each tile (the host's wake-up, e.g. to repaint).
    notify: Wakeup,
}

/// Extra time every tile read takes, from `MICRO_VIEWER_SLOW_READS_MS` (to try and test
/// how the viewer behaves on slow sources: remote stores, slow codecs, other formats).
fn slow_reads() -> std::time::Duration {
    let ms = std::env::var("MICRO_VIEWER_SLOW_READS_MS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(0);
    std::time::Duration::from_millis(ms)
}

fn work(shared: Arc<Shared>, tx: Sender<Loaded>, notify: Wakeup, slow: std::time::Duration) {
    loop {
        let r = {
            let mut q = shared.queue.lock().expect("loader queue");
            loop {
                if q.stop {
                    return;
                }
                if let Some(r) = q.items.pop_front() {
                    break r;
                }
                q = shared.ready.wait(q).expect("loader queue");
            }
        };
        let n = r.start.len();
        let mut w = (r.stop[n - 1] - r.start[n - 1]) as u32;
        let mut h = (r.stop[n - 2] - r.start[n - 2]) as u32;
        let t0 = std::time::Instant::now();
        // a panic while reading (a bug, or a backend's) becomes this tile's error: the
        // thread lives on and the tile stays missing
        let read = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            shared
                .blocks
                .read(r.gen, &r.array, &r.start, &r.stop, r.spatial())
        }));
        let mut data = read.unwrap_or_else(|p| {
            let why = p
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "unknown".into());
            Err(format!(
                "reading {:?} .. {:?} failed: {why}",
                r.start, r.stop
            ))
        });
        let mut range = None;
        if r.key.brick {
            if let Ok(b) = &data {
                range = crate::source::value_range(r.array.dtype(), b);
            }
            // laid out for the brick atlas: a strip of its planes side by side
            let d = (r.stop[n - 3] - r.start[n - 3]) as usize;
            let item = r.array.dtype().size();
            data = data.map(|b| crate::bricks::strip(&b, d, h as usize, w as usize, item));
            w = crate::bricks::STRIP_W;
            h = crate::bricks::BRICK;
        }
        if !slow.is_zero() {
            std::thread::sleep(slow);
        }
        let took = t0.elapsed().as_secs_f64();
        shared.waiting.fetch_add(1, Ordering::Release);
        if tx
            .send(Loaded {
                key: r.key,
                gen: r.gen,
                w,
                h,
                data,
                took,
                range,
            })
            .is_err()
        {
            return; // the viewer is gone
        }
        if let Some(f) = notify.lock().expect("loader notify").as_ref() {
            f();
        }
    }
}

impl Loader {
    /// `threads` workers (at least one).
    pub fn new(threads: usize) -> Self {
        let shared = Arc::new(Shared {
            queue: Mutex::new(Queue {
                items: VecDeque::new(),
                stop: false,
            }),
            ready: Condvar::new(),
            waiting: AtomicUsize::new(0),
            blocks: Blocks::default(),
        });
        let (tx, results) = channel();
        let notify: Wakeup = Arc::new(Mutex::new(None));
        let slow = slow_reads();
        let workers = (0..threads.max(1))
            .map(|i| {
                let (s, t, n) = (shared.clone(), tx.clone(), notify.clone());
                std::thread::Builder::new()
                    .name(format!("micro-viewer tiles {i}"))
                    .spawn(move || work(s, t, n, slow))
                    .expect("a tile reading thread")
            })
            .collect();
        Self {
            shared,
            results,
            workers,
            in_flight: HashSet::new(),
            gen: 0,
            notify,
        }
    }

    /// The default number of workers: the machine's threads, 2 to 8.
    pub fn default_threads() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .clamp(2, 8)
    }

    /// Set the host's wake-up, called from a worker thread after each tile.
    pub fn set_notify(&self, f: Option<Box<dyn Fn() + Send>>) {
        *self.notify.lock().expect("loader notify") = f;
    }

    /// Replace the queue with `wanted`, in priority order (tiles being read are left
    /// alone; tiles queued before and not wanted now are dropped).
    pub fn want(&mut self, wanted: Vec<Request>) {
        let mut q = self.shared.queue.lock().expect("loader queue");
        for r in q.items.drain(..) {
            self.in_flight.remove(&r.key);
        }
        for mut r in wanted {
            r.gen = self.gen;
            if self.in_flight.insert(r.key) {
                q.items.push_back(r);
            }
        }
        drop(q);
        self.shared.ready.notify_all();
    }

    /// The tiles read since the last call.
    pub fn take(&mut self) -> Vec<Loaded> {
        let all: Vec<Loaded> = self.results.try_iter().collect();
        self.shared.waiting.fetch_sub(all.len(), Ordering::AcqRel);
        let gen = self.gen;
        all.into_iter()
            .filter(|l| l.gen == gen)
            .inspect(|l| {
                self.in_flight.remove(&l.key);
            })
            .collect()
    }

    /// Tiles requested and not back yet.
    pub fn pending(&self) -> usize {
        self.in_flight.len()
    }

    /// Whether tiles are waiting to be taken (callable from any thread's view of the
    /// viewer; cheap).
    pub fn has_results(&self) -> bool {
        self.shared.waiting.load(Ordering::Acquire) > 0
    }

    /// Drop every queued request and forget what was in flight (a new image).
    pub fn reset(&mut self) {
        self.want(Vec::new());
        self.in_flight.clear();
        self.gen += 1; // tiles of the old image still being read are dropped on arrival
        let _ = self.take();
        self.shared.blocks.clear();
    }
}

impl Drop for Loader {
    fn drop(&mut self) {
        self.shared.queue.lock().expect("loader queue").stop = true;
        self.shared.ready.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}
