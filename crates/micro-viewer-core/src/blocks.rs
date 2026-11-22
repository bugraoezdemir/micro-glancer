//! Decoded storage blocks, shared by the reading threads.
//!
//! A tile (256 x 256 pixels of one plane) rarely matches the storage chunks: writers use
//! chunks like 1024 x 1024, 1 x 41 x 96 x 128 or 1024 x 16384, and every read decodes
//! whole chunks. Read tile by tile, a 1024 x 16384 chunk would be decoded once per tile
//! it holds (64 times). So a tile is cut out of its block, the chunk-aligned region
//! around it in its own plane, decoded once and kept here (least recently used blocks
//! leave beyond the budget); its neighbours in the plane come from the same block. A
//! block larger than [`BLOCK_MAX`] is not made: the tile is read on its own.
//!
//! Blocks stay within the tile's plane. Chunks deep in z (1 x 41 x 96 x 128) would give
//! every tile's block all 41 planes: measured on such an image with 12 channels, the
//! blocks a view needs outgrew the budget five times over and the view took 1.0 s to
//! sharpen instead of 0.58 s. The planes next to the current one are read ahead tile by
//! tile instead (the scheduler).
//!
//! Two threads wanting the same block wait for one decode (a `OnceLock` per block).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use dyna_zarr_arrays::SourceRef;

use crate::source::read_region;

/// The largest block decoded for a tile.
pub const BLOCK_MAX: u64 = 64 << 20;
/// Decoded blocks kept, in bytes.
pub const BUDGET: u64 = 512 << 20;

type Decoded = Result<Arc<Vec<u8>>, String>;

/// A block: the image's generation, its level's array (pointer), the block's start.
type Key = (u64, usize, Vec<u64>);

struct Entry {
    slot: Arc<OnceLock<Decoded>>,
    used: u64,
    bytes: u64,
}

#[derive(Default)]
struct State {
    map: HashMap<Key, Entry>,
    clock: u64,
    bytes: u64,
}

#[derive(Default)]
pub struct Blocks {
    state: Mutex<State>,
}

/// The chunk-aligned region around `start .. stop` (None: read the tile on its own).
fn block_of(array: &SourceRef, start: &[u64], stop: &[u64]) -> Option<(Vec<u64>, Vec<u64>)> {
    let grid = array.grid()?;
    let shape = array.shape();
    let n = shape.len();
    if grid.len() != n || n < 2 {
        return None;
    }
    let item = array.dtype().size() as u64;
    let mut b0 = start.to_vec();
    let mut b1 = stop.to_vec();
    for i in n - 2..n {
        let g = grid[i].max(1);
        b0[i] = start[i] / g * g;
        b1[i] = (stop[i].div_ceil(g) * g).min(shape[i]);
    }
    let bytes = b0.iter().zip(&b1).map(|(a, b)| b - a).product::<u64>() * item;
    (bytes <= BLOCK_MAX && (b0 != start || b1 != stop)).then_some((b0, b1))
}

/// The tile `start .. stop` (one index on every axis before the last two) out of the
/// decoded block `b0 .. b1`.
fn cut(block: &[u8], b0: &[u64], b1: &[u64], start: &[u64], stop: &[u64], item: usize) -> Vec<u8> {
    let n = b0.len();
    let shape: Vec<usize> = b0.iter().zip(b1).map(|(a, b)| (b - a) as usize).collect();
    let mut strides = vec![1usize; n];
    for i in (0..n - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    let base: usize = (0..n)
        .map(|i| (start[i] - b0[i]) as usize * strides[i])
        .sum();
    let (h, w) = (
        (stop[n - 2] - start[n - 2]) as usize,
        (stop[n - 1] - start[n - 1]) as usize,
    );
    let mut out = Vec::with_capacity(h * w * item);
    for r in 0..h {
        let at = (base + r * strides[n - 2]) * item;
        out.extend_from_slice(&block[at..at + w * item]);
    }
    out
}

impl Blocks {
    /// The tile `start .. stop` of `array` (image generation `gen`), as bytes.
    pub fn read(
        &self,
        gen: u64,
        array: &SourceRef,
        start: &[u64],
        stop: &[u64],
    ) -> Result<Vec<u8>, String> {
        let Some((b0, b1)) = block_of(array, start, stop) else {
            return read_region(array.as_ref(), start, stop).map_err(|e| e.0);
        };
        let item = array.dtype().size();
        let key: Key = (gen, Arc::as_ptr(array) as *const () as usize, b0.clone());
        let slot = {
            let mut s = self.state.lock().expect("block cache");
            s.clock += 1;
            let clock = s.clock;
            let bytes = b0.iter().zip(&b1).map(|(a, b)| b - a).product::<u64>() * item as u64;
            let fresh = !s.map.contains_key(&key);
            let e = s.map.entry(key.clone()).or_insert_with(|| Entry {
                slot: Arc::new(OnceLock::new()),
                used: clock,
                bytes,
            });
            e.used = clock;
            let slot = e.slot.clone();
            if fresh {
                s.bytes += bytes;
            }
            slot
        };
        let block = slot
            .get_or_init(|| {
                read_region(array.as_ref(), &b0, &b1)
                    .map(Arc::new)
                    .map_err(|e| e.0)
            })
            .clone()?;
        self.evict(&key);
        Ok(cut(&block, &b0, &b1, start, stop, item))
    }

    /// Drop least recently used decoded blocks (never `keep`, never one being decoded)
    /// until the cache is within its budget.
    fn evict(&self, keep: &Key) {
        let mut s = self.state.lock().expect("block cache");
        while s.bytes > BUDGET {
            let oldest = s
                .map
                .iter()
                .filter(|(k, e)| *k != keep && e.slot.get().is_some())
                .min_by_key(|(_, e)| e.used)
                .map(|(k, _)| k.clone());
            let Some(k) = oldest else { break };
            if let Some(e) = s.map.remove(&k) {
                s.bytes -= e.bytes;
            }
        }
    }

    /// Forget every block (a new image).
    pub fn clear(&self) {
        let mut s = self.state.lock().expect("block cache");
        s.map.clear();
        s.bytes = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tile_cut_from_its_block() {
        // a 2 x 4 x 6 block of u16 holding value = z * 100 + y * 10 + x
        let (b0, b1) = (vec![10, 4, 6], vec![12, 8, 12]);
        let mut block = Vec::new();
        for z in 0..2u16 {
            for y in 0..4u16 {
                for x in 0..6u16 {
                    block.extend_from_slice(&(z * 100 + y * 10 + x).to_ne_bytes());
                }
            }
        }
        let t = cut(&block, &b0, &b1, &[11, 5, 8], &[12, 7, 11], 2);
        let v: Vec<u16> = t
            .chunks(2)
            .map(|c| u16::from_ne_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(v, vec![112, 113, 114, 122, 123, 124]);
    }
}
