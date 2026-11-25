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
//! A brick (a 3-D piece of a volume, for planes at any angle) has its block over its
//! last three axes: the chunk-aligned region around it in z, y and x.
//!
//! Decode units (2026-11-24, V12): when the scheduler finds that the units a view needs
//! (the inner chunks of a sharded array, else its chunks) fit half the budget, a tile is
//! put together from whole units instead ([`Blocks::read_units`]), each decoded once and
//! kept: a chunk 96 planes deep then serves every plane it holds, and every tile it
//! touches, from memory. Stepping through such a volume decoded every unit in view
//! again per plane (386 ms per step on a 96^3-chunked volume zoomed in); from the units it
//! is a copy. Same budget, same eviction.
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

/// A block: the image generation, its level array (pointer), the block's start and end
/// (a tile's block and a brick's may start at the same corner).
type Key = (u64, usize, Vec<u64>, Vec<u64>);

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

/// The chunk-aligned region around `start .. stop` over the last `spatial` axes (None:
/// read the region on its own).
fn block_of(
    array: &SourceRef,
    start: &[u64],
    stop: &[u64],
    spatial: usize,
) -> Option<(Vec<u64>, Vec<u64>)> {
    let grid = array.grid()?;
    let shape = array.shape();
    let n = shape.len();
    if grid.len() != n || n < spatial {
        return None;
    }
    let item = array.dtype().size() as u64;
    let mut b0 = start.to_vec();
    let mut b1 = stop.to_vec();
    for i in n - spatial..n {
        let g = grid[i].max(1);
        b0[i] = start[i] / g * g;
        b1[i] = (stop[i].div_ceil(g) * g).min(shape[i]);
    }
    let bytes = b0.iter().zip(&b1).map(|(a, b)| b - a).product::<u64>() * item;
    (bytes <= BLOCK_MAX && (b0 != start || b1 != stop)).then_some((b0, b1))
}

/// The region `start .. stop` out of the decoded block `b0 .. b1` (row-major bytes).
fn cut(block: &[u8], b0: &[u64], b1: &[u64], start: &[u64], stop: &[u64], item: usize) -> Vec<u8> {
    let n = b0.len();
    let shape: Vec<usize> = b0.iter().zip(b1).map(|(a, b)| (b - a) as usize).collect();
    let mut strides = vec![1usize; n];
    for i in (0..n - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    let size: Vec<usize> = start
        .iter()
        .zip(stop)
        .map(|(a, b)| (b - a) as usize)
        .collect();
    let w = size[n - 1];
    let rows: usize = size[..n - 1].iter().product();
    let mut out = Vec::with_capacity(rows * w * item);
    let mut at = vec![0usize; n - 1]; // position within the region, all axes but the last
    for _ in 0..rows {
        let base: usize = (0..n - 1)
            .map(|i| (start[i] - b0[i]) as usize * strides[i] + at[i] * strides[i])
            .sum::<usize>()
            + (start[n - 1] - b0[n - 1]) as usize;
        out.extend_from_slice(&block[base * item..(base + w) * item]);
        for i in (0..n - 1).rev() {
            at[i] += 1;
            if at[i] < size[i] {
                break;
            }
            at[i] = 0;
        }
    }
    out
}

/// Copy the part `s .. e` of the decoded block `b0 .. b1` into `out`, the region of
/// `size` that starts at `start` (all row-major bytes of `item` bytes per element).
#[allow(clippy::too_many_arguments)]
fn paste(
    block: &[u8],
    b0: &[u64],
    b1: &[u64],
    out: &mut [u8],
    start: &[u64],
    size: &[u64],
    s: &[u64],
    e: &[u64],
    item: usize,
) {
    let n = b0.len();
    let bshape: Vec<usize> = b0.iter().zip(b1).map(|(a, b)| (b - a) as usize).collect();
    let mut bstr = vec![1usize; n];
    let mut ostr = vec![1usize; n];
    for i in (0..n - 1).rev() {
        bstr[i] = bstr[i + 1] * bshape[i + 1];
        ostr[i] = ostr[i + 1] * size[i + 1] as usize;
    }
    let w = (e[n - 1] - s[n - 1]) as usize;
    if w == 0 {
        return;
    }
    let rows: usize = (0..n - 1).map(|i| (e[i] - s[i]) as usize).product();
    let mut at = vec![0usize; n - 1];
    for _ in 0..rows {
        let mut bi = (s[n - 1] - b0[n - 1]) as usize;
        let mut oi = (s[n - 1] - start[n - 1]) as usize;
        for i in 0..n - 1 {
            bi += ((s[i] - b0[i]) as usize + at[i]) * bstr[i];
            oi += ((s[i] - start[i]) as usize + at[i]) * ostr[i];
        }
        out[oi * item..(oi + w) * item].copy_from_slice(&block[bi * item..(bi + w) * item]);
        for i in (0..n - 1).rev() {
            at[i] += 1;
            if at[i] < (e[i] - s[i]) as usize {
                break;
            }
            at[i] = 0;
        }
    }
}

impl Blocks {
    /// The region `start .. stop` of `array` (image generation `gen`), as bytes; its
    /// block spans the last `spatial` axes (2 for a tile, 3 for a brick).
    pub fn read(
        &self,
        gen: u64,
        array: &SourceRef,
        start: &[u64],
        stop: &[u64],
        spatial: usize,
    ) -> Result<Vec<u8>, String> {
        let Some((b0, b1)) = block_of(array, start, stop, spatial) else {
            return read_region(array.as_ref(), start, stop).map_err(|e| e.0);
        };
        let item = array.dtype().size();
        let block = self.block(gen, array, &b0, &b1)?;
        Ok(cut(&block, &b0, &b1, start, stop, item))
    }

    /// The region `start .. stop` of `array` put together from whole decode units of
    /// shape `unit` (each decoded once and kept, see the module's notes).
    pub fn read_units(
        &self,
        gen: u64,
        array: &SourceRef,
        start: &[u64],
        stop: &[u64],
        unit: &[u64],
    ) -> Result<Vec<u8>, String> {
        let n = start.len();
        let shape = array.shape().to_vec();
        if unit.len() != n || unit.contains(&0) {
            return read_region(array.as_ref(), start, stop).map_err(|e| e.0);
        }
        let item = array.dtype().size();
        let size: Vec<u64> = (0..n).map(|i| stop[i] - start[i]).collect();
        let mut out = vec![0u8; size.iter().product::<u64>() as usize * item];
        // every unit the region touches, in row-major order of their indices
        let first: Vec<u64> = (0..n).map(|i| start[i] / unit[i]).collect();
        let last: Vec<u64> = (0..n).map(|i| (stop[i] - 1) / unit[i]).collect();
        let mut at = first.clone();
        loop {
            let u0: Vec<u64> = (0..n).map(|i| at[i] * unit[i]).collect();
            let u1: Vec<u64> = (0..n).map(|i| (u0[i] + unit[i]).min(shape[i])).collect();
            let block = self.block(gen, array, &u0, &u1)?;
            let s: Vec<u64> = (0..n).map(|i| start[i].max(u0[i])).collect();
            let e: Vec<u64> = (0..n).map(|i| stop[i].min(u1[i])).collect();
            paste(&block, &u0, &u1, &mut out, start, &size, &s, &e, item);
            // the next unit
            let mut i = n;
            loop {
                if i == 0 {
                    return Ok(out);
                }
                i -= 1;
                if at[i] < last[i] {
                    at[i] += 1;
                    break;
                }
                at[i] = first[i];
            }
        }
    }

    /// The decoded block `b0 .. b1` of `array`, from the cache or decoded now (two
    /// threads wanting the same block wait for one decode); least recently used blocks
    /// leave beyond the budget.
    fn block(
        &self,
        gen: u64,
        array: &SourceRef,
        b0: &[u64],
        b1: &[u64],
    ) -> Result<Arc<Vec<u8>>, String> {
        let item = array.dtype().size();
        let key: Key = (
            gen,
            Arc::as_ptr(array) as *const () as usize,
            b0.to_vec(),
            b1.to_vec(),
        );
        let slot = {
            let mut s = self.state.lock().expect("block cache");
            s.clock += 1;
            let clock = s.clock;
            let bytes = b0.iter().zip(b1).map(|(a, b)| b - a).product::<u64>() * item as u64;
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
                read_region(array.as_ref(), b0, b1)
                    .map(Arc::new)
                    .map_err(|e| e.0)
            })
            .clone()?;
        self.evict(&key);
        Ok(block)
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
    fn a_region_pasted_from_its_units() {
        // a 3 x 5 x 7 array of u16, value = z * 100 + y * 10 + x, cut into units 2 x 2 x 3
        let shape = [3u64, 5, 7];
        let value = |z: u64, y: u64, x: u64| (z * 100 + y * 10 + x) as u16;
        let unit = [2u64, 2, 3];
        let (start, stop) = ([1u64, 1, 2], [3u64, 4, 7]);
        let size: Vec<u64> = (0..3).map(|i| stop[i] - start[i]).collect();
        let mut out = vec![0u8; size.iter().product::<u64>() as usize * 2];
        for uz in 0..2u64 {
            for uy in 0..3u64 {
                for ux in 0..3u64 {
                    let u0 = [uz * 2, uy * 2, ux * 3];
                    let u1: Vec<u64> = (0..3).map(|i| (u0[i] + unit[i]).min(shape[i])).collect();
                    let mut block = Vec::new();
                    for z in u0[0]..u1[0] {
                        for y in u0[1]..u1[1] {
                            for x in u0[2]..u1[2] {
                                block.extend_from_slice(&value(z, y, x).to_ne_bytes());
                            }
                        }
                    }
                    let s: Vec<u64> = (0..3).map(|i| start[i].max(u0[i])).collect();
                    let e: Vec<u64> = (0..3).map(|i| stop[i].min(u1[i])).collect();
                    if (0..3).all(|i| s[i] < e[i]) {
                        paste(&block, &u0, &u1, &mut out, &start, &size, &s, &e, 2);
                    }
                }
            }
        }
        let got: Vec<u16> = out
            .chunks(2)
            .map(|c| u16::from_ne_bytes([c[0], c[1]]))
            .collect();
        let mut want = Vec::new();
        for z in start[0]..stop[0] {
            for y in start[1]..stop[1] {
                for x in start[2]..stop[2] {
                    want.push(value(z, y, x));
                }
            }
        }
        assert_eq!(got, want);
    }

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
        // a brick: both planes
        let t = cut(&block, &b0, &b1, &[10, 6, 9], &[12, 8, 11], 2);
        let v: Vec<u16> = t
            .chunks(2)
            .map(|c| u16::from_ne_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(v, vec![23, 24, 33, 34, 123, 124, 133, 134]);
    }
}
