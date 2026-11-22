//! The tile cache on the GPU: an atlas of `slots` tiles of `TILE` x `TILE` float32
//! texels, and a page table from each tile of the plane to its slot (0 = not loaded),
//! least recently used tiles evicted when the atlas is full.

use std::collections::HashMap;

/// Tile edge in pixels.
pub const TILE: u32 = 256;
/// Tiles per atlas row and column: a 4096 x 4096 float32 texture, 64 MiB.
pub const SLOTS_PER_ROW: u32 = 16;

/// Which tile sits in which slot, and when each was last used.
pub struct TileTable {
    pub grid_w: u32,
    pub grid_h: u32,
    /// Per tile of the plane (row-major): slot + 1, or 0.
    pub pages: Vec<u32>,
    /// Per slot: the tile it holds and the frame it was last drawn in.
    slots: Vec<Option<(u32, u64)>>,
    free: Vec<u32>,
    tile_of: HashMap<u32, u32>,
}

impl TileTable {
    pub fn new(width: u64, height: u64) -> Self {
        let grid_w = width.div_ceil(TILE as u64) as u32;
        let grid_h = height.div_ceil(TILE as u64) as u32;
        let n = SLOTS_PER_ROW * SLOTS_PER_ROW;
        Self {
            grid_w,
            grid_h,
            pages: vec![0; (grid_w * grid_h) as usize],
            slots: vec![None; n as usize],
            free: (0..n).rev().collect(),
            tile_of: HashMap::new(),
        }
    }

    /// Mark tile `t` as drawn in `frame`; whether it is loaded.
    pub fn touch(&mut self, t: u32, frame: u64) -> bool {
        match self.tile_of.get(&t) {
            Some(&s) => {
                self.slots[s as usize] = Some((t, frame));
                true
            }
            None => false,
        }
    }

    /// A slot for tile `t`: a free one, else the least recently used one not drawn in
    /// `frame` (its tile leaves the page table). None when every slot is in this frame.
    pub fn place(&mut self, t: u32, frame: u64) -> Option<u32> {
        let s = match self.free.pop() {
            Some(s) => s,
            None => {
                let (s, old) = self
                    .slots
                    .iter()
                    .enumerate()
                    .filter_map(|(s, e)| e.map(|(old, f)| (s as u32, old, f)))
                    .filter(|&(_, _, f)| f < frame)
                    .min_by_key(|&(_, _, f)| f)
                    .map(|(s, old, _)| (s, old))?;
                self.pages[old as usize] = 0;
                self.tile_of.remove(&old);
                s
            }
        };
        self.slots[s as usize] = Some((t, frame));
        self.tile_of.insert(t, s);
        self.pages[t as usize] = s + 1;
        Some(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn least_recently_used_tiles_leave() {
        let mut t = TileTable::new(256 * 20, 256 * 20); // 400 tiles, 256 slots
        for k in 0..256 {
            assert!(t.place(k, 1).is_some());
        }
        assert!(t.place(300, 1).is_none()); // every slot drawn in this frame
        assert!(t.touch(5, 2));
        let s = t.place(300, 2).unwrap();
        assert_eq!(t.pages[300], s + 1);
        assert_eq!(t.pages[0], 0); // tile 0 (frame 1) went first
        assert!(t.pages[5] != 0);
    }
}
