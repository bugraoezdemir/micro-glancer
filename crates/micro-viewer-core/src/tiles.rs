//! The tile cache on the GPU: an atlas of slots (`TILE` x `TILE` texels each, 256 slots
//! per 4096 x 4096 layer of a texture array) and which tile sits in which slot, least
//! recently used tiles evicted when the atlas is full. A tile is one channel of one
//! plane of one level; tiles of every level, channel and plane share the atlas.

use std::collections::HashMap;

/// Tile edge in pixels.
pub const TILE: u32 = 256;
/// Slots per atlas layer row and column: a layer is 4096 x 4096 texels.
pub const SLOTS_PER_ROW: u32 = 16;
/// Slots per atlas layer.
pub const SLOTS_PER_LAYER: u32 = SLOTS_PER_ROW * SLOTS_PER_ROW;

/// One tile: the plane (a hash of its indices on the axes before y and x, the channel
/// axis left out, at its level), the channel, the level and the tile's column and row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileKey {
    pub plane: u64,
    pub channel: u16,
    pub level: u8,
    pub x: u32,
    pub y: u32,
}

/// Which tile sits in which slot, and the frame each was last used in.
pub struct Cache {
    slots: Vec<Option<(TileKey, u64)>>,
    free: Vec<u32>,
    map: HashMap<TileKey, u32>,
}

impl Cache {
    pub fn new(slots: u32) -> Self {
        Self {
            slots: vec![None; slots as usize],
            free: (0..slots).rev().collect(),
            map: HashMap::new(),
        }
    }

    pub fn capacity(&self) -> u32 {
        self.slots.len() as u32
    }

    /// The slot of `key` if cached, marked as used in `frame`.
    pub fn touch(&mut self, key: &TileKey, frame: u64) -> Option<u32> {
        let s = *self.map.get(key)?;
        self.slots[s as usize] = Some((*key, frame));
        Some(s)
    }

    /// The slot of `key` if cached (not marked as used).
    pub fn peek(&self, key: &TileKey) -> Option<u32> {
        self.map.get(key).copied()
    }

    /// Every cached tile and its slot.
    pub fn iter(&self) -> impl Iterator<Item = (&TileKey, &u32)> {
        self.map.iter()
    }

    /// A slot for `key`: a free one, else the least recently used one not used in
    /// `frame` (the tile it held is returned). None when every slot is in this frame.
    pub fn place(&mut self, key: TileKey, frame: u64) -> Option<(u32, Option<TileKey>)> {
        if let Some(s) = self.touch(&key, frame) {
            return Some((s, None));
        }
        let (s, old) = match self.free.pop() {
            Some(s) => (s, None),
            None => {
                let (s, old) = self
                    .slots
                    .iter()
                    .enumerate()
                    .filter_map(|(s, e)| e.map(|(old, f)| (s as u32, old, f)))
                    .filter(|&(_, _, f)| f < frame)
                    .min_by_key(|&(_, _, f)| f)
                    .map(|(s, old, _)| (s, old))?;
                self.map.remove(&old);
                (s, Some(old))
            }
        };
        self.slots[s as usize] = Some((key, frame));
        self.map.insert(key, s);
        Some((s, old))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(x: u32) -> TileKey {
        TileKey {
            plane: 0,
            channel: 0,
            level: 0,
            x,
            y: 0,
        }
    }

    #[test]
    fn least_recently_used_tiles_leave() {
        let mut c = Cache::new(4);
        for k in 0..4 {
            assert_eq!(c.place(key(k), 1).unwrap().1, None);
        }
        assert!(c.place(key(9), 1).is_none()); // every slot used in this frame
        assert!(c.touch(&key(2), 2).is_some());
        let (s, old) = c.place(key(9), 2).unwrap();
        assert_eq!(old, Some(key(0))); // the oldest of frame 1 went first
        assert_eq!(c.peek(&key(9)), Some(s));
        assert_eq!(c.peek(&key(0)), None);
        assert!(c.peek(&key(2)).is_some());
        assert_eq!(c.place(key(9), 3).unwrap(), (s, None)); // already there
    }
}
