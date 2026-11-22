//! Bricks: 3-D pieces of a volume for planes at any angle.
//!
//! A brick holds BRICK^3 voxels and bricks start every STEP = BRICK - 1 voxels, so each
//! one also holds the first voxel of the next: a sample between two voxel centres finds
//! both (for trilinear sampling) in the brick of the lower one. Voxel i belongs to brick
//! i / STEP (that brick holds i and i + 1).
//!
//! On the GPU a brick is a strip of its BRICK planes side by side (STRIP_W x BRICK
//! texels) in a texture array of the volume's own texel type; 4 strips make a row of a
//! 4096-wide atlas layer and 128 rows fill it: 512 bricks per layer. Bricks of 32^3 keep
//! the waste low for a plane, which uses one 32 x 32 sheet of each brick it cuts.

use crate::geom::{cross, dot, norm, sub, Plane3, V3};

/// Brick edge in voxels (stored).
pub const BRICK: u32 = 32;
/// Voxels from one brick's start to the next.
pub const STEP: u64 = BRICK as u64 - 1;
/// A brick's strip in the atlas: its planes side by side.
pub const STRIP_W: u32 = BRICK * BRICK;
/// Strips per atlas row and rows per atlas layer (4096 x 4096 texels).
const STRIPS_PER_ROW: u32 = 4096 / STRIP_W;
const ROWS: u32 = 4096 / BRICK;
/// Bricks per atlas layer.
pub const PER_LAYER: u32 = STRIPS_PER_ROW * ROWS;

/// Bricks along an axis of `size` voxels.
pub fn count(size: u64) -> u32 {
    (size.max(1) - 1) as u32 / STEP as u32 + 1
}

/// The voxels brick `b` holds along an axis of `size` voxels: `start .. stop`.
pub fn span(b: u32, size: u64) -> (u64, u64) {
    let start = b as u64 * STEP;
    (start, (start + BRICK as u64).min(size))
}

/// Where brick slot `slot` sits: (atlas layer, x, y) of its strip.
pub fn place(slot: u32) -> (u32, u32, u32) {
    let within = slot % PER_LAYER;
    (
        slot / PER_LAYER,
        (within % STRIPS_PER_ROW) * STRIP_W,
        (within / STRIPS_PER_ROW) * BRICK,
    )
}

/// A brick read as `d` x `h` x `w` voxels of `item` bytes (z, y, x), laid out as its
/// strip: BRICK rows (y) of BRICK planes (z) of BRICK voxels (x); what the brick does not
/// fill (at the volume's far edges) is zero.
pub fn strip(data: &[u8], d: usize, h: usize, w: usize, item: usize) -> Vec<u8> {
    let b = BRICK as usize;
    let mut out = vec![0u8; b * b * b * item];
    for z in 0..d {
        for y in 0..h {
            let src = (z * h + y) * w * item;
            let dst = (y * b * b + z * b) * item;
            out[dst..dst + w * item].copy_from_slice(&data[src..src + w * item]);
        }
    }
    out
}

/// The screen point (sx, sy) whose plane point is nearest `p` (in the plane's own
/// coordinates: least squares through `ux`, `uy`).
fn to_screen(plane: &Plane3, p: V3) -> (f64, f64) {
    let d = sub(p, plane.origin);
    let (a, b, c) = (
        dot(plane.ux, plane.ux),
        dot(plane.ux, plane.uy),
        dot(plane.uy, plane.uy),
    );
    let (r0, r1) = (dot(d, plane.ux), dot(d, plane.uy));
    let det = a * c - b * b;
    if det.abs() < 1e-300 {
        return (f64::NAN, f64::NAN);
    }
    ((c * r0 - b * r1) / det, (a * r1 - b * r0) / det)
}

/// The bricks a plane (in a level's voxel indices, see [`Plane3`]) cuts within a
/// `tw` x `th` target, for a volume of `size` (x, y, z) voxels: (x, y, z) brick indices,
/// those nearest the target's centre first. Empty when the plane misses the volume.
pub fn visible(plane: &Plane3, tw: u32, th: u32, size: [u64; 3]) -> Vec<(u32, u32, u32)> {
    let (w, h) = (tw as f64, th as f64);
    let corners = [
        plane.at(0.0, 0.0),
        plane.at(w, 0.0),
        plane.at(0.0, h),
        plane.at(w, h),
    ];
    let n = cross(plane.ux, plane.uy);
    if norm(n) == 0.0 {
        return Vec::new();
    }
    // voxel ranges the target's corners span, within the volume
    let mut lo = [0i64; 3];
    let mut hi = [0i64; 3];
    for a in 0..3 {
        let mn = corners.iter().map(|c| c[a]).fold(f64::MAX, f64::min);
        let mx = corners.iter().map(|c| c[a]).fold(f64::MIN, f64::max);
        lo[a] = (mn.floor() as i64).max(0);
        hi[a] = (mx.floor() as i64 + 1).min(size[a] as i64 - 1);
        if lo[a] > hi[a] {
            return Vec::new();
        }
    }
    let brick = |v: i64| (v.max(0) as u64 / STEP) as u32;
    // walk the two axes the plane leans least along; the third comes from the plane
    let k = (0..3)
        .max_by(|&i, &j| n[i].abs().total_cmp(&n[j].abs()))
        .expect("three axes");
    let (a, b) = match k {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    // the brick's reach in screen pixels, to keep bricks just outside the target
    let step_px = norm(plane.ux).min(norm(plane.uy)).max(1e-12);
    let reach = (BRICK as f64 * 0.87) / step_px + 1.0;
    let mut out = Vec::new();
    for ba in brick(lo[a])..=brick(hi[a]) {
        for bb in brick(lo[b])..=brick(hi[b]) {
            // the column's corners (voxel cells a brick covers, its overlap included)
            let (a0, a1) = (
                ba as f64 * STEP as f64 - 0.5,
                ba as f64 * STEP as f64 + BRICK as f64 - 0.5,
            );
            let (b0, b1) = (
                bb as f64 * STEP as f64 - 0.5,
                bb as f64 * STEP as f64 + BRICK as f64 - 0.5,
            );
            let mut kmin = f64::MAX;
            let mut kmax = f64::MIN;
            for pa in [a0, a1] {
                for pb in [b0, b1] {
                    let pk = plane.origin[k]
                        - (n[a] * (pa - plane.origin[a]) + n[b] * (pb - plane.origin[b])) / n[k];
                    kmin = kmin.min(pk);
                    kmax = kmax.max(pk);
                }
            }
            let k0 = (kmin.floor() as i64).max(lo[k]);
            let k1 = (kmax.floor() as i64 + 1).min(hi[k]);
            if k0 > k1 {
                continue;
            }
            for bk in brick(k0)..=brick(k1) {
                let mut idx = [0u32; 3];
                idx[a] = ba;
                idx[b] = bb;
                idx[k] = bk;
                let centre = [
                    idx[0] as f64 * STEP as f64 + BRICK as f64 / 2.0 - 0.5,
                    idx[1] as f64 * STEP as f64 + BRICK as f64 / 2.0 - 0.5,
                    idx[2] as f64 * STEP as f64 + BRICK as f64 / 2.0 - 0.5,
                ];
                let (sx, sy) = to_screen(plane, centre);
                if sx >= -reach && sx <= w + reach && sy >= -reach && sy <= h + reach {
                    out.push((
                        (sx - w / 2.0).powi(2) + (sy - h / 2.0).powi(2),
                        (idx[0], idx[1], idx[2]),
                    ));
                }
            }
        }
    }
    out.sort_by(|p, q| p.0.total_cmp(&q.0));
    out.into_iter().map(|(_, i)| i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bricks_overlap_by_one_voxel() {
        assert_eq!(count(1), 1);
        assert_eq!(count(32), 2); // voxel 31 is in brick 1 (brick 0 holds it too)
        assert_eq!(count(31), 1);
        assert_eq!(span(1, 40), (31, 40));
        assert_eq!(place(0), (0, 0, 0));
        assert_eq!(place(5), (0, 1024, 32));
        assert_eq!(place(PER_LAYER + 1), (1, 1024, 0));
    }

    #[test]
    fn a_strip_holds_planes_side_by_side() {
        // 2 x 2 x 3 voxels of one byte: value = z * 100 + y * 10 + x
        let data: Vec<u8> = (0..2)
            .flat_map(|z| {
                (0..2).flat_map(move |y| (0..3).map(move |x| (z * 100 + y * 10 + x) as u8))
            })
            .collect();
        let s = strip(&data, 2, 2, 3, 1);
        let b = BRICK as usize;
        assert_eq!(s.len(), b * b * b);
        assert_eq!(&s[0..3], &[0, 1, 2]); // y 0, z 0
        assert_eq!(&s[b..b + 3], &[100, 101, 102]); // y 0, z 1
        assert_eq!(&s[b * b..b * b + 3], &[10, 11, 12]); // y 1, z 0
    }

    #[test]
    fn an_xy_plane_cuts_one_layer_of_bricks() {
        // a 100 x 80 x 70 volume, plane z = 40, one voxel per pixel, 100 x 80 target
        let plane = Plane3 {
            origin: [-0.5, -0.5, 40.0],
            ux: [1.0, 0.0, 0.0],
            uy: [0.0, 1.0, 0.0],
        };
        let v = visible(&plane, 100, 80, [100, 80, 70]);
        assert!(!v.is_empty() && v.iter().all(|&(_, _, z)| z == 1));
        assert_eq!(v.len() as u32, count(100) * count(80));
        // a plane outside the volume cuts nothing
        let away = Plane3 {
            origin: [-0.5, -0.5, 500.0],
            ..plane
        };
        assert!(visible(&away, 100, 80, [100, 80, 70]).is_empty());
        // a tilted plane cuts more than one layer
        let tilt = Plane3 {
            origin: [-0.5, -0.5, 10.0],
            ux: [1.0, 0.0, 0.0],
            uy: [0.0, 0.7, 0.7],
        };
        let t = visible(&tilt, 100, 80, [100, 80, 70]);
        let zs: std::collections::HashSet<u32> = t.iter().map(|b| b.2).collect();
        assert!(zs.len() >= 2);
    }
}
