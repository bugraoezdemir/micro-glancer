//! Several images shown as one, each at its position: a well's fields, the series of a
//! collection (a mosaic's tiles, an acquisition's views or positions), the fields inside
//! each cell of a plate.
//!
//! Every member is opened with its levels' matrices and its placement (see
//! `placement.rs`), so a member's level `l` maps its pixels into the shared system. At
//! each level the members are put on one pixel grid: the grid's pixel is the first
//! member's, its corner the smallest member corner, and each member lands at its
//! position rounded to the nearest pixel of that level (half a pixel at most off; the
//! members' own pixels are never resampled). Members of another pixel size, other axes
//! or another number of channels are left out, and named. Where members overlap, the
//! later one in the list is shown.
//!
//! Members that all sit at the same place (a well whose fields carry no positions) are
//! laid out on a grid in their order instead, a small gap apart, and the layout says so.
//!
//! A read leaves the places no member covers as they are: the viewer reads through dyna's
//! `read` (a zeroed destination), and dyna's views above read into zeroed buffers.

use std::sync::Arc;

use dyna_zarr_arrays::{region, DType, Out, ReadError, Source, SourceRef};
use dyna_zarr_transforms::Matrix;
use ome_zarr_pyramid::{Image, Location};

use crate::placement::{placement, Scene};

/// Images placed on one pixel grid: (array, its corner (y, x) on the grid), the later
/// shown where they overlap.
pub struct MosaicArray {
    shape: Vec<u64>,
    dtype: DType,
    pieces: Vec<(SourceRef, [u64; 2])>,
}

impl MosaicArray {
    /// `pieces` (all of one dtype and number of axes) on a grid at least `size` (y, x)
    /// large, the lead axes as long as `lead`.
    pub fn new(
        pieces: Vec<(SourceRef, [u64; 2])>,
        lead: Vec<u64>,
        size: [u64; 2],
        dtype: DType,
    ) -> Self {
        let n = lead.len() + 2;
        let mut sy = size[0];
        let mut sx = size[1];
        for (a, at) in &pieces {
            let s = a.shape();
            sy = sy.max(at[0] + s[n - 2]);
            sx = sx.max(at[1] + s[n - 1]);
        }
        let mut shape = lead;
        shape.extend([sy, sx]);
        Self {
            shape,
            dtype,
            pieces,
        }
    }

    /// The pieces a read of `start .. stop` touches: (array, its start, its stop, where
    /// its region lands in the destination), in the order they are drawn.
    #[allow(clippy::type_complexity)]
    fn jobs(&self, start: &[u64], stop: &[u64]) -> Vec<(&SourceRef, Vec<u64>, Vec<u64>, Vec<u64>)> {
        let n = self.shape.len();
        let (y, x) = (n - 2, n - 1);
        let mut jobs = Vec::new();
        for (a, at) in &self.pieces {
            let shape = a.shape();
            let origin = |k: usize| match k {
                _ if k == y => at[0],
                _ if k == x => at[1],
                _ => 0,
            };
            let mut s = vec![0; n];
            let mut e = vec![0; n];
            let mut put = vec![0; n];
            let mut empty = false;
            for k in 0..n {
                let o = origin(k);
                let lo = start[k].max(o);
                let hi = stop[k].min(o + shape[k]);
                if lo >= hi {
                    empty = true;
                    break;
                }
                s[k] = lo - o;
                e[k] = hi - o;
                put[k] = lo - start[k];
            }
            if !empty {
                jobs.push((a, s, e, put));
            }
        }
        jobs
    }
}

impl Source for MosaicArray {
    fn shape(&self) -> &[u64] {
        &self.shape
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn read_into(
        &self,
        start: &[u64],
        stop: &[u64],
        out: &Out,
        threads: usize,
    ) -> Result<(), ReadError> {
        region(&self.shape, start, stop)?;
        if out.dtype() != self.dtype {
            return Err(ReadError::new(format!(
                "reading a {} mosaic into a {} destination",
                self.dtype.name(),
                out.dtype().name()
            )));
        }
        for (a, s, e, at) in self.jobs(start, stop) {
            a.read_into(&s, &e, &out.moved(at), threads)?;
        }
        Ok(())
    }

    fn cost(&self, start: &[u64], stop: &[u64], threads: usize) -> Result<u64, ReadError> {
        region(&self.shape, start, stop)?;
        let mut total = 0u64;
        for (a, s, e, _) in self.jobs(start, stop) {
            total = total.saturating_add(a.cost(&s, &e, threads)?);
        }
        Ok(total)
    }
}

/// One member opened: its name, its path below the group, the image, its levels' arrays
/// and each level's matrix with its placement (index to the shared system).
pub struct Member {
    pub name: String,
    pub rel: String,
    pub image: Image,
    pub arrays: Vec<SourceRef>,
    pub matrices: Vec<Matrix>,
}

/// The member at `rel` below the group at `group` (whose scene, if it has one, is
/// `scene`), named `name`.
pub fn open_member(
    group: &Location,
    scene: Option<&Scene>,
    rel: &str,
    name: &str,
) -> Result<Member, String> {
    let loc = group.child(rel);
    let image = Image::at(loc.clone()).map_err(|e| e.to_string())?;
    let ms = image
        .multiscales
        .first()
        .ok_or_else(|| "no multiscale".to_string())?;
    let axes: Vec<String> = ms.axes.iter().map(|a| a.name.clone()).collect();
    // the group's own scene first (one read for all its members), else the image's
    let place = match (scene, ms.system.as_deref()) {
        (Some(s), Some(system)) => s
            .placement(rel, system, &axes, None)
            .or_else(|| placement(&loc, &image)),
        (_, Some(_)) => placement(&loc, &image),
        _ => None,
    };
    let mut arrays = Vec::new();
    let mut matrices = Vec::new();
    for (l, lv) in ms.levels.iter().enumerate() {
        arrays.push(Arc::new(image.level_array(0, l).map_err(|e| e.to_string())?) as SourceRef);
        matrices.push(match &place {
            Some(p) => p.matmul(&lv.matrix).map_err(|e| e.to_string())?,
            None => lv.matrix.clone(),
        });
    }
    if arrays.is_empty() {
        return Err("no levels".into());
    }
    Ok(Member {
        name: name.to_string(),
        rel: rel.to_string(),
        image,
        arrays,
        matrices,
    })
}

/// How members are put together: each member's offset in level-0 world units (x, y)
/// added to its position (all 0 when the members carry positions), and whether they were
/// laid out on a grid because they all sat at one place.
#[derive(Debug, Clone, PartialEq)]
pub struct Arrangement {
    pub extra: Vec<(f64, f64)>,
    pub gridded: bool,
}

/// A member's y-x placement at level `l`: (scale x, scale y, translation x, translation
/// y), or why it cannot be put on a grid (rotated, sheared, or a zero scale).
fn yx(m: &Matrix, n: usize) -> Result<(f64, f64, f64, f64), String> {
    let (y, x) = (n - 2, n - 1);
    let tol = 1e-9 * (m.get(x, x).abs() + m.get(y, y).abs());
    if m.get(x, y).abs() > tol || m.get(y, x).abs() > tol {
        return Err("rotated or sheared in y-x".into());
    }
    let (sx, sy) = (m.get(x, x), m.get(y, y));
    if !(sx > 0.0 && sy > 0.0) {
        return Err(format!(
            "y-x scales ({sy}, {sx}): positive scales are needed"
        ));
    }
    Ok((sx, sy, m.get(x, n), m.get(y, n)))
}

/// The members that fit the first one (same axes, same lead shape, same pixel size at
/// level 0), each with its tag, and the others with why. The first member decides.
pub fn compatible<T>(members: Vec<(T, Member)>) -> (Vec<(T, Member)>, Vec<String>) {
    let mut kept: Vec<(T, Member)> = Vec::new();
    let mut skipped = Vec::new();
    let axes = |m: &Member| -> Vec<String> {
        m.image.multiscales[0]
            .axes
            .iter()
            .map(|a| a.name.clone())
            .collect()
    };
    for (tag, m) in members {
        let Some((_, first)) = kept.first() else {
            match yx(&m.matrices[0], m.arrays[0].shape().len()) {
                Ok(_) => kept.push((tag, m)),
                Err(e) => skipped.push(format!("{}: {e}", m.name)),
            }
            continue;
        };
        let n = first.arrays[0].shape().len();
        if axes(&m) != axes(first) || m.arrays[0].shape().len() != n {
            skipped.push(format!("{}: other axes than {}", m.name, first.name));
            continue;
        }
        if m.arrays[0].shape()[..n - 2] != first.arrays[0].shape()[..n - 2] {
            skipped.push(format!(
                "{}: other channels or planes than {}",
                m.name, first.name
            ));
            continue;
        }
        match (yx(&m.matrices[0], n), yx(&first.matrices[0], n)) {
            (Ok((sx, sy, ..)), Ok((fx, fy, ..))) => {
                let close = |a: f64, b: f64| (a - b).abs() <= 1e-6 * b.abs();
                if !(close(sx, fx) && close(sy, fy)) {
                    skipped.push(format!(
                        "{}: pixel size ({sy}, {sx}), {} has ({fy}, {fx})",
                        m.name, first.name
                    ));
                    continue;
                }
                kept.push((tag, m));
            }
            (Err(e), _) | (_, Err(e)) => skipped.push(format!("{}: {e}", m.name)),
        }
    }
    (kept, skipped)
}

/// How `members` (compatible, see [`compatible`]) are arranged: at their positions, or,
/// when they all sit at one place, on a grid of about square shape in their order, each
/// cell the largest member plus a gap of 2 % (at least 2 pixels).
pub fn arrange(members: &[Member]) -> Arrangement {
    let n = members[0].arrays[0].shape().len();
    let at: Vec<(f64, f64, f64, f64)> = members
        .iter()
        .map(|m| yx(&m.matrices[0], n).unwrap_or((1.0, 1.0, 0.0, 0.0)))
        .collect();
    let (sx, sy) = (at[0].0, at[0].1);
    let same = members.len() > 1
        && at
            .iter()
            .all(|p| (p.2 - at[0].2).abs() < 0.5 * sx && (p.3 - at[0].3).abs() < 0.5 * sy);
    if !same {
        return Arrangement {
            extra: vec![(0.0, 0.0); members.len()],
            gridded: false,
        };
    }
    let most = |k: usize| {
        members
            .iter()
            .map(|m| m.arrays[0].shape()[k])
            .max()
            .unwrap_or(1)
    };
    let (h, w) = (most(n - 2), most(n - 1));
    let (py, px) = (h + (h / 50).max(2), w + (w / 50).max(2));
    let cols = (members.len() as f64).sqrt().ceil().max(1.0) as usize;
    Arrangement {
        extra: (0..members.len())
            .map(|i| {
                (
                    (i % cols) as f64 * px as f64 * sx,
                    (i / cols) as f64 * py as f64 * sy,
                )
            })
            .collect(),
        gridded: true,
    }
}

/// The members at level `l` on one grid: each member's index and corner (y, x), the
/// grid's size (y, x), its pixel (x, y) and the world position (x, y) of its pixel
/// (0, 0)'s centre.
#[derive(Debug, Clone, PartialEq)]
pub struct LevelLayout {
    pub pieces: Vec<(usize, [u64; 2])>,
    pub size: [u64; 2],
    pub pixel: (f64, f64),
    pub origin: (f64, f64),
}

/// The layout of `members` at level `l` (every member has it), arranged by `arr`.
pub fn layout(members: &[Member], arr: &Arrangement, l: usize) -> LevelLayout {
    let n = members[0].arrays[0].shape().len();
    let at: Vec<(f64, f64, f64, f64)> = members
        .iter()
        .zip(&arr.extra)
        .map(|(m, e)| {
            let (sx, sy, tx, ty) = yx(&m.matrices[l], n).unwrap_or((1.0, 1.0, 0.0, 0.0));
            (sx, sy, tx + e.0, ty + e.1)
        })
        .collect();
    let (sx, sy) = (at[0].0, at[0].1);
    let ox = at.iter().map(|p| p.2).fold(f64::INFINITY, f64::min);
    let oy = at.iter().map(|p| p.3).fold(f64::INFINITY, f64::min);
    let mut pieces = Vec::new();
    let (mut h, mut w) = (0u64, 0u64);
    for (i, (m, p)) in members.iter().zip(&at).enumerate() {
        let cy = ((p.3 - oy) / sy).round().max(0.0) as u64;
        let cx = ((p.2 - ox) / sx).round().max(0.0) as u64;
        let s = m.arrays[l].shape();
        h = h.max(cy + s[n - 2]);
        w = w.max(cx + s[n - 1]);
        pieces.push((i, [cy, cx]));
    }
    LevelLayout {
        pieces,
        size: [h, w],
        pixel: (sx, sy),
        origin: (ox, oy),
    }
}

/// The members' names and their rectangles at level 0 ([y0, x0, y1, x1), grid pixels),
/// for what is under the pointer: the last member covering a pixel is the one shown.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Placed {
    pub names: Vec<String>,
    pub rels: Vec<String>,
    pub rects: Vec<[u64; 4]>,
    pub gridded: bool,
    pub skipped: Vec<String>,
}

impl Placed {
    pub fn of(members: &[Member], l0: &LevelLayout, gridded: bool, skipped: Vec<String>) -> Self {
        let n = members[0].arrays[0].shape().len();
        let rects = l0
            .pieces
            .iter()
            .map(|(i, at)| {
                let s = members[*i].arrays[0].shape();
                [at[0], at[1], at[0] + s[n - 2], at[1] + s[n - 1]]
            })
            .collect();
        Placed {
            names: members.iter().map(|m| m.name.clone()).collect(),
            rels: members.iter().map(|m| m.rel.clone()).collect(),
            rects,
            gridded,
            skipped,
        }
    }

    /// The member shown at grid pixel (`x`, `y`) of level 0: its index.
    pub fn at(&self, x: u64, y: u64) -> Option<usize> {
        self.rects
            .iter()
            .rposition(|r| y >= r[0] && y < r[2] && x >= r[1] && x < r[3])
    }
}

/// A shared mosaic array.
pub fn mosaic_array(a: MosaicArray) -> SourceRef {
    Arc::new(a)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Shaped(Vec<u64>);

    impl Source for Shaped {
        fn shape(&self) -> &[u64] {
            &self.0
        }
        fn dtype(&self) -> DType {
            DType::UInt16
        }
        fn read_into(&self, _: &[u64], _: &[u64], _: &Out, _: usize) -> Result<(), ReadError> {
            Err(ReadError::new("not read".into()))
        }
        fn cost(&self, _: &[u64], _: &[u64], _: usize) -> Result<u64, ReadError> {
            Ok(1)
        }
    }

    #[test]
    fn jobs_cut_reads_into_pieces_in_drawing_order() {
        let a: SourceRef = Arc::new(Shaped(vec![2, 10, 12]));
        let b: SourceRef = Arc::new(Shaped(vec![2, 10, 12]));
        let m = MosaicArray::new(
            vec![(a, [0, 0]), (b, [5, 8])],
            vec![2],
            [0, 0],
            DType::UInt16,
        );
        assert_eq!(m.shape(), &[2, 15, 20]);
        // the overlap: both, the later last
        let jobs = m.jobs(&[0, 6, 9], &[2, 8, 11]);
        assert_eq!(jobs.len(), 2);
        assert_eq!(
            (jobs[1].1.clone(), jobs[1].3.clone()),
            (vec![0, 1, 1], vec![0, 0, 0])
        );
        // the gap beside the first
        assert!(m.jobs(&[0, 0, 12], &[2, 5, 20]).is_empty());
    }

    #[test]
    fn a_pixel_shows_the_last_member_covering_it() {
        let p = Placed {
            names: vec!["a".into(), "b".into()],
            rels: vec!["0".into(), "1".into()],
            rects: vec![[0, 0, 10, 10], [5, 5, 15, 15]],
            gridded: false,
            skipped: vec![],
        };
        assert_eq!(p.at(2, 2), Some(0));
        assert_eq!(p.at(6, 6), Some(1));
        assert_eq!(p.at(12, 2), None);
    }
}
