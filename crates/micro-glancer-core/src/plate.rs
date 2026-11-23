//! HCS plates (NGFF "plate" and "well" metadata): the plate's layout, the navigation
//! tree of a store (plate, wells, fields, label images; series of a bioformats2raw
//! collection), and the virtual array that shows a whole plate as one image.
//!
//! A plate is shown as one image per level: its wells on a grid, one cell per well, each
//! cell holding every field of the well at its position (see `mosaic.rs`; fields without
//! positions on a small grid), or one field (the same field index in every well), with
//! a small gap between cells. Positions without a well stay 0. Every level's cells line up
//! in world space, so the viewer's levels, tiles, bricks and histograms work on a plate as
//! on any image: zoomed out it reads the wells' coarse levels, zoomed in only the wells in
//! view at finer ones.
//!
//! The layout and the tree belong in ozp's Rust crate (DR9) later; they live here until
//! that crate reads plates.

use std::sync::Arc;

use dyna_zarr_arrays::{region, DType, Out, ReadError, Source, SourceRef};
use ome_zarr_pyramid::Location;

use crate::mosaic::Placed;
use serde_json::{json, Map, Value};

/// One well: its path in the plate ("A/1"), its row and column indices, its name ("A1")
/// and its fields' paths below it ("0", "1", ...).
#[derive(Debug, Clone, PartialEq)]
pub struct WellInfo {
    pub path: String,
    pub name: String,
    pub row: usize,
    pub column: usize,
    pub fields: Vec<String>,
}

/// A plate as its metadata describes it.
#[derive(Debug, Clone, PartialEq)]
pub struct PlateInfo {
    pub name: Option<String>,
    pub rows: Vec<String>,
    pub columns: Vec<String>,
    pub wells: Vec<WellInfo>,
}

/// The NGFF object `key` of a node's attributes: under "ome" (0.5, 0.6) or at the top
/// (0.4).
fn ome<'a>(attrs: &'a Map<String, Value>, key: &str) -> Option<&'a Value> {
    attrs
        .get("ome")
        .and_then(|o| o.get(key))
        .or_else(|| attrs.get(key))
}

fn names(v: Option<&Value>) -> Vec<String> {
    v.and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .enumerate()
                .map(|(i, e)| {
                    e.get("name")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                        .unwrap_or_else(|| i.to_string())
                })
                .collect()
        })
        .unwrap_or_default()
}

/// The fields of the well at `loc` (their paths), in the order the well lists them.
fn well_fields(loc: &Location) -> Vec<String> {
    let Ok(attrs) = loc.attributes() else {
        return Vec::new();
    };
    ome(&attrs, "well")
        .and_then(|w| w.get("images"))
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|i| i.get("path").and_then(Value::as_str))
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// `f` over `items` on up to 8 threads, in order.
pub(crate) fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let threads = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(8)
        .min(items.len().max(1));
    if threads <= 1 {
        return items.iter().map(&f).collect();
    }
    let per = items.len().div_ceil(threads);
    std::thread::scope(|s| {
        let handles: Vec<_> = items
            .chunks(per)
            .map(|chunk| s.spawn(|| chunk.iter().map(&f).collect::<Vec<R>>()))
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("a plate reader thread panicked"))
            .collect()
    })
}

impl PlateInfo {
    /// The plate at `loc`, or None when `loc` is not a plate. Reads every well's
    /// metadata (for its fields), on several threads.
    pub fn read(loc: &Location) -> Option<PlateInfo> {
        let attrs = loc.attributes().ok()?;
        let plate = ome(&attrs, "plate")?;
        let rows = names(plate.get("rows"));
        let columns = names(plate.get("columns"));
        let listed: Vec<(String, usize, usize)> = plate
            .get("wells")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|w| {
                        let path = w.get("path").and_then(Value::as_str)?.to_string();
                        let index = |k: &str, pos: usize, names: &[String]| {
                            w.get(k)
                                .and_then(Value::as_u64)
                                .map(|i| i as usize)
                                .or_else(|| {
                                    // NGFF 0.4 before rowIndex: the path's parts by name
                                    let part = path.split('/').nth(pos)?;
                                    names.iter().position(|n| n == part)
                                })
                        };
                        let row = index("rowIndex", 0, &rows)?;
                        let column = index("columnIndex", 1, &columns)?;
                        Some((path, row, column))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let fields = par_map(&listed, |(path, _, _)| well_fields(&loc.child(path)));
        let wells = listed
            .into_iter()
            .zip(fields)
            .map(|((path, row, column), fields)| WellInfo {
                name: format!(
                    "{}{}",
                    rows.get(row).cloned().unwrap_or_else(|| row.to_string()),
                    columns
                        .get(column)
                        .cloned()
                        .unwrap_or_else(|| column.to_string())
                ),
                path,
                row,
                column,
                fields,
            })
            .collect();
        Some(PlateInfo {
            name: plate
                .get("name")
                .and_then(Value::as_str)
                .map(str::to_string),
            rows,
            columns,
            wells,
        })
    }

    /// The grid's size: (rows, columns), at least as large as every well's indices.
    pub fn grid(&self) -> (usize, usize) {
        let r = self.wells.iter().map(|w| w.row + 1).max().unwrap_or(0);
        let c = self.wells.iter().map(|w| w.column + 1).max().unwrap_or(0);
        (self.rows.len().max(r), self.columns.len().max(c))
    }

    /// The most fields any well has.
    pub fn field_count(&self) -> usize {
        self.wells.iter().map(|w| w.fields.len()).max().unwrap_or(0)
    }
}

/// How a plate is shown (see the module's notes): the plate, the field shown, each
/// cell's pitch and the largest field's size (level-0 pixels of the virtual image, y
/// then x), and the wells left out (with why).
#[derive(Debug, Clone, PartialEq)]
pub struct PlateLayout {
    pub info: PlateInfo,
    /// The field shown in every well, or `ALL_FIELDS` (every field at its position).
    pub field: usize,
    pub pitch: [u64; 2],
    pub cell: [u64; 2],
    pub skipped: Vec<String>,
    /// Per well (in `info.wells`' order): its fields as placed in its cell (level-0
    /// pixels from the cell's corner); empty for a well left out.
    pub fields: Vec<Placed>,
}

impl PlateLayout {
    /// The well whose cell holds level-0 pixel (`x`, `y`) of the virtual image (its index
    /// in `info.wells`), when one of its fields covers the pixel; and that field's path
    /// in the well.
    pub fn field_at(&self, x: u64, y: u64) -> Option<(usize, String)> {
        let (r, c) = ((y / self.pitch[0]) as usize, (x / self.pitch[1]) as usize);
        let (cy, cx) = (y % self.pitch[0], x % self.pitch[1]);
        if cy >= self.cell[0] || cx >= self.cell[1] {
            return None;
        }
        let wi = self
            .info
            .wells
            .iter()
            .position(|w| w.row == r && w.column == c)?;
        let placed = self.fields.get(wi)?;
        let k = placed.at(cx, cy)?;
        Some((wi, placed.rels[k].clone()))
    }

    /// The well whose field covers level-0 pixel (`x`, `y`) of the virtual image.
    pub fn well_at(&self, x: u64, y: u64) -> Option<&WellInfo> {
        self.field_at(x, y).map(|(wi, _)| &self.info.wells[wi])
    }

    /// Where well `wi`'s fields lie in its cell: [y0, x0, y1, x1) in level-0 pixels from
    /// the cell's corner, or the whole cell when nothing is known.
    pub fn well_extent(&self, wi: usize) -> [u64; 4] {
        match self.fields.get(wi) {
            Some(p) if !p.rects.is_empty() => {
                let mut e = [u64::MAX, u64::MAX, 0, 0];
                for r in &p.rects {
                    e[0] = e[0].min(r[0]);
                    e[1] = e[1].min(r[1]);
                    e[2] = e[2].max(r[2]);
                    e[3] = e[3].max(r[3]);
                }
                e
            }
            _ => [0, 0, self.cell[0], self.cell[1]],
        }
    }
}

/// One level of a plate as one array: the wells' arrays of that level on a grid of cells
/// `pitch` (y, x) apart. A read leaves the positions without a well as they are: the
/// viewer reads it through dyna's `read` (a zeroed destination, its corner at 0) and
/// never wraps it in another view.
pub struct PlateArray {
    shape: Vec<u64>,
    dtype: DType,
    pitch: [u64; 2],
    columns: usize,
    /// Row-major over the grid: the cell's array, if a well fills it.
    cells: Vec<Option<SourceRef>>,
}

impl PlateArray {
    /// `cells` (row, column, array: all of one dtype and number of axes) on a grid of
    /// `rows` x `columns` cells `pitch` apart; the lead axes as long as `lead`.
    pub fn new(
        cells: Vec<(usize, usize, SourceRef)>,
        rows: usize,
        columns: usize,
        pitch: [u64; 2],
        lead: Vec<u64>,
        dtype: DType,
    ) -> Self {
        let mut grid: Vec<Option<SourceRef>> = vec![None; rows * columns];
        for (r, c, a) in cells {
            if r < rows && c < columns {
                grid[r * columns + c] = Some(a);
            }
        }
        let mut shape = lead;
        shape.extend([rows as u64 * pitch[0], columns as u64 * pitch[1]]);
        Self {
            shape,
            dtype,
            pitch,
            columns,
            cells: grid,
        }
    }

    /// The cells a read of `start .. stop` touches: (cell array, its start, its stop,
    /// where its region lands in the destination).
    #[allow(clippy::type_complexity)]
    fn jobs(&self, start: &[u64], stop: &[u64]) -> Vec<(&SourceRef, Vec<u64>, Vec<u64>, Vec<u64>)> {
        let n = self.shape.len();
        let (y, x) = (n - 2, n - 1);
        let rows = (start[y] / self.pitch[0])..stop[y].div_ceil(self.pitch[0]);
        let cols = (start[x] / self.pitch[1])..stop[x].div_ceil(self.pitch[1]);
        let mut jobs = Vec::new();
        for r in rows {
            for c in cols.clone() {
                let Some(Some(a)) = self.cells.get(r as usize * self.columns + c as usize) else {
                    continue;
                };
                let shape = a.shape();
                let origin = |k: usize| match k {
                    _ if k == y => r * self.pitch[0],
                    _ if k == x => c * self.pitch[1],
                    _ => 0,
                };
                let mut s = vec![0; n];
                let mut e = vec![0; n];
                let mut at = vec![0; n];
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
                    at[k] = lo - start[k];
                }
                if !empty {
                    jobs.push((a, s, e, at));
                }
            }
        }
        jobs
    }
}

impl Source for PlateArray {
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
                "reading a {} plate into a {} destination",
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
        let mut most = 0;
        for (a, s, e, _) in self.jobs(start, stop) {
            most = most.max(a.cost(&s, &e, threads)?);
        }
        Ok(most)
    }
}

/// A shared plate array.
pub fn plate_array(a: PlateArray) -> SourceRef {
    Arc::new(a)
}

/// The navigation tree of the store at `loc` (named `name`, at `path` on disk), as JSON:
/// `{"kind", "name", "path", "children"}`. A plate lists its wells, each well its fields
/// (kind "image", with "field"); an image lists its label images (kind "label"); a
/// bioformats2raw collection its series. A field's label images are not read here (ask
/// for the field's own tree); a well on its own its fields. `kind` is "plate", "well",
/// "image", "label", "collection" or "group" (a Zarr group without NGFF metadata the
/// viewer knows).
pub fn tree(loc: &Location, name: &str, path: &str) -> Result<Value, String> {
    let attrs = loc.attributes().map_err(|e| e.to_string())?;
    let join = |rel: &str| format!("{}/{}", path.trim_end_matches(['/', '\\']), rel);
    if ome(&attrs, "plate").is_some() {
        let info = PlateInfo::read(loc).ok_or("unreadable plate metadata")?;
        let mut wells = info.wells.clone();
        wells.sort_by_key(|w| (w.row, w.column));
        let children: Vec<Value> = wells
            .iter()
            .map(|w| {
                let fields: Vec<Value> = w
                    .fields
                    .iter()
                    .enumerate()
                    .map(|(k, f)| {
                        json!({"kind": "image", "name": format!("field {f}"),
                               "path": join(&format!("{}/{f}", w.path)), "field": k,
                               "well": w.name, "children": []})
                    })
                    .collect();
                json!({"kind": "well", "name": w.name, "path": join(&w.path),
                       "row": w.row, "column": w.column, "children": fields})
            })
            .collect();
        let (rows, columns) = info.grid();
        return Ok(
            json!({"kind": "plate", "name": info.name.clone().unwrap_or(name.to_string()),
                         "path": path, "rows": info.rows, "columns": info.columns,
                         "grid": [rows, columns], "fields": info.field_count(),
                         "children": children}),
        );
    }
    if ome(&attrs, "well").is_some() && ome(&attrs, "multiscales").is_none() {
        // a well on its own: its fields
        let children: Vec<Value> = well_fields(loc)
            .iter()
            .enumerate()
            .map(|(k, f)| {
                json!({"kind": "image", "name": format!("field {f}"), "path": join(f),
                       "field": k, "children": []})
            })
            .collect();
        return Ok(json!({"kind": "well", "name": name, "path": path, "children": children}));
    }
    if ome(&attrs, "multiscales").is_some() {
        let label = ome(&attrs, "image-label").is_some();
        let labels = loc
            .child("labels")
            .attributes()
            .ok()
            .and_then(|a| ome(&a, "labels").cloned())
            .and_then(|v| v.as_array().cloned())
            .unwrap_or_default();
        let children: Vec<Value> = labels
            .iter()
            .filter_map(Value::as_str)
            .map(|l| {
                json!({"kind": "label", "name": l, "path": join(&format!("labels/{l}")),
                       "children": []})
            })
            .collect();
        return Ok(
            json!({"kind": if label { "label" } else { "image" }, "name": name,
                         "path": path, "children": children}),
        );
    }
    if ome(&attrs, "bioformats2raw.layout").is_some() || attrs.contains_key("bioformats2raw.layout")
    {
        // the series "0", "1", ... (the zip and the folder alike: no listing needed)
        let mut children = Vec::new();
        for k in 0.. {
            let child = loc.child(&k.to_string());
            match tree(&child, &format!("series {k}"), &join(&k.to_string())) {
                Ok(t) => children.push(t),
                Err(_) => break,
            }
            if k >= 10_000 {
                break;
            }
        }
        return Ok(json!({"kind": "collection", "name": name, "path": path,
                         "children": children}));
    }
    Ok(json!({"kind": "group", "name": name, "path": path, "children": []}))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An array of a shape only (the layout test never reads it).
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
            Ok(0)
        }
    }

    #[test]
    fn jobs_cut_reads_into_cells() {
        let a: SourceRef = Arc::new(Shaped(vec![1, 10, 12]));
        let b: SourceRef = Arc::new(Shaped(vec![1, 10, 12]));
        let p = PlateArray::new(
            vec![(0, 0, a), (1, 2, b)],
            2,
            3,
            [11, 14],
            vec![1],
            DType::UInt16,
        );
        assert_eq!(p.shape(), &[1, 22, 42]);
        // a read across cell (0, 0)'s lower-right corner and the gap
        let jobs = p.jobs(&[0, 5, 8], &[1, 15, 20]);
        assert_eq!(jobs.len(), 1);
        let (_, s, e, at) = &jobs[0];
        assert_eq!(
            (s.clone(), e.clone(), at.clone()),
            (vec![0, 5, 8], vec![1, 10, 12], vec![0, 0, 0])
        );
        // cell (1, 2): rows 11.., columns 28..
        let jobs = p.jobs(&[0, 12, 30], &[1, 22, 42]);
        assert_eq!(jobs.len(), 1);
        let (_, s, e, at) = &jobs[0];
        assert_eq!(
            (s.clone(), e.clone(), at.clone()),
            (vec![0, 1, 2], vec![1, 10, 12], vec![0, 0, 0])
        );
        // nothing but gaps and empty cells
        assert!(p.jobs(&[0, 0, 14], &[1, 11, 28]).is_empty());
    }
}
