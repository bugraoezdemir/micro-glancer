//! An OME-Zarr image as the viewer reads it: every level of its first multiscale, each
//! placed in world space by its matrix (from ozp-in-Rust, DR9), read in tiles through
//! dyna's arrays crate.
//!
//! The last two axes are the plane's y and x. Of the axes before them, the channel axis
//! (type "channel", or named "c") gives the channels; the others (z, t, ...) are the
//! plane's position, held in level-0 index units and mapped into each level through the
//! levels' scales and translations (a coarser level of a 3-D pyramid has fewer planes).
//!
//! World coordinates follow NGFF: a level's matrix maps a pixel's index to the world
//! position of the pixel's centre. A world point shows the pixel whose centre is nearest.
//!
//! The axis right before y, when it is spatial (type "space", or named "z") and not the
//! channel axis, is the image's z: such an image is a volume, placed in 3-D by its
//! matrix's x, y and z rows and columns, and can be cut by planes at any angle.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::Arc;

use dyna_zarr_arrays::{read, Cast, DType, Source as _, SourceRef};
use ome_zarr_pyramid::Image;
use serde_json::Value;

use crate::geom::Affine3;
use crate::ViewerError;

fn err(e: impl std::fmt::Display) -> ViewerError {
    ViewerError(e.to_string())
}

/// How tiles are kept on the GPU: the image's own type where a texture format holds it
/// exactly, float32 otherwise (32- and 64-bit integers and floats, float16, booleans as
/// uint8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Texel {
    U8,
    I8,
    U16,
    I16,
    F32,
}

impl Texel {
    /// The texel type for `dtype`, and the type tiles are cast to first (None: as read).
    fn of(dtype: DType) -> Result<(Texel, Option<DType>), ViewerError> {
        Ok(match dtype {
            DType::UInt8 => (Texel::U8, None),
            DType::Int8 => (Texel::I8, None),
            DType::UInt16 => (Texel::U16, None),
            DType::Int16 => (Texel::I16, None),
            DType::Float32 => (Texel::F32, None),
            DType::Bool => (Texel::U8, Some(DType::UInt8)),
            DType::Complex64 | DType::Complex128 => {
                return Err(err(format!(
                    "a {} image: complex values have no display yet",
                    dtype.name()
                )))
            }
            _ => (Texel::F32, Some(DType::Float32)),
        })
    }

    pub fn dtype(self) -> DType {
        match self {
            Texel::U8 => DType::UInt8,
            Texel::I8 => DType::Int8,
            Texel::U16 => DType::UInt16,
            Texel::I16 => DType::Int16,
            Texel::F32 => DType::Float32,
        }
    }

    pub fn bytes(self) -> usize {
        match self {
            Texel::U8 | Texel::I8 => 1,
            Texel::U16 | Texel::I16 => 2,
            Texel::F32 => 4,
        }
    }

    /// Value `i` of `data` (native byte order).
    pub fn value(self, data: &[u8], i: usize) -> f64 {
        match self {
            Texel::U8 => data[i] as f64,
            Texel::I8 => data[i] as i8 as f64,
            Texel::U16 => u16::from_ne_bytes([data[2 * i], data[2 * i + 1]]) as f64,
            Texel::I16 => i16::from_ne_bytes([data[2 * i], data[2 * i + 1]]) as f64,
            Texel::F32 => f32::from_ne_bytes([
                data[4 * i],
                data[4 * i + 1],
                data[4 * i + 2],
                data[4 * i + 3],
            ]) as f64,
        }
    }

    /// The type's own range, for contrast limits when nothing better is known.
    pub fn range(self) -> (f64, f64) {
        match self {
            Texel::U8 => (0.0, 255.0),
            Texel::I8 => (-128.0, 127.0),
            Texel::U16 => (0.0, 65535.0),
            Texel::I16 => (-32768.0, 32767.0),
            Texel::F32 => (0.0, 1.0),
        }
    }
}

/// A 2-D affine map of (x, y): `[x', y'] = m * [x, y, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Affine2 {
    pub m: [[f64; 3]; 2],
}

impl Affine2 {
    pub const IDENTITY: Affine2 = Affine2 {
        m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    };

    /// A shift by (`dx`, `dy`).
    pub fn translation(dx: f64, dy: f64) -> Self {
        Self {
            m: [[1.0, 0.0, dx], [0.0, 1.0, dy]],
        }
    }

    /// A turn by `degrees` about (`cx`, `cy`). With y pointing down (as image rows and
    /// the screen do) a positive angle turns clockwise.
    pub fn rotation(degrees: f64, cx: f64, cy: f64) -> Self {
        let (s, c) = degrees.to_radians().sin_cos();
        Self {
            m: [[c, -s, cx - c * cx + s * cy], [s, c, cy - s * cx - c * cy]],
        }
    }

    /// `self` after `first`: x -> self(first(x)).
    pub fn after(&self, first: &Affine2) -> Self {
        let (a, b) = (&self.m, &first.m);
        let mut m = [[0.0; 3]; 2];
        for (r, row) in m.iter_mut().enumerate() {
            for (c, v) in row.iter_mut().enumerate() {
                *v = a[r][0] * b[0][c] + a[r][1] * b[1][c];
            }
            row[2] += a[r][2];
        }
        Self { m }
    }

    /// Whether the linear part is the identity (only a shift), within `tol`.
    pub fn is_translation(&self, tol: f64) -> bool {
        let m = &self.m;
        (m[0][0] - 1.0).abs() <= tol
            && m[0][1].abs() <= tol
            && m[1][0].abs() <= tol
            && (m[1][1] - 1.0).abs() <= tol
    }

    pub fn is_finite(&self) -> bool {
        self.m.iter().flatten().all(|v| v.is_finite())
    }

    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let m = &self.m;
        (
            m[0][0] * x + m[0][1] * y + m[0][2],
            m[1][0] * x + m[1][1] * y + m[1][2],
        )
    }

    /// The linear part applied to a step (`dx`, `dy`).
    pub fn linear(&self, dx: f64, dy: f64) -> (f64, f64) {
        let m = &self.m;
        (m[0][0] * dx + m[0][1] * dy, m[1][0] * dx + m[1][1] * dy)
    }

    pub fn det(&self) -> f64 {
        self.m[0][0] * self.m[1][1] - self.m[0][1] * self.m[1][0]
    }

    pub fn inverse(&self) -> Option<Self> {
        let d = self.det();
        if d == 0.0 || !d.is_finite() {
            return None;
        }
        let m = &self.m;
        let (a, b, c, e) = (m[1][1] / d, -m[0][1] / d, -m[1][0] / d, m[0][0] / d);
        Some(Self {
            m: [
                [a, b, -(a * m[0][2] + b * m[1][2])],
                [c, e, -(c * m[0][2] + e * m[1][2])],
            ],
        })
    }
}

/// One resolution level.
pub struct Level {
    pub path: String,
    pub shape: Vec<u64>,
    array: SourceRef,
    pub width: u64,
    pub height: u64,
    /// Index (x, y) to world (x, y) in the plane.
    pub to_world: Affine2,
    pub to_index: Affine2,
    /// World units per pixel (the square root of the plane's area scale).
    pub pixel: f64,
    /// Index (x, y, z) to world (x, y, z) and back; z is the identity for a 2-D image.
    pub to_world3: Affine3,
    pub to_index3: Affine3,
    /// Voxels along z (1 for a 2-D image).
    pub depth: u64,
    /// Per axis: index to world scale and translation (used for the axes before y, x).
    lead: Vec<(f64, f64)>,
}

/// A channel as the image describes it (omero), or as defaults give it.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelInfo {
    pub label: String,
    pub color: [f32; 3],
    /// omero's window (start, end), if given.
    pub window: Option<(f64, f64)>,
    pub active: bool,
}

/// An opened image.
pub struct Source {
    pub axes: Vec<String>,
    pub levels: Vec<Level>,
    pub channel_axis: Option<usize>,
    /// The z axis (see the module's notes), if the image is a volume.
    pub z_axis: Option<usize>,
    /// The axes before y and x other than the channel axis (z, t, ...).
    pub plane_axes: Vec<usize>,
    pub channels: Vec<ChannelInfo>,
    /// Where the image opens on its plane axes (level-0 indices): omero's defaultZ and
    /// defaultT where given, else 0.
    pub default_position: Vec<f64>,
    pub dtype: String,
    pub texel: Texel,
    pub version: String,
    /// The NGFF affine (n rows of n + 1, over the image's axes) of an edge from the
    /// image's own coordinate system to the "aligned" one (NGFF 0.6), if there is one.
    pub aligned: Option<Vec<Vec<f64>>>,
}

/// Default colours for `n` channels without colours of their own.
fn default_colors(n: usize) -> Vec<[f32; 3]> {
    const CYCLE: [[f32; 3]; 7] = [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ];
    match n {
        1 => vec![[1.0, 1.0, 1.0]],
        3 => vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        _ => (0..n).map(|i| CYCLE[i % CYCLE.len()]).collect(),
    }
}

/// "RRGGBB" (omero's colour) as RGB in 0..1.
fn hex_color(s: &str) -> Option<[f32; 3]> {
    let s = s.trim_start_matches('#');
    if s.len() < 6 {
        return None;
    }
    let c = |i: usize| u8::from_str_radix(s.get(i..i + 2)?, 16).ok();
    Some([
        c(0)? as f32 / 255.0,
        c(2)? as f32 / 255.0,
        c(4)? as f32 / 255.0,
    ])
}

fn channels_from(omero: Option<&Value>, n: usize) -> Vec<ChannelInfo> {
    let defaults = default_colors(n);
    let listed = omero
        .and_then(|o| o.get("channels"))
        .and_then(Value::as_array);
    let greyscale = omero
        .and_then(|o| o.pointer("/rdefs/model"))
        .and_then(Value::as_str)
        == Some("greyscale");
    (0..n)
        .map(|i| {
            let c = listed.and_then(|l| l.get(i));
            let text = |k: &str| c.and_then(|c| c.get(k)).and_then(Value::as_str);
            let num = |p: &str| c.and_then(|c| c.pointer(p)).and_then(Value::as_f64);
            let window = match (num("/window/start"), num("/window/end")) {
                (Some(a), Some(b)) if b > a => Some((a, b)),
                _ => None,
            };
            let color = if greyscale && n == 1 {
                [1.0, 1.0, 1.0]
            } else {
                text("color").and_then(hex_color).unwrap_or(defaults[i])
            };
            ChannelInfo {
                label: text("label")
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("channel {i}")),
                color,
                window,
                active: c
                    .and_then(|c| c.get("active"))
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
            }
        })
        .collect()
}

impl Source {
    /// The image at `path`: a folder, or a zip file (`.zip`, `.ozx`) holding the image at
    /// its root.
    pub fn open(path: &Path) -> Result<Self, ViewerError> {
        let zipped = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("zip") || e.eq_ignore_ascii_case("ozx"));
        let image = if zipped {
            Image::open_zip(path, "")
        } else {
            Image::open(path)
        }
        .map_err(err)?;
        let ms = image
            .multiscales
            .first()
            .ok_or_else(|| err("the image has no multiscale"))?;
        let n = ms.axes.len();
        if n < 2 {
            return Err(err("a plane needs at least two axes"));
        }
        let axes: Vec<String> = ms.axes.iter().map(|a| a.name.clone()).collect();
        let (y, x) = (n - 2, n - 1);
        let channel_axis = ms.axes[..y]
            .iter()
            .position(|a| a.kind.as_deref() == Some("channel"))
            .or_else(|| ms.axes[..y].iter().position(|a| a.name == "c"));
        let z_axis = (n >= 3)
            .then_some(n - 3)
            .filter(|&a| Some(a) != channel_axis)
            .filter(|&a| ms.axes[a].kind.as_deref() == Some("space") || ms.axes[a].name == "z");
        let mut levels = Vec::new();
        let mut texel = None;
        let mut dtype_name = String::new();
        for (l, lv) in ms.levels.iter().enumerate() {
            let zarr = image.level_array(0, l).map_err(err)?;
            let shape = zarr.shape().to_vec();
            if shape.len() != n {
                return Err(err(format!(
                    "level {:?}: {} axes in the array, {n} in the metadata",
                    lv.path,
                    shape.len()
                )));
            }
            let dtype = zarr.dtype();
            let source: SourceRef = Arc::new(zarr);
            let cast = match texel {
                None => {
                    dtype_name = dtype.name().to_string();
                    let (t, cast) = Texel::of(dtype)?;
                    texel = Some(t);
                    cast
                }
                // a level of another type than the first is cast to the first's texels
                Some(t) => (dtype != t.dtype()).then_some(t.dtype()),
            };
            let array: SourceRef = match cast {
                Some(to) => Arc::new(Cast::new(source, to).map_err(err)?),
                None => source,
            };
            let m = &lv.matrix;
            if m.ndim() != n {
                return Err(err(format!(
                    "level {:?}: a matrix over {} axes for {n} axes",
                    lv.path,
                    m.ndim()
                )));
            }
            let to_world = Affine2 {
                m: [
                    [m.get(x, x), m.get(x, y), m.get(x, n)],
                    [m.get(y, x), m.get(y, y), m.get(y, n)],
                ],
            };
            let to_index = to_world.inverse().ok_or_else(|| {
                err(format!(
                    "level {:?}: its y-x placement cannot be inverted",
                    lv.path
                ))
            })?;
            let to_world3 = match z_axis {
                None => Affine3::from_2d(&to_world),
                Some(z) => Affine3 {
                    m: [
                        [m.get(x, x), m.get(x, y), m.get(x, z), m.get(x, n)],
                        [m.get(y, x), m.get(y, y), m.get(y, z), m.get(y, n)],
                        [m.get(z, x), m.get(z, y), m.get(z, z), m.get(z, n)],
                    ],
                },
            };
            let to_index3 = to_world3.inverse().ok_or_else(|| {
                err(format!(
                    "level {:?}: its z-y-x placement cannot be inverted",
                    lv.path
                ))
            })?;
            levels.push(Level {
                path: lv.path.clone(),
                width: shape[x],
                height: shape[y],
                depth: z_axis.map_or(1, |z| shape[z]),
                to_world3,
                to_index3,
                shape,
                array,
                pixel: to_world.det().abs().sqrt(),
                to_world,
                to_index,
                lead: (0..n).map(|a| (m.get(a, a), m.get(a, n))).collect(),
            });
        }
        let texel = texel.ok_or_else(|| err("the image has no levels"))?;
        let plane_axes: Vec<usize> = (0..y).filter(|&a| Some(a) != channel_axis).collect();
        let nc = channel_axis.map_or(1, |c| levels[0].shape[c] as usize);
        let omero = image.omero.as_ref();
        let channels = channels_from(omero, nc);
        let default_position = plane_axes
            .iter()
            .map(|&a| {
                let key = match axes[a].as_str() {
                    "z" => "/rdefs/defaultZ",
                    "t" => "/rdefs/defaultT",
                    _ => return 0.0,
                };
                let v = omero
                    .and_then(|o| o.pointer(key))
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0);
                v.clamp(0.0, (levels[0].shape[a].max(1) - 1) as f64)
            })
            .collect();
        let name = |v: &Option<Value>| {
            v.as_ref()
                .and_then(|v| v.get("name"))
                .and_then(Value::as_str)
                .map(str::to_string)
        };
        let aligned = ms
            .edges
            .iter()
            .find(|e| {
                name(&e.output).as_deref() == Some(crate::ALIGNED)
                    && ms.system.is_some()
                    && name(&e.input) == ms.system
            })
            .and_then(|e| e.matrix(n).ok())
            .map(|m| m.rows());
        Ok(Self {
            aligned,
            z_axis,
            axes,
            levels,
            channel_axis,
            plane_axes,
            channels,
            default_position,
            dtype: dtype_name,
            texel,
            version: image.version.clone(),
        })
    }

    /// The level-`level` index on axis `a` of the plane at level-0 index `pos`.
    fn lead_index(&self, level: usize, a: usize, pos: f64) -> u64 {
        let (s0, t0) = self.levels[0].lead[a];
        let (s, t) = self.levels[level].lead[a];
        let i = if s != 0.0 {
            ((s0 * pos + t0 - t) / s).round()
        } else {
            pos.round()
        };
        i.clamp(0.0, (self.levels[level].shape[a].max(1) - 1) as f64) as u64
    }

    /// The plane at `position` (level-0 indices on the plane axes) at `level`: its index
    /// on every axis before y and x (the channel axis at 0), and its id.
    pub fn plane(&self, level: usize, position: &[f64]) -> (Vec<u64>, u64) {
        let n = self.axes.len();
        let mut idx = vec![0u64; n - 2];
        for (k, &a) in self.plane_axes.iter().enumerate() {
            idx[a] = self.lead_index(level, a, position.get(k).copied().unwrap_or(0.0));
        }
        let mut h = DefaultHasher::new();
        idx.hash(&mut h);
        (idx, h.finish())
    }

    /// The tile's region: rows `ty .. ty + h`, columns `tx .. tx + w` of channel
    /// `channel` of the plane `plane` (from [`Source::plane`]).
    pub fn region(
        &self,
        plane: &[u64],
        channel: usize,
        ty: u64,
        tx: u64,
        h: u64,
        w: u64,
    ) -> (Vec<u64>, Vec<u64>) {
        let mut start = plane.to_vec();
        if let Some(c) = self.channel_axis {
            start[c] = channel as u64;
        }
        let mut stop: Vec<u64> = start.iter().map(|i| i + 1).collect();
        start.extend([ty, tx]);
        stop.extend([ty + h, tx + w]);
        (start, stop)
    }

    /// The volume's lead indices at `level` without z (its other plane axes, the channel
    /// axis at 0) and their id: what a brick shares with its neighbours in z.
    pub fn lead(&self, level: usize, position: &[f64]) -> (Vec<u64>, u64) {
        let (mut idx, _) = self.plane(level, position);
        if let Some(z) = self.z_axis {
            idx[z] = 0;
        }
        let mut h = DefaultHasher::new();
        (idx.clone(), 1u8).hash(&mut h);
        (idx, h.finish())
    }

    /// A brick's region: z `z0 .. z1`, rows `y0 .. y1`, columns `x0 .. x1` of channel
    /// `channel` (lead indices from [`Source::lead`]).
    pub fn region3(
        &self,
        lead: &[u64],
        channel: usize,
        z: (u64, u64),
        y: (u64, u64),
        x: (u64, u64),
    ) -> (Vec<u64>, Vec<u64>) {
        let za = self.z_axis.expect("a volume");
        let mut start = lead.to_vec();
        if let Some(c) = self.channel_axis {
            start[c] = channel as u64;
        }
        let mut stop: Vec<u64> = start.iter().map(|i| i + 1).collect();
        start[za] = z.0;
        stop[za] = z.1;
        start.extend([y.0, x.0]);
        stop.extend([y.1, x.1]);
        (start, stop)
    }

    pub fn array(&self, level: usize) -> SourceRef {
        self.levels[level].array.clone()
    }

    /// The bytes of `start .. stop` of level `level` (texels, native byte order).
    pub fn read(&self, level: usize, start: &[u64], stop: &[u64]) -> Result<Vec<u8>, ViewerError> {
        read_region(self.levels[level].array.as_ref(), start, stop)
    }
}

/// A region of an array as bytes, on this thread.
pub fn read_region(
    array: &dyn dyna_zarr_arrays::Source,
    start: &[u64],
    stop: &[u64],
) -> Result<Vec<u8>, ViewerError> {
    read(array, start, stop, 1).map_err(err)
}

/// The lowest and highest finite value of `data` (native byte order) of `dtype` (one of
/// the texel types), or None.
pub fn value_range(dtype: DType, data: &[u8]) -> Option<(f32, f32)> {
    let texel = match dtype {
        DType::UInt8 => Texel::U8,
        DType::Int8 => Texel::I8,
        DType::UInt16 => Texel::U16,
        DType::Int16 => Texel::I16,
        DType::Float32 => Texel::F32,
        _ => return None,
    };
    let n = data.len() / texel.bytes();
    let (mut lo, mut hi) = (f64::MAX, f64::MIN);
    for i in 0..n {
        let v = texel.value(data, i);
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    (lo <= hi).then_some((lo as f32, hi as f32))
}

/// The `lo` and `hi` quantiles (0..1) of the finite values in `data`.
pub fn quantiles(texel: Texel, data: &[u8], lo: f64, hi: f64) -> Option<(f64, f64)> {
    let n = data.len() / texel.bytes();
    let mut v: Vec<f64> = (0..n)
        .map(|i| texel.value(data, i))
        .filter(|v| v.is_finite())
        .collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    let at = |q: f64| v[((v.len() - 1) as f64 * q).round() as usize];
    Some((at(lo), at(hi)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_inverse_round_trips() {
        let a = Affine2 {
            m: [[0.5, 0.1, 3.0], [-0.2, 2.0, -7.0]],
        };
        let b = a.inverse().unwrap();
        let (x, y) = a.apply(11.0, -4.0);
        let (u, v) = b.apply(x, y);
        assert!((u - 11.0).abs() < 1e-12 && (v + 4.0).abs() < 1e-12);
        let id = b.after(&a);
        assert!(id.is_translation(1e-12) && id.m[0][2].abs() < 1e-12);
    }

    #[test]
    fn rotation_turns_clockwise_about_its_centre() {
        let r = Affine2::rotation(90.0, 10.0, 20.0);
        let (x, y) = r.apply(10.0, 20.0);
        assert!((x - 10.0).abs() < 1e-12 && (y - 20.0).abs() < 1e-12);
        // y down: the point to the right of the centre goes below it
        let (x, y) = r.apply(11.0, 20.0);
        assert!((x - 10.0).abs() < 1e-12 && (y - 21.0).abs() < 1e-12);
        let t = Affine2::translation(3.0, -1.0).after(&r);
        assert_eq!(t.apply(11.0, 20.0), (13.0, 20.0));
        assert!(!r.is_translation(1e-9) && t.is_finite());
    }

    #[test]
    fn omero_channels_and_defaults() {
        let o: Value = serde_json::json!({"channels": [
            {"color": "0000FF", "label": "LaminB1", "window": {"start": 0.0, "end": 1500.0}},
            {"color": "FFFF00", "active": false}
        ]});
        let c = channels_from(Some(&o), 2);
        assert_eq!(c[0].color, [0.0, 0.0, 1.0]);
        assert_eq!(c[0].window, Some((0.0, 1500.0)));
        assert_eq!(c[1].label, "channel 1");
        assert!(!c[1].active);
        let d = channels_from(None, 3);
        assert_eq!(d[2].color, [0.0, 0.0, 1.0]);
        assert!(d.iter().all(|c| c.active && c.window.is_none()));
    }
}
