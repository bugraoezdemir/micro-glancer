//! A 2D plane of an OME-Zarr image as the viewer reads it: one level of its first
//! multiscale, the plane at index 0 of every axis before the last two, read in tiles as
//! float32 through dyna's arrays crate (the spike's scope; channels, planes and level
//! choice come in task V7).

use std::path::Path;
use std::sync::Arc;

use dyna_zarr_arrays::{read, Cast, DType, Source, SourceRef};
use ome_zarr_pyramid::Image;

use crate::ViewerError;

/// One plane of one level, read tile by tile.
pub struct Plane {
    array: SourceRef,
    /// Rows and columns of the plane (the array's last two axes).
    pub height: u64,
    pub width: u64,
    /// The array's full shape (the plane is at index 0 of the leading axes).
    shape: Vec<u64>,
    pub dtype: String,
    pub version: String,
    pub level: String,
}

fn err(e: impl std::fmt::Display) -> ViewerError {
    ViewerError(e.to_string())
}

impl Plane {
    /// Level `level` (0 the finest) of the image at `path`: a folder, or a zip file
    /// (`.zip`, `.ozx`) holding the image at its root.
    pub fn open(path: &Path, level: usize) -> Result<Self, ViewerError> {
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
        if ms.axes.len() < 2 {
            return Err(err("a plane needs at least two axes"));
        }
        let lv = ms.levels.get(level).ok_or_else(|| {
            err(format!(
                "level {level}: the image has {} levels",
                ms.levels.len()
            ))
        })?;
        let zarr = image.level_array(0, level).map_err(err)?;
        let dtype = zarr.dtype().name().to_string();
        let shape = zarr.shape().to_vec();
        let n = shape.len();
        let source: SourceRef = Arc::new(zarr);
        let array: SourceRef = if source.dtype() == DType::Float32 {
            source
        } else {
            Arc::new(Cast::new(source, DType::Float32).map_err(err)?)
        };
        Ok(Self {
            array,
            height: shape[n - 2],
            width: shape[n - 1],
            shape,
            dtype,
            version: image.version.clone(),
            level: lv.path.clone(),
        })
    }

    /// The tile at rows `y0 .. y0 + h`, columns `x0 .. x0 + w`, row-major float32.
    pub fn read_tile(&self, y0: u64, x0: u64, h: u64, w: u64) -> Result<Vec<f32>, ViewerError> {
        let n = self.shape.len();
        let mut start = vec![0u64; n];
        let mut stop = vec![1u64; n];
        start[n - 2] = y0;
        start[n - 1] = x0;
        stop[n - 2] = y0 + h;
        stop[n - 1] = x0 + w;
        let bytes = read(self.array.as_ref(), &start, &stop, 1).map_err(err)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_ne_bytes(*c))
            .collect())
    }
}
