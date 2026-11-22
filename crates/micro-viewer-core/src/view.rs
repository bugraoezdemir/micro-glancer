//! The view as data (WP14's API decision: the whole state serialises to JSON, defined
//! here in Rust; Python mirrors it with the same names): the camera and the layers, each
//! with its image, transform, plane position and channels. Label images, plates and
//! annotation sets join later (V9, V10).
//!
//! ```json
//! {"version": 3,
//!  "camera": {"center": [512.0, 384.0, 20.0], "scale": 0.5, "rotation": 0.0,
//!             "orientation": [[1, 0, 0], [0, 1, 0]]},
//!  "current": 0,
//!  "layers": [{"image": "C:/data/img.ome.zarr", "visible": true, "opacity": 1.0,
//!              "interpolation": "nearest",
//!              "transform": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
//!              "position": {"z": 20, "t": 3},
//!              "channels": [{"label": "DAPI", "color": "#0000ff", "lo": 0, "hi": 1500,
//!                            "gamma": 1.0, "visible": true}]}]}
//! ```
//!
//! The camera's `center` is a world point (x, y, z); `orientation` holds the screen's
//! right and down directions in the world; `rotation` is the turn about the view's
//! normal (degrees, as for a view along z), used when there is no `orientation`. A
//! layer's `transform` is its 3-D affine over (x, y, z) in world units: rows
//! `[a, b, c, tx]`, `[d, e, f, ty]`, `[g, h, i, tz]` (2 rows of 3 over (x, y) are read
//! too). A channel's `lut` names a colour map instead of its colour.
//!
//! Applying a state changes what it names and leaves the rest: a state with only a
//! camera moves the camera. A state whose layers name other images than the view shows
//! opens them (in its order); the same images are changed in place. Versions 1 (one
//! image: `image`, `position`, `channels` at the top) and 2 (a camera in the plane) are
//! read too.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{Affine2, Affine3, ChannelDisplay, Viewer, ViewerError};

/// The current version of the view state's layout.
pub const VERSION: u32 = 3;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraState {
    /// The world point at the view's centre: (x, y, z), or (x, y) keeping z.
    pub center: Vec<f64>,
    /// World units per screen pixel.
    pub scale: f64,
    /// Degrees the image appears turned clockwise (when there is no orientation).
    #[serde(default)]
    pub rotation: f64,
    /// The screen's right and down directions in the world.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub orientation: Option<[[f64; 3]; 2]>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ChannelState {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// "#rrggbb"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    /// A colour map's name, or "" for none (the colour).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lut: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lo: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hi: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gamma: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub visible: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct LayerState {
    /// The image's path (folder, .zip or .ozx).
    pub image: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub visible: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opacity: Option<f32>,
    /// "nearest" or "linear" (volumes).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interpolation: Option<String>,
    /// The affine over (x, y, z): 3 rows of 4 (or over (x, y): 2 rows of 3).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transform: Option<Vec<Vec<f64>>>,
    /// Level-0 index per plane axis (z, t, ...).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub position: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub channels: Vec<ChannelState>,
}

/// The projection: whether the main view shows it, what it shows ("mip", "volume",
/// "iso"), the volume's opacity per voxel, and whether it is shaded.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ProjectionState {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shown: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub density: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shading: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViewState {
    #[serde(default = "version")]
    pub version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera: Option<CameraState>,
    /// The current layer's index.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current: Option<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub layers: Vec<LayerState>,
    /// Linked views (the plane, YZ, XZ and the projection in quarters).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub linked: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection: Option<ProjectionState>,
    /// Version 1: the one image.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
    /// Version 1: the image's plane.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub position: BTreeMap<String, f64>,
    /// Version 1: the image's channels.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub channels: Vec<ChannelState>,
}

fn version() -> u32 {
    VERSION
}

fn hex(c: [f32; 3]) -> String {
    let b = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("#{:02x}{:02x}{:02x}", b(c[0]), b(c[1]), b(c[2]))
}

fn parse_hex(s: &str) -> Result<[f32; 3], ViewerError> {
    let t = s.trim_start_matches('#');
    let c = |i: usize| {
        t.get(i..i + 2)
            .and_then(|h| u8::from_str_radix(h, 16).ok())
            .map(|v| v as f32 / 255.0)
    };
    match (t.len(), c(0), c(2), c(4)) {
        (6, Some(r), Some(g), Some(b)) => Ok([r, g, b]),
        _ => Err(ViewerError(format!("colour {s:?}: expected \"#rrggbb\""))),
    }
}

/// A transform from its JSON rows: 3 of 4, or 2 of 3 (over x and y).
pub fn parse_transform(rows: &[Vec<f64>]) -> Result<Affine3, ViewerError> {
    match (
        rows.len(),
        rows.iter().map(Vec::len).collect::<Vec<_>>().as_slice(),
    ) {
        (3, [4, 4, 4]) => Ok(Affine3 {
            m: [
                [rows[0][0], rows[0][1], rows[0][2], rows[0][3]],
                [rows[1][0], rows[1][1], rows[1][2], rows[1][3]],
                [rows[2][0], rows[2][1], rows[2][2], rows[2][3]],
            ],
        }),
        (2, [3, 3]) => Ok(Affine3::from_2d(&Affine2 {
            m: [
                [rows[0][0], rows[0][1], rows[0][2]],
                [rows[1][0], rows[1][1], rows[1][2]],
            ],
        })),
        _ => Err(ViewerError(
            "a transform is 3 rows of 4 ([[a, b, c, tx], [d, e, f, ty], [g, h, i, tz]]) or \
             2 rows of 3 ([[a, b, tx], [c, d, ty]])"
                .into(),
        )),
    }
}

impl ViewState {
    /// The viewer's current view.
    pub fn of(v: &Viewer) -> Self {
        let open = !v.layers().is_empty();
        let c = &v.camera;
        ViewState {
            version: VERSION,
            camera: open.then(|| CameraState {
                center: c.center.to_vec(),
                scale: c.scale,
                rotation: c.rotation(),
                orientation: Some([c.right, c.down]),
            }),
            current: open.then(|| v.current()),
            layers: v
                .layers()
                .iter()
                .map(|l| LayerState {
                    image: l.path().to_string_lossy().into_owned(),
                    visible: Some(l.visible),
                    opacity: Some(l.opacity),
                    interpolation: l
                        .is_volume()
                        .then(|| if l.linear { "linear" } else { "nearest" }.to_string()),
                    transform: Some(l.transform().m.iter().map(|r| r.to_vec()).collect()),
                    position: l.position().into_iter().map(|(n, i, _)| (n, i)).collect(),
                    channels: l
                        .channels()
                        .iter()
                        .map(|d| ChannelState {
                            label: Some(d.label.clone()),
                            color: Some(hex(d.color)),
                            lut: Some(d.lut.clone().unwrap_or_default()),
                            lo: Some(d.lo),
                            hi: Some(d.hi),
                            gamma: Some(d.gamma),
                            visible: Some(d.visible),
                        })
                        .collect(),
                })
                .collect(),
            linked: open.then(|| v.linked()),
            projection: open.then(|| ProjectionState {
                shown: Some(v.mip()),
                mode: Some(v.projection().name().to_string()),
                density: Some(v.density()),
                shading: Some(v.shading()),
            }),
            image: None,
            position: BTreeMap::new(),
            channels: Vec::new(),
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("a view state serialises")
    }

    pub fn from_json(text: &str) -> Result<Self, ViewerError> {
        let s: ViewState =
            serde_json::from_str(text).map_err(|e| ViewerError(format!("view state: {e}")))?;
        if s.version > VERSION {
            return Err(ViewerError(format!(
                "view state version {}: this viewer reads up to {VERSION}",
                s.version
            )));
        }
        Ok(s)
    }

    /// The layers this state names (version 1's image as one layer).
    fn layer_states(&self) -> Vec<LayerState> {
        if !self.layers.is_empty() {
            return self.layers.clone();
        }
        match &self.image {
            Some(image) => vec![LayerState {
                image: image.clone(),
                position: self.position.clone(),
                channels: self.channels.clone(),
                ..Default::default()
            }],
            None if !self.position.is_empty() || !self.channels.is_empty() => {
                // version 1 without an image: the current layer
                vec![LayerState {
                    image: String::new(),
                    position: self.position.clone(),
                    channels: self.channels.clone(),
                    ..Default::default()
                }]
            }
            None => Vec::new(),
        }
    }

    /// Apply to `v` (see the module's notes). Each layer is checked before it is
    /// changed, so a bad field leaves that layer as it was (images named are opened).
    pub fn apply(&self, v: &mut Viewer) -> Result<(), ViewerError> {
        if let Some(c) = &self.camera {
            let ok = c.scale > 0.0
                && c.scale.is_finite()
                && c.rotation.is_finite()
                && (c.center.len() == 2 || c.center.len() == 3)
                && c.center.iter().all(|x| x.is_finite());
            if !ok {
                return Err(ViewerError(format!(
                    "view state camera at {:?} with scale {} and rotation {}",
                    c.center, c.scale, c.rotation
                )));
            }
        }
        let states = self.layer_states();
        let named: Vec<&str> = states
            .iter()
            .filter(|s| !s.image.is_empty())
            .map(|s| s.image.as_str())
            .collect();
        let shown: Vec<String> = v
            .layers()
            .iter()
            .map(|l| l.path().to_string_lossy().into_owned())
            .collect();
        let same = named.len() == states.len()
            && named.len() == shown.len()
            && named
                .iter()
                .zip(&shown)
                .all(|(a, b)| Path::new(a) == Path::new(b));
        if !named.is_empty() && !same {
            if named.len() != states.len() {
                return Err(ViewerError(
                    "view state: every layer needs its image when the images change".into(),
                ));
            }
            v.open(Path::new(named[0]))?;
            for path in &named[1..] {
                v.add(Path::new(path))?;
            }
        }
        // the camera first: a layer's z position then moves the camera's plane
        if let Some(c) = &self.camera {
            match c.orientation {
                Some([r, d]) => v.set_orientation(r, d)?,
                None => {
                    let (s, k) = c.rotation.to_radians().sin_cos();
                    v.set_orientation([k, -s, 0.0], [s, k, 0.0])?;
                }
            }
            let z = c.center.get(2).copied().unwrap_or(v.camera.center[2]);
            v.look_at_3d([c.center[0], c.center[1], z], c.scale);
        }
        let camera_z = self.camera.as_ref().is_some_and(|c| c.center.len() == 3);
        for (k, s) in states.iter().enumerate() {
            let i = if s.image.is_empty() { v.current() } else { k };
            apply_layer(v, i, s, camera_z)?;
        }
        if let Some(c) = self.current {
            v.set_current(c)?;
        }
        if let Some(p) = &self.projection {
            if let Some(m) = &p.mode {
                v.set_projection(crate::Projection::parse(m)?);
            }
            if let Some(d) = p.density {
                v.set_density(d)?;
            }
            if let Some(sh) = p.shading {
                v.set_shading(sh);
            }
            if let Some(shown) = p.shown {
                v.set_mip(shown);
            }
        }
        if let Some(l) = self.linked {
            v.set_linked(l);
        }
        Ok(())
    }
}

/// Apply one layer's state. With `camera_z` (the camera named its depth) a volume's z
/// position is left to the camera.
fn apply_layer(
    v: &mut Viewer,
    i: usize,
    s: &LayerState,
    camera_z: bool,
) -> Result<(), ViewerError> {
    let l = v.layer(i)?;
    let n = l.channels().len();
    if s.channels.len() > n {
        return Err(ViewerError(format!(
            "view state with {} channels for an image with {n}",
            s.channels.len()
        )));
    }
    let mut displays: Vec<ChannelDisplay> = l.channels().to_vec();
    for (d, c) in displays.iter_mut().zip(&s.channels) {
        if let Some(t) = &c.label {
            d.label = t.clone();
        }
        if let Some(t) = &c.color {
            d.color = parse_hex(t)?;
        }
        if let Some(t) = &c.lut {
            d.lut = (!t.is_empty()).then(|| t.clone());
        }
        d.lo = c.lo.unwrap_or(d.lo);
        d.hi = c.hi.unwrap_or(d.hi);
        d.gamma = c.gamma.unwrap_or(d.gamma);
        d.visible = c.visible.unwrap_or(d.visible);
    }
    let linear = match s.interpolation.as_deref() {
        None => l.linear,
        Some("linear") => true,
        Some("nearest") => false,
        Some(other) => {
            return Err(ViewerError(format!(
                "interpolation {other:?}: \"nearest\" or \"linear\""
            )))
        }
    };
    let transform = s.transform.as_deref().map(parse_transform).transpose()?;
    let z_name = l.source().z_axis.map(|a| l.source().axes[a].clone());
    let old_position = l.position();
    let (visible, opacity) = (
        s.visible.unwrap_or(l.visible),
        s.opacity.unwrap_or(l.opacity),
    );
    let old_transform = l.transform();
    if let Some(t) = transform {
        v.set_transform(i, t)?;
    }
    for (axis, &p) in &s.position {
        if camera_z && Some(axis) == z_name.as_ref() {
            continue;
        }
        if let Err(e) = v.set_position(i, axis, p) {
            for (a, p, _) in &old_position {
                if Some(a) != z_name.as_ref() {
                    let _ = v.set_position(i, a, *p);
                }
            }
            let _ = v.set_transform(i, old_transform);
            return Err(e);
        }
    }
    v.set_layer(i, visible, opacity)?;
    v.set_linear(i, linear)?;
    for (c, d) in displays.into_iter().enumerate() {
        v.set_channel(i, c, d)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_round_trip_and_partial_states() {
        let s = ViewState::from_json(r#"{"camera": {"center": [1.5, 2], "scale": 0.25}}"#).unwrap();
        assert_eq!(s.version, VERSION);
        assert_eq!(s.camera.as_ref().unwrap().rotation, 0.0);
        assert!(s.layers.is_empty() && s.image.is_none());
        let back = ViewState::from_json(&s.to_json()).unwrap();
        assert_eq!(back, s);
        assert!(ViewState::from_json(r#"{"version": 99}"#).is_err());
        assert_eq!(parse_hex("#ff8000").unwrap(), [1.0, 128.0 / 255.0, 0.0]);
        assert!(parse_hex("red").is_err());
        assert_eq!(hex([0.0, 1.0, 0.5]), "#00ff80");
        let t = parse_transform(&[vec![1.0, 0.0, 5.0], vec![0.0, 1.0, -2.0]]).unwrap();
        assert_eq!(t.m[0][3], 5.0);
        assert_eq!(t.m[2], [0.0, 0.0, 1.0, 0.0]);
        assert!(parse_transform(&[vec![1.0]]).is_err());
    }

    #[test]
    fn version_one_reads_as_one_layer() {
        let s = ViewState::from_json(
            r#"{"version": 1, "image": "a.zarr", "position": {"z": 3},
                "channels": [{"lo": 1, "hi": 9}]}"#,
        )
        .unwrap();
        let l = s.layer_states();
        assert_eq!(l.len(), 1);
        assert_eq!(l[0].image, "a.zarr");
        assert_eq!(l[0].position["z"], 3.0);
        assert_eq!(l[0].channels[0].hi, Some(9.0));
    }
}
