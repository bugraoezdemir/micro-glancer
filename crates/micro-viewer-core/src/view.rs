//! The view as data (WP14's API decision: the whole state serialises to JSON, defined
//! here in Rust; Python mirrors it with the same names). For now: the image, the camera,
//! the plane's position and the channels' display. Sources beyond one image (label
//! images, plates), coordinate systems and annotation sets join in V9 and V10.
//!
//! ```json
//! {"version": 1, "image": "C:/data/img.ome.zarr",
//!  "camera": {"center": [512.0, 384.0], "scale": 0.5},
//!  "position": {"z": 118},
//!  "channels": [{"label": "DAPI", "color": "#0000ff", "lo": 0, "hi": 1500,
//!                "gamma": 1.0, "visible": true}]}
//! ```
//!
//! Applying a state changes what it names and leaves the rest: a state with only a
//! camera moves the camera.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{ChannelDisplay, Viewer, ViewerError};

/// The current version of the view state's layout.
pub const VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraState {
    /// The world point (x, y) at the view's centre.
    pub center: [f64; 2],
    /// World units per screen pixel.
    pub scale: f64,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ChannelState {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// "#rrggbb"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lo: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hi: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gamma: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub visible: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViewState {
    #[serde(default = "version")]
    pub version: u32,
    /// The image's path (folder, .zip or .ozx).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera: Option<CameraState>,
    /// Level-0 index per plane axis (z, t, ...).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub position: BTreeMap<String, f64>,
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

impl ViewState {
    /// The viewer's current view.
    pub fn of(v: &Viewer) -> Self {
        let open = v.source().is_some();
        ViewState {
            version: VERSION,
            image: v.path().map(|p| p.to_string_lossy().into_owned()),
            camera: open.then(|| CameraState {
                center: v.center(),
                scale: v.camera.scale,
            }),
            position: v
                .position()
                .into_iter()
                .map(|(name, i, _)| (name, i))
                .collect(),
            channels: v
                .channels()
                .iter()
                .map(|d| ChannelState {
                    label: Some(d.label.clone()),
                    color: Some(hex(d.color)),
                    lo: Some(d.lo),
                    hi: Some(d.hi),
                    gamma: Some(d.gamma),
                    visible: Some(d.visible),
                })
                .collect(),
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

    /// Apply to `v` (see the module's notes). Checked first, then applied, so a state
    /// with a bad field changes nothing (beyond opening its image).
    pub fn apply(&self, v: &mut Viewer) -> Result<(), ViewerError> {
        if let Some(path) = &self.image {
            let same = v.path().is_some_and(|p| p == Path::new(path));
            if !same {
                v.open(Path::new(path))?;
            }
        }
        if let Some(c) = &self.camera {
            if !(c.scale > 0.0 && c.scale.is_finite() && c.center.iter().all(|x| x.is_finite())) {
                return Err(ViewerError(format!(
                    "view state camera at {:?} with scale {}",
                    c.center, c.scale
                )));
            }
        }
        let n = v.channels().len();
        if self.channels.len() > n {
            return Err(ViewerError(format!(
                "view state with {} channels for an image with {n}",
                self.channels.len()
            )));
        }
        let mut displays: Vec<ChannelDisplay> = v.channels().to_vec();
        for (d, s) in displays.iter_mut().zip(&self.channels) {
            if let Some(l) = &s.label {
                d.label = l.clone();
            }
            if let Some(c) = &s.color {
                d.color = parse_hex(c)?;
            }
            d.lo = s.lo.unwrap_or(d.lo);
            d.hi = s.hi.unwrap_or(d.hi);
            d.gamma = s.gamma.unwrap_or(d.gamma);
            d.visible = s.visible.unwrap_or(d.visible);
        }
        let old_position = v.position();
        for (axis, &i) in &self.position {
            if let Err(e) = v.set_position(axis, i) {
                for (a, p, _) in &old_position {
                    let _ = v.set_position(a, *p);
                }
                return Err(e);
            }
        }
        for (c, d) in displays.into_iter().enumerate() {
            v.set_channel(c, d)?;
        }
        if let Some(c) = &self.camera {
            v.look_at(c.center, c.scale);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_round_trip_and_partial_states() {
        let s = ViewState::from_json(r#"{"camera": {"center": [1.5, 2], "scale": 0.25}}"#).unwrap();
        assert_eq!(s.version, VERSION);
        assert!(s.channels.is_empty() && s.position.is_empty() && s.image.is_none());
        let back = ViewState::from_json(&s.to_json()).unwrap();
        assert_eq!(back, s);
        assert!(ViewState::from_json(r#"{"version": 99}"#).is_err());
        assert_eq!(parse_hex("#ff8000").unwrap(), [1.0, 128.0 / 255.0, 0.0]);
        assert!(parse_hex("red").is_err());
        assert_eq!(hex([0.0, 1.0, 0.5]), "#00ff80");
    }
}
