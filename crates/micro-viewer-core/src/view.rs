//! The view as data (WP14's API decision: the whole state serialises to JSON, defined
//! here in Rust; Python mirrors it with the same names): the camera and the layers, each
//! with its image, transform, plane position and channels. Label images, plates and
//! annotation sets join later (V9, V10).
//!
//! ```json
//! {"version": 2,
//!  "camera": {"center": [512.0, 384.0], "scale": 0.5, "rotation": 0.0},
//!  "current": 0,
//!  "layers": [{"image": "C:/data/img.ome.zarr", "visible": true, "opacity": 1.0,
//!              "transform": [[1, 0, 0], [0, 1, 0]],
//!              "position": {"z": 118},
//!              "channels": [{"label": "DAPI", "color": "#0000ff", "lo": 0, "hi": 1500,
//!                            "gamma": 1.0, "visible": true}]}]}
//! ```
//!
//! `transform` is the layer's 2-D affine over (x, y) in world units: rows
//! `[a, b, tx]`, `[c, d, ty]`. A channel's `lut` names a colour map instead of its
//! colour.
//!
//! Applying a state changes what it names and leaves the rest: a state with only a
//! camera moves the camera. A state whose layers name other images than the view shows
//! opens them (in its order); the same images are changed in place. Version 1 (one
//! image: `image`, `position`, `channels` at the top) is read as one layer.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{Affine2, ChannelDisplay, Viewer, ViewerError};

/// The current version of the view state's layout.
pub const VERSION: u32 = 2;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraState {
    /// The world point (x, y) at the view's centre.
    pub center: [f64; 2],
    /// World units per screen pixel.
    pub scale: f64,
    /// Degrees the image appears turned clockwise.
    #[serde(default)]
    pub rotation: f64,
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
    /// The 2-D affine over (x, y): [[a, b, tx], [c, d, ty]].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transform: Option<[[f64; 3]; 2]>,
    /// Level-0 index per plane axis (z, t, ...).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub position: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub channels: Vec<ChannelState>,
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

impl ViewState {
    /// The viewer's current view.
    pub fn of(v: &Viewer) -> Self {
        let open = !v.layers().is_empty();
        ViewState {
            version: VERSION,
            camera: open.then(|| CameraState {
                center: v.center(),
                scale: v.camera.scale,
                rotation: v.camera.rotation,
            }),
            current: open.then(|| v.current()),
            layers: v
                .layers()
                .iter()
                .map(|l| LayerState {
                    image: l.path().to_string_lossy().into_owned(),
                    visible: Some(l.visible),
                    opacity: Some(l.opacity),
                    transform: Some(l.transform().m),
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
            if !(c.scale > 0.0
                && c.scale.is_finite()
                && c.rotation.is_finite()
                && c.center.iter().all(|x| x.is_finite()))
            {
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
        for (k, s) in states.iter().enumerate() {
            let i = if s.image.is_empty() { v.current() } else { k };
            apply_layer(v, i, s)?;
        }
        if let Some(c) = self.current {
            v.set_current(c)?;
        }
        if let Some(c) = &self.camera {
            v.camera.rotation = c.rotation.rem_euclid(360.0);
            v.look_at(c.center, c.scale);
        }
        Ok(())
    }
}

fn apply_layer(v: &mut Viewer, i: usize, s: &LayerState) -> Result<(), ViewerError> {
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
    let old_position = l.position();
    let (visible, opacity) = (
        s.visible.unwrap_or(l.visible),
        s.opacity.unwrap_or(l.opacity),
    );
    let old_transform = l.transform();
    for (axis, &p) in &s.position {
        if let Err(e) = v.set_position(i, axis, p) {
            for (a, p, _) in &old_position {
                let _ = v.set_position(i, a, *p);
            }
            return Err(e);
        }
    }
    if let Some(m) = s.transform {
        if let Err(e) = v.set_transform(i, Affine2 { m }) {
            let _ = v.set_transform(i, old_transform);
            return Err(e);
        }
    }
    v.set_layer(i, visible, opacity)?;
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
