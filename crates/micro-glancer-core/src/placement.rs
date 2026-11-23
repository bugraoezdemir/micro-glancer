//! Where an image sits among its neighbours: the transformation from its own coordinate
//! system into the system it shares with the other images of its collection (a well's
//! fields, a mosaic's tiles, the views or positions of an acquisition written as series).
//!
//! How the files hold positions (ozp's spatial model, plan 5.5 / 5.7 / 5.18):
//!
//! - NGFF 0.4 / 0.5: in the image's own multiscale (its levels' translations). The levels'
//!   matrices place it already; nothing more is needed.
//! - NGFF 0.6: in the `scene` of the group that owns the images (`ome.scene`): one edge
//!   per image, from `{"name": <its own system>, "path": <its path below the group>}` to
//!   the shared system, and the image's own levels carry no position. ozp marks such an
//!   image with `eubi:scene`, the path from it to that group (".."); without the mark the
//!   groups above it are searched.
//! - A label image (`<image>/labels/<name>`) sits where its image sits.
//!
//! The placement is a matrix over the image's own axes (by name; homogeneous), applied
//! after its levels' matrices: `world = placement * level matrix * index`.

use dyna_zarr_transforms::{Matrix, Transformation};
use ome_zarr_pyramid::{Image, Location};
use serde_json::{Map, Value};

/// A coordinate transformation end's system name (`{"name": ...}` or a bare string).
fn end_name(end: Option<&Value>) -> Option<&str> {
    match end? {
        Value::String(s) => Some(s.as_str()),
        Value::Object(m) => m.get("name").and_then(Value::as_str),
        _ => None,
    }
}

fn end_path(end: Option<&Value>) -> Option<String> {
    end?.get("path").and_then(Value::as_str).map(clean)
}

/// A relative path without "./", doubled or trailing slashes.
fn clean(p: &str) -> String {
    p.split('/')
        .filter(|s| !s.is_empty() && *s != ".")
        .collect::<Vec<_>>()
        .join("/")
}

/// The NGFF object `key` of a node's attributes: under "ome" (0.5, 0.6) or at the top.
fn ome<'a>(attrs: &'a Map<String, Value>, key: &str) -> Option<&'a Value> {
    attrs
        .get("ome")
        .and_then(|o| o.get(key))
        .or_else(|| attrs.get(key))
}

/// The groups above `loc`, nearest first, with `loc`'s path below each ("0", "1/0"), at
/// most `depth` of them. A folder's parents are taken while they are Zarr groups; in a zip,
/// up to its root.
fn ancestors(loc: &Location, depth: usize) -> Vec<(Location, String)> {
    let mut out = Vec::new();
    match loc {
        Location::Folder(p) => {
            let mut rel: Vec<String> = Vec::new();
            let mut here = p.clone();
            while out.len() < depth {
                let Some(name) = here.file_name().map(|n| n.to_string_lossy().into_owned()) else {
                    break;
                };
                let Some(up) = here.parent().map(|u| u.to_path_buf()) else {
                    break;
                };
                rel.insert(0, name);
                let up_loc = Location::Folder(up.clone());
                if up_loc.attributes().is_err() {
                    break; // not a Zarr group: the store ends here
                }
                out.push((up_loc, rel.join("/")));
                here = up;
            }
        }
        Location::Zip { zip, inner } => {
            let parts: Vec<&str> = inner.split('/').filter(|s| !s.is_empty()).collect();
            for k in (0..parts.len()).rev() {
                if out.len() >= depth {
                    break;
                }
                out.push((
                    Location::Zip {
                        zip: zip.clone(),
                        inner: parts[..k].join("/"),
                    },
                    parts[k..].join("/"),
                ));
            }
        }
    }
    out
}

/// The node at the relative path `rel` ("..", "../..") from `loc`, if it stays in the store.
fn resolve(loc: &Location, rel: &str) -> Option<Location> {
    let mut parts: Vec<String> = match loc {
        Location::Folder(_) => Vec::new(),
        Location::Zip { inner, .. } => inner
            .split('/')
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect(),
    };
    let mut folder = match loc {
        Location::Folder(p) => Some(p.clone()),
        Location::Zip { .. } => None,
    };
    for part in rel.split('/').filter(|s| !s.is_empty() && *s != ".") {
        if part == ".." {
            match &mut folder {
                Some(p) => {
                    if !p.pop() {
                        return None;
                    }
                }
                None => {
                    parts.pop()?;
                }
            }
        } else {
            match &mut folder {
                Some(p) => p.push(part),
                None => parts.push(part.to_string()),
            }
        }
    }
    Some(match (loc, folder) {
        (_, Some(p)) => Location::Folder(p),
        (Location::Zip { zip, .. }, None) => Location::Zip {
            zip: zip.clone(),
            inner: parts.join("/"),
        },
        _ => return None,
    })
}

/// A 0.6 scene as read from the group that holds it: its systems' axis names and its
/// transformations.
#[derive(Debug, Clone)]
pub struct Scene {
    systems: Vec<(String, Vec<String>)>,
    edges: Vec<Transformation>,
}

impl Scene {
    /// The scene of the group with attributes `attrs`, if it has one (`ome.scene`).
    pub fn from_attributes(attrs: &Map<String, Value>) -> Option<Scene> {
        let scene = ome(attrs, "scene")?;
        let systems = scene
            .get("coordinateSystems")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|s| {
                        let name = s.get("name")?.as_str()?.to_string();
                        let axes = s
                            .get("axes")
                            .and_then(Value::as_array)
                            .map(|ax| {
                                ax.iter()
                                    .filter_map(|a| a.get("name").and_then(Value::as_str))
                                    .map(str::to_string)
                                    .collect()
                            })
                            .unwrap_or_default();
                        Some((name, axes))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let edges = scene
            .get("coordinateTransformations")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|t| Transformation::from_json(t).ok())
                    .collect()
            })
            .unwrap_or_default();
        Some(Scene { systems, edges })
    }

    /// The scene of the group at `loc`, if it has one.
    pub fn read(loc: &Location) -> Option<Scene> {
        Self::from_attributes(&loc.attributes().ok()?)
    }

    /// The placement of the image at `rel` below the scene's group, whose own system is
    /// `system` over axes `axes`: its edge into a shared system (`prefer` when it has one
    /// into that system), as a matrix over `axes`. None when the scene does not place it,
    /// or the edge is not over the image's axes.
    pub fn placement(
        &self,
        rel: &str,
        system: &str,
        axes: &[String],
        prefer: Option<&str>,
    ) -> Option<Matrix> {
        let rel = clean(rel);
        let mine: Vec<&Transformation> = self
            .edges
            .iter()
            .filter(|t| {
                end_name(t.input.as_ref()) == Some(system)
                    && end_path(t.input.as_ref()).as_deref() == Some(rel.as_str())
                    && end_path(t.output.as_ref()).is_none()
            })
            .collect();
        let edge = prefer
            .and_then(|p| {
                mine.iter()
                    .find(|t| end_name(t.output.as_ref()) == Some(p))
                    .copied()
            })
            .or_else(|| mine.first().copied())?;
        let n = axes.len();
        let out = end_name(edge.output.as_ref())?;
        let out_axes = self
            .systems
            .iter()
            .find(|(name, _)| name == out)
            .map(|(_, a)| a.clone())
            .unwrap_or_default();
        if !out_axes.is_empty() && out_axes != axes {
            // over other axes, or the same in another order: put them in the image's
            if out_axes.len() != n || !axes.iter().all(|a| out_axes.contains(a)) {
                return None;
            }
            let m = edge.matrix(n).ok()?;
            let from: Vec<&str> = out_axes.iter().map(String::as_str).collect();
            let to: Vec<&str> = axes.iter().map(String::as_str).collect();
            return m.embed(&from, &to).ok();
        }
        edge.matrix(n).ok()
    }
}

/// The image's placement among its neighbours (see the module's notes), over its own
/// axes, or None when its levels place it already (0.4 / 0.5, an image of its own).
pub fn placement(loc: &Location, image: &Image) -> Option<Matrix> {
    let attrs = loc.attributes().ok()?;
    // a label image sits where its image sits
    if ome(&attrs, "image-label").is_some() {
        let ancestors = ancestors(loc, 2);
        if let Some((parent, rel)) = ancestors.get(1) {
            if rel.starts_with("labels/") {
                let parent_image = Image::at(parent.clone()).ok()?;
                let p = placement(parent, &parent_image)?;
                let from: Vec<String> = parent_image
                    .multiscales
                    .first()?
                    .axes
                    .iter()
                    .map(|a| a.name.clone())
                    .collect();
                let to: Vec<String> = image
                    .multiscales
                    .first()?
                    .axes
                    .iter()
                    .map(|a| a.name.clone())
                    .collect();
                if from == to {
                    return Some(p);
                }
                let f: Vec<&str> = from.iter().map(String::as_str).collect();
                let t: Vec<&str> = to.iter().map(String::as_str).collect();
                return p.embed(&f, &t).ok();
            }
        }
        return None;
    }
    let ms = image.multiscales.first()?;
    let system = ms.system.as_deref()?; // 0.4 / 0.5: the levels place it
    let axes: Vec<String> = ms.axes.iter().map(|a| a.name.clone()).collect();
    let prefer = attrs.get("eubi:coordinate_system").and_then(Value::as_str);
    // ozp's mark: the group holding the scene
    if let Some(up) = attrs.get("eubi:scene").and_then(Value::as_str) {
        if let Some(owner) = resolve(loc, up) {
            let rel = ancestors(loc, 8)
                .into_iter()
                .find(|(a, _)| *a == owner)
                .map(|(_, r)| r);
            if let (Some(rel), Some(scene)) = (rel, Scene::read(&owner)) {
                if let Some(m) = scene.placement(&rel, system, &axes, prefer) {
                    return Some(m);
                }
            }
        }
    }
    for (owner, rel) in ancestors(loc, 4) {
        if let Some(scene) = Scene::read(&owner) {
            if let Some(m) = scene.placement(&rel, system, &axes, prefer) {
                return Some(m);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_scene_places_an_image_by_its_path() {
        let attrs = json!({"ome": {"version": "0.6", "scene": {
        "coordinateSystems": [{"name": "well", "axes": [{"name": "y"}, {"name": "x"}]}],
        "coordinateTransformations": [
            {"type": "translation", "translation": [10.0, -5.0],
             "input": {"name": "physical", "path": "0"}, "output": {"name": "well"}},
            {"type": "translation", "translation": [20.0, 7.0],
             "input": {"name": "physical", "path": "./1/"}, "output": {"name": "well"}}
        ]}}});
        let scene = Scene::from_attributes(attrs.as_object().unwrap()).unwrap();
        let axes = vec!["y".to_string(), "x".to_string()];
        let m = scene.placement("1", "physical", &axes, None).unwrap();
        assert_eq!(m.apply(&[0.0, 0.0]), vec![20.0, 7.0]);
        assert!(scene.placement("2", "physical", &axes, None).is_none());
        assert!(scene.placement("0", "other", &axes, None).is_none());
        // the shared system's axes in another order: put in the image's
        let swapped = vec!["x".to_string(), "y".to_string()];
        let m = scene.placement("0", "physical", &swapped, None).unwrap();
        assert_eq!(m.apply(&[0.0, 0.0]), vec![-5.0, 10.0]);
    }

    #[test]
    fn relative_paths_resolve_in_folders_and_zips() {
        let z = Location::Zip {
            zip: "p.ozx".into(),
            inner: "A/1/0".into(),
        };
        assert_eq!(
            resolve(&z, ".."),
            Some(Location::Zip {
                zip: "p.ozx".into(),
                inner: "A/1".into()
            })
        );
        assert_eq!(resolve(&z, "../../../.."), None);
        let a = ancestors(&z, 8);
        assert_eq!(a[0].1, "0");
        assert_eq!(a[1].1, "1/0");
        assert_eq!(a.len(), 3);
    }
}
