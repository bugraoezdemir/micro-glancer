//! The GPU side: device, the target (a window's surface or an off-screen texture), and
//! per layer the tile atlas (a texture array in the image's own texel format), the
//! buffers the shader reads (page table, levels, channels) and the channels' colour maps.
//!
//! A volume layer also has a brick atlas (bricks of its voxels for planes at any angle,
//! see [`crate::bricks`]), drawn by the volume pipeline of its texel type.
//!
//! A frame clears to nothing, adds the layers up one by one (each drawn by the pipeline
//! of its texel type, colours and coverage added), then fills the background where no
//! layer drew.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use crate::bricks;
use crate::source::Texel;
use crate::tiles::{SLOTS_PER_LAYER, SLOTS_PER_ROW, TILE};
use crate::ViewerError;

/// The shader's per-frame parameters for one layer (uniform, 64 bytes). `outside` is
/// what the layer draws where it has nothing: the background when it is the only,
/// opaque layer (the frame then needs no background pass), else nothing.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct FrameParams {
    pub n_levels: u32,
    pub n_channels: u32,
    pub level: u32,
    pub channel_stride: u32,
    pub missing: [f32; 4],
    pub outside: [f32; 4],
    pub opacity: f32,
    /// Volumes: 1 samples trilinearly between voxel centres, 0 takes the nearest voxel.
    pub linear: u32,
    /// Projections: 0 maximum intensity, 1 volume, 2 isosurfaces; + 4 shaded.
    pub mode: u32,
    /// Volume: the opacity per voxel at the top of the contrast window.
    pub density: f32,
    /// Label images: the selected id's low and high 32 bits, the colour seed.
    pub extra: [u32; 4],
}

/// One level's placement for a volume's plane (storage, 112 bytes): the voxel at screen
/// position p is origin + p.x * ux + p.y * uy (voxel centres at integers).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct LevelGpu3 {
    pub origin_int: [i32; 4],
    pub origin_frac: [f32; 4],
    pub ux: [f32; 4],
    pub uy: [f32; 4],
    /// Voxels along x, y, z, and the level's first page entry.
    pub size: [i32; 4],
    /// Bricks along x, y, z, and 1 when the level has a page table (0: never drawn).
    pub grid: [u32; 4],
    /// Projections: the voxel index change per world unit along the view's normal, and
    /// the camera's scale (w).
    pub dir: [f32; 4],
}

/// What a frame draws for one layer.
#[derive(Clone, Copy, Debug)]
pub enum Draw {
    /// From the tile atlas (a plane of an image, as the image's levels lie).
    Flat(u16, FrameParams),
    /// From the brick atlas (a volume cut by a plane at any angle).
    Volume(u16, FrameParams),
    /// From the brick atlas: the volume projected along the view (maximum intensity).
    Mip(u16, FrameParams),
    /// A label image from the tile atlas (blended over what lies beneath).
    Label(u16, FrameParams),
    /// A label volume cut by a plane, from the brick atlas.
    LabelVolume(u16, FrameParams),
}

/// A pipeline and its bind group layout.
type Built = (wgpu::RenderPipeline, wgpu::BindGroupLayout);

/// Pipeline kinds.
const KIND_FLAT: u8 = 0;
const KIND_BRICKS: u8 = 1;
const KIND_MIP: u8 = 2;
const KIND_LABEL: u8 = 3;
const KIND_LABEL_BRICKS: u8 = 4;

impl Draw {
    fn params(&self) -> &FrameParams {
        match self {
            Draw::Flat(_, p)
            | Draw::Volume(_, p)
            | Draw::Mip(_, p)
            | Draw::Label(_, p)
            | Draw::LabelVolume(_, p) => p,
        }
    }

    /// The layer and the pipeline kind.
    fn target(&self) -> (u16, u8) {
        match *self {
            Draw::Flat(id, _) => (id, KIND_FLAT),
            Draw::Volume(id, _) => (id, KIND_BRICKS),
            Draw::Mip(id, _) => (id, KIND_MIP),
            Draw::Label(id, _) => (id, KIND_LABEL),
            Draw::LabelVolume(id, _) => (id, KIND_LABEL_BRICKS),
        }
    }
}

/// A frame being drawn: the target's view (and the window's texture, to present).
pub struct Frame {
    view: wgpu::TextureView,
    surface: Option<wgpu::SurfaceTexture>,
}

/// The colour between views and of an empty view's surroundings.
pub const GAP: [f32; 4] = [0.03, 0.03, 0.035, 1.0];

/// One level's placement on screen for the frame (storage, 48 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct LevelGpu {
    pub origin_int: [i32; 2],
    pub origin_frac: [f32; 2],
    pub ux: [f32; 2],
    pub uy: [f32; 2],
    pub size: [i32; 2],
    pub grid_w: u32,
    pub page_base: u32,
}

/// One channel's display (storage, 32 bytes): its colour, or (`lut` 1) its colour map,
/// row `channel` of the layer's colour map texture.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct ChannelGpu {
    pub color: [f32; 3],
    pub gamma: f32,
    pub lo: f32,
    pub hi: f32,
    pub active: u32,
    pub lut: u32,
    /// Isosurfaces: the level, when `has_iso` (else the middle of the contrast window).
    pub iso: f32,
    pub has_iso: u32,
    pub _pad: [u32; 2],
}

/// Entries of a colour map row.
pub const LUT_SIZE: u32 = 256;

pub const BACKGROUND: [f32; 4] = [0.08, 0.08, 0.08, 1.0];
pub const MISSING: [f32; 4] = [0.16, 0.16, 0.18, 1.0];

/// Where frames go.
enum Target {
    /// A window's surface: presented on screen.
    Surface {
        surface: wgpu::Surface<'static>,
        config: wgpu::SurfaceConfiguration,
    },
    /// A texture read back as RGBA8 (Wayland, remote desktops, tests).
    Offscreen {
        texture: wgpu::Texture,
        width: u32,
        height: u32,
    },
}

/// How a viewer gets its GPU.
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuOptions {
    /// Use the system's software adapter (WARP, lavapipe) instead of a GPU.
    pub software: bool,
    /// Present without waiting for the display's refresh (for measuring frame times).
    pub no_vsync: bool,
    /// The most GPU memory the tile atlas may take, in MiB (0: 1024).
    pub atlas_mib: u32,
}

/// A layer's atlas and buffers.
struct Plane {
    texel: Texel,
    params: wgpu::Buffer,
    atlas: wgpu::Texture,
    slots: u32,
    pages: wgpu::Buffer,
    levels: wgpu::Buffer,
    channels: wgpu::Buffer,
    /// LUT_SIZE x channels, RGBA8: row c is channel c's colour map.
    luts: wgpu::Texture,
    bind: Option<wgpu::BindGroup>,
    /// A volume's bricks, once a plane at an angle needed them.
    bricks: Option<BrickAtlas>,
}

/// A volume's brick atlas and the buffers its pipeline reads (channels and colour maps
/// are the layer's).
struct BrickAtlas {
    params: wgpu::Buffer,
    atlas: wgpu::Texture,
    slots: u32,
    pages: wgpu::Buffer,
    levels: wgpu::Buffer,
    /// Per slot: the lowest and highest value its brick holds (to skip empty bricks).
    ranges: wgpu::Buffer,
    bind: Option<wgpu::BindGroup>,
}

pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    target: Target,
    format: wgpu::TextureFormat,
    /// Per layer id.
    planes: HashMap<u16, Plane>,
    /// Per texel type and kind (tiles, bricks, projection): the pipeline and its bind
    /// group layout.
    pipelines: HashMap<(Texel, u8), (wgpu::RenderPipeline, wgpu::BindGroupLayout)>,
    /// Pipelines being built on threads (see [`Gpu::warm`]).
    warming: HashMap<(Texel, u8), std::thread::JoinHandle<Result<Built, String>>>,
    /// Fills the background where no layer drew.
    background: wgpu::RenderPipeline,
    /// Fills what no view covered.
    gaps: wgpu::RenderPipeline,
    /// The crosshair of linked views, its centre and bind group.
    cross: wgpu::RenderPipeline,
    cross_params: wgpu::Buffer,
    cross_bind: wgpu::BindGroup,
    /// A view drawn smaller (the projection) is drawn here, then scaled into its place.
    scratch: Option<(wgpu::Texture, u32, u32)>,
    blit: wgpu::RenderPipeline,
    blit_layout: wgpu::BindGroupLayout,
    blit_params: wgpu::Buffer,
    sampler: wgpu::Sampler,
    /// The off-screen frames' readback buffer, kept between frames.
    readback: Option<wgpu::Buffer>,
    atlas_mib: u32,
    /// The first GPU error no error scope caught.
    errors: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    pub adapter_info: String,
}

fn err(e: impl std::fmt::Display) -> ViewerError {
    ViewerError(e.to_string())
}

fn instance_for(backends: wgpu::Backends) -> wgpu::Instance {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = backends;
    wgpu::Instance::new(desc)
}

/// The backends to try, in order: `MICRO_GLANCER_BACKEND` ("vulkan", "dx12", "metal",
/// "gl") picks one; else Vulkan first on Windows (DX12's shader compiler took 7 to 11 s
/// for the projection shader, uncached, freezing the window), then the platform's own.
fn backend_choices() -> Vec<wgpu::Backends> {
    use wgpu::Backends as B;
    match std::env::var("MICRO_GLANCER_BACKEND")
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "vulkan" => vec![B::VULKAN],
        "dx12" => vec![B::DX12],
        "metal" => vec![B::METAL],
        "gl" => vec![B::GL],
        _ if cfg!(windows) => vec![B::VULKAN, B::PRIMARY],
        _ => vec![B::PRIMARY],
    }
}

fn texel_format(t: Texel) -> (wgpu::TextureFormat, wgpu::TextureSampleType, &'static str) {
    use wgpu::{TextureFormat as F, TextureSampleType as S};
    match t {
        Texel::U8 => (F::R8Uint, S::Uint, "u32"),
        Texel::I8 => (F::R8Sint, S::Sint, "i32"),
        Texel::U16 => (F::R16Uint, S::Uint, "u32"),
        Texel::I16 => (F::R16Sint, S::Sint, "i32"),
        Texel::F32 => (F::R32Float, S::Float { filterable: false }, "f32"),
        Texel::L32 => (F::R32Uint, S::Uint, "u32"),
        Texel::L64 => (F::Rg32Uint, S::Uint, "u32"),
    }
}

fn storage(device: &wgpu::Device, label: &str, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(64),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

impl Gpu {
    /// A GPU drawing into the native window `window` (of `display`), `width` x `height`
    /// physical pixels.
    ///
    /// # Safety
    /// The window must outlive the viewer (the Qt widget owns both).
    pub unsafe fn for_window(
        display: RawDisplayHandle,
        window: RawWindowHandle,
        width: u32,
        height: u32,
        options: GpuOptions,
    ) -> Result<Self, ViewerError> {
        let mut last = None;
        let (surface, adapter) = 'found: {
            for backends in backend_choices() {
                let instance = instance_for(backends);
                let surface =
                    match instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                        raw_display_handle: Some(display),
                        raw_window_handle: window,
                    }) {
                        Ok(s) => s,
                        Err(e) => {
                            last = Some(err(e));
                            continue;
                        }
                    };
                let options = wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: options.software,
                    compatible_surface: Some(&surface),
                    apply_limit_buckets: false,
                };
                match pollster::block_on(instance.request_adapter(&options)) {
                    Ok(adapter) => break 'found (surface, adapter),
                    Err(e) => last = Some(err(e)),
                }
            }
            return Err(last.unwrap_or_else(|| err("no GPU adapter for the window")));
        };
        let (device, queue) = request_device(&adapter)?;
        let caps = surface.get_capabilities(&adapter);
        // a plain (non-sRGB) format: the shader's values are written as they are
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let present_mode = if options.no_vsync {
            [wgpu::PresentMode::Immediate, wgpu::PresentMode::Mailbox]
                .into_iter()
                .find(|m| caps.present_modes.contains(m))
                .unwrap_or(wgpu::PresentMode::Fifo)
        } else {
            wgpu::PresentMode::Fifo
        };
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: width.max(1),
            height: height.max(1),
            present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        // before the error recorder is installed, wgpu's default handler panics: a
        // window that cannot be drawn into (Qt's off-screen platform) is an error here
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        surface.configure(&device, &config);
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(err(format!(
                "the window cannot be drawn into ({e}); draw off-screen instead"
            )));
        }
        let info = adapter_label(&adapter);
        Ok(Self::with(
            device,
            queue,
            Target::Surface { surface, config },
            format,
            info,
            options,
        ))
    }

    /// A GPU drawing into an off-screen texture of `width` x `height` pixels.
    pub fn offscreen(width: u32, height: u32, options: GpuOptions) -> Result<Self, ViewerError> {
        let mut last = None;
        let adapter = 'found: {
            for backends in backend_choices() {
                let instance = instance_for(backends);
                let options = wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: options.software,
                    compatible_surface: None,
                    apply_limit_buckets: false,
                };
                match pollster::block_on(instance.request_adapter(&options)) {
                    Ok(adapter) => break 'found adapter,
                    Err(e) => last = Some(err(e)),
                }
            }
            return Err(last.unwrap_or_else(|| err("no GPU adapter")));
        };
        let (device, queue) = request_device(&adapter)?;
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let texture = offscreen_texture(&device, format, width, height);
        let info = adapter_label(&adapter);
        Ok(Self::with(
            device,
            queue,
            Target::Offscreen {
                texture,
                width: width.max(1),
                height: height.max(1),
            },
            format,
            info,
            options,
        ))
    }

    fn with(
        device: wgpu::Device,
        queue: wgpu::Queue,
        target: Target,
        format: wgpu::TextureFormat,
        adapter_info: String,
        options: GpuOptions,
    ) -> Self {
        // errors outside an error scope are recorded, never a panic across the bindings
        let errors = std::sync::Arc::new(std::sync::Mutex::new(None::<String>));
        let sink = errors.clone();
        device.on_uncaptured_error(std::sync::Arc::new(move |e: wgpu::Error| {
            if let Ok(mut slot) = sink.lock() {
                slot.get_or_insert_with(|| e.to_string());
            }
        }));
        let background = background_pipeline(&device, format, BACKGROUND);
        let gaps = background_pipeline(&device, format, GAP);
        let (cross, cross_params, cross_bind) = cross_pipeline(&device, format);
        let (blit, blit_layout, blit_params) = blit_pipeline(&device, format);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("colour maps"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            errors,
            device,
            queue,
            target,
            format,
            planes: HashMap::new(),
            warming: HashMap::new(),
            pipelines: HashMap::new(),
            background,
            gaps,
            cross,
            cross_params,
            cross_bind,
            scratch: None,
            blit,
            blit_layout,
            blit_params,
            sampler,
            readback: None,
            atlas_mib: if options.atlas_mib == 0 {
                1024
            } else {
                options.atlas_mib
            },
            adapter_info,
        }
    }

    /// The target's size in physical pixels.
    pub fn size(&self) -> (u32, u32) {
        match &self.target {
            Target::Surface { config, .. } => (config.width, config.height),
            Target::Offscreen { width, height, .. } => (*width, *height),
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let (w, h) = (width.max(1), height.max(1));
        match &mut self.target {
            Target::Surface { surface, config } => {
                config.width = w;
                config.height = h;
                surface.configure(&self.device, config);
            }
            Target::Offscreen {
                texture,
                width,
                height,
            } => {
                *texture = offscreen_texture(&self.device, self.format, w, h);
                *width = w;
                *height = h;
            }
        }
    }

    /// The atlas slots a layer can have: `wanted`, within `budget_mib` and the device's
    /// limits (whole atlas layers of 256 slots).
    pub fn slots_for(&self, texel: Texel, wanted: u32, budget_mib: u32) -> u32 {
        let layer_bytes = (SLOTS_PER_LAYER * TILE * TILE) as u64 * texel.bytes() as u64;
        let budget = (budget_mib as u64 * 1024 * 1024 / layer_bytes).max(1) as u32;
        let limit = self.device.limits().max_texture_array_layers.max(1);
        wanted.div_ceil(SLOTS_PER_LAYER).clamp(1, budget.min(limit)) * SLOTS_PER_LAYER
    }

    /// The tile atlas's memory budget in MiB, for all layers together.
    pub fn atlas_mib(&self) -> u32 {
        self.atlas_mib
    }

    /// Set up layer `id` for `texel`s wanting `slots` atlas slots within `budget_mib`:
    /// the pipeline of its texel type (made once) and a new atlas (the old one's tiles
    /// are gone). Returns the slots made (fewer when memory is short: halved until the
    /// GPU can hold them).
    pub fn prepare_layer(
        &mut self,
        id: u16,
        texel: Texel,
        slots: u32,
        budget_mib: u32,
    ) -> Result<u32, ViewerError> {
        self.ensure_pipeline(texel, KIND_FLAT)?;
        // the old atlas goes first, so its memory is free for the new one
        let old = self.planes.remove(&id).filter(|p| p.texel == texel);
        let mut layers = self.slots_for(texel, slots, budget_mib) / SLOTS_PER_LAYER;
        let (format, _, _) = texel_format(texel);
        let atlas = loop {
            let scope = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
            let atlas = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("atlas"),
                size: wgpu::Extent3d {
                    width: TILE * SLOTS_PER_ROW,
                    height: TILE * SLOTS_PER_ROW,
                    depth_or_array_layers: layers,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            match pollster::block_on(scope.pop()) {
                None => break atlas,
                Some(e) if layers == 1 => {
                    return Err(err(format!("no GPU memory for the tile atlas: {e}")))
                }
                Some(_) => layers = layers.div_ceil(2),
            }
        };
        let (params, pages, levels, channels, luts, mut bricks) = match old {
            Some(p) => (p.params, p.pages, p.levels, p.channels, p.luts, p.bricks),
            None => (
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("params"),
                    size: std::mem::size_of::<FrameParams>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                storage(&self.device, "pages", 4),
                storage(&self.device, "levels", 48),
                storage(&self.device, "channels", 48),
                lut_texture(&self.device, 1),
                None,
            ),
        };
        if let Some(b) = &mut bricks {
            b.bind = None;
        }
        self.planes.insert(
            id,
            Plane {
                texel,
                params,
                atlas,
                slots: layers * SLOTS_PER_LAYER,
                pages,
                levels,
                channels,
                luts,
                bind: None,
                bricks,
            },
        );
        Ok(layers * SLOTS_PER_LAYER)
    }

    /// The pipeline for `texel`s of one kind, made once.
    fn ensure_pipeline(&mut self, texel: Texel, kind: u8) -> Result<(), ViewerError> {
        // built on a thread already (or still being built: wait for it)
        if let Some(handle) = self.warming.remove(&(texel, kind)) {
            match handle.join() {
                Ok(Ok(made)) => {
                    self.pipelines.insert((texel, kind), made);
                }
                Ok(Err(e)) => {
                    return Err(err(format!("the GPU refused the drawing pipeline: {e}")))
                }
                Err(_) => {} // the thread failed: build it here
            }
        }
        if !self.pipelines.contains_key(&(texel, kind)) {
            let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let made = Self::build_pipeline(&self.device, self.format, texel, kind);
            if let Some(e) = pollster::block_on(scope.pop()) {
                return Err(err(format!("the GPU refused the drawing pipeline: {e}")));
            }
            self.pipelines.insert((texel, kind), made);
        }
        Ok(())
    }

    /// Start building the volume pipelines (planes from bricks, projections) for
    /// `texel`s on threads, so the first view that needs them does not wait for the
    /// shader compiler.
    pub fn warm(&mut self, texel: Texel, label: bool) {
        let kinds: &[u8] = if label {
            &[KIND_LABEL_BRICKS]
        } else {
            &[KIND_BRICKS, KIND_MIP]
        };
        for &kind in kinds {
            let key = (texel, kind);
            if self.pipelines.contains_key(&key) || self.warming.contains_key(&key) {
                continue;
            }
            let device = self.device.clone();
            let format = self.format;
            let spawned = std::thread::Builder::new()
                .name("micro-glancer pipelines".into())
                .spawn(move || {
                    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
                    let made = Gpu::build_pipeline(&device, format, texel, kind);
                    match pollster::block_on(scope.pop()) {
                        Some(e) => Err(e.to_string()),
                        None => Ok(made),
                    }
                });
            if let Ok(handle) = spawned {
                self.warming.insert(key, handle);
            }
        }
    }

    /// The brick slots a volume can have: `wanted`, within `budget_mib` and the device's
    /// limits (whole atlas layers of bricks::PER_LAYER bricks).
    pub fn brick_slots_for(&self, texel: Texel, wanted: u32, budget_mib: u32) -> u32 {
        let layer_bytes = 4096u64 * 4096 * texel.bytes() as u64;
        let budget = (budget_mib as u64 * 1024 * 1024 / layer_bytes).max(1) as u32;
        let limit = self.device.limits().max_texture_array_layers.max(1);
        wanted
            .div_ceil(bricks::PER_LAYER)
            .clamp(1, budget.min(limit))
            * bricks::PER_LAYER
    }

    /// Set up layer `id`'s brick atlas for `slots` bricks within `budget_mib` (a new
    /// atlas: the old one's bricks are gone). Returns the slots made.
    pub fn prepare_bricks(
        &mut self,
        id: u16,
        slots: u32,
        budget_mib: u32,
    ) -> Result<u32, ViewerError> {
        let texel = self
            .planes
            .get(&id)
            .ok_or_else(|| err("no such layer"))?
            .texel;

        if let Some(p) = self.planes.get_mut(&id) {
            p.bricks = None; // its memory first
        }
        let mut layers = self.brick_slots_for(texel, slots, budget_mib) / bricks::PER_LAYER;
        let (format, _, _) = texel_format(texel);
        let atlas = loop {
            let scope = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
            let atlas = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("bricks"),
                size: wgpu::Extent3d {
                    width: 4096,
                    height: 4096,
                    depth_or_array_layers: layers,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            match pollster::block_on(scope.pop()) {
                None => break atlas,
                Some(e) if layers == 1 => {
                    return Err(err(format!("no GPU memory for the brick atlas: {e}")))
                }
                Some(_) => layers = layers.div_ceil(2),
            }
        };
        let made = BrickAtlas {
            params: self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("brick params"),
                size: std::mem::size_of::<FrameParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            atlas,
            slots: layers * bricks::PER_LAYER,
            pages: storage(&self.device, "brick pages", 4),
            levels: storage(&self.device, "brick levels", 112),
            ranges: storage(
                &self.device,
                "brick ranges",
                layers as u64 * bricks::PER_LAYER as u64 * 8,
            ),
            bind: None,
        };
        let slots = made.slots;
        if let Some(p) = self.planes.get_mut(&id) {
            p.bricks = Some(made);
        }
        Ok(slots)
    }

    /// Brick slots of layer `id` (0 without a brick atlas).
    pub fn brick_slots(&self, id: u16) -> u32 {
        self.planes
            .get(&id)
            .and_then(|p| p.bricks.as_ref())
            .map_or(0, |b| b.slots)
    }

    /// A brick's strip (bricks::STRIP_W x bricks::BRICK texels) into brick slot `slot`
    /// of layer `id`.
    pub fn upload_brick(&self, id: u16, slot: u32, data: &[u8]) {
        let Some(p) = self.planes.get(&id) else {
            return;
        };
        let Some(b) = &p.bricks else {
            return;
        };
        let (layer, x, y) = bricks::place(slot);
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &b.atlas,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: layer },
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bricks::STRIP_W * p.texel.bytes() as u32),
                rows_per_image: Some(bricks::BRICK),
            },
            wgpu::Extent3d {
                width: bricks::STRIP_W,
                height: bricks::BRICK,
                depth_or_array_layers: 1,
            },
        );
    }

    /// The lowest and highest value of the brick in layer `id`'s slot `slot`.
    pub fn write_brick_range(&self, id: u16, slot: u32, range: (f32, f32)) {
        let Some(b) = self.planes.get(&id).and_then(|p| p.bricks.as_ref()) else {
            return;
        };
        let data = [range.0, range.1];
        self.queue
            .write_buffer(&b.ranges, slot as u64 * 8, bytemuck::cast_slice(&data));
    }

    /// Layer `id`'s brick page table (as [`Gpu::write_pages`]).
    pub fn write_brick_pages(&mut self, id: u16, len: usize, offset: usize, data: &[u32]) -> bool {
        let Some(b) = self.planes.get_mut(&id).and_then(|p| p.bricks.as_mut()) else {
            return false;
        };
        let need = (len.max(1) * 4) as u64;
        let mut grown = false;
        if b.pages.size() < need {
            b.pages = storage(&self.device, "brick pages", need.next_power_of_two());
            b.bind = None;
            grown = true;
        }
        if !data.is_empty() {
            self.queue
                .write_buffer(&b.pages, (offset * 4) as u64, bytemuck::cast_slice(data));
        }
        grown
    }

    pub fn write_brick_levels(&mut self, id: u16, data: &[LevelGpu3]) {
        let Some(b) = self.planes.get_mut(&id).and_then(|p| p.bricks.as_mut()) else {
            return;
        };
        let need = (data.len().max(1) * std::mem::size_of::<LevelGpu3>()) as u64;
        if b.levels.size() < need {
            b.levels = storage(&self.device, "brick levels", need.next_power_of_two());
            b.bind = None;
        }
        if !data.is_empty() {
            self.queue
                .write_buffer(&b.levels, 0, bytemuck::cast_slice(data));
        }
    }

    /// Forget layer `id` (its atlas and buffers are freed).
    pub fn remove_layer(&mut self, id: u16) {
        self.planes.remove(&id);
    }

    /// The first GPU error no error scope caught, if any (taken: reported once).
    pub fn take_error(&self) -> Option<String> {
        self.errors.lock().ok().and_then(|mut e| e.take())
    }

    /// Atlas slots of every layer together.
    pub fn all_slots(&self) -> u32 {
        self.planes.values().map(|p| p.slots).sum()
    }

    fn build_pipeline(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        texel: Texel,
        kind: u8,
    ) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
        let (_, sample_type, wgsl_type) = texel_format(texel);
        // the brick shader's bindings and sampling (all before its fragment entry), for
        // the kinds with fragment entries of their own
        let bricks = include_str!("shader_bricks.wgsl");
        let shared = bricks.find("@fragment").map_or(bricks, |at| &bricks[..at]);
        let code = match kind {
            KIND_FLAT => include_str!("shader.wgsl").to_string(),
            KIND_BRICKS => bricks.to_string(),
            KIND_LABEL => {
                include_str!("shader.wgsl").to_string() + include_str!("shader_label.wgsl")
            }
            KIND_LABEL_BRICKS => shared.to_string() + include_str!("shader_label_bricks.wgsl"),
            _ => shared.to_string() + include_str!("shader_mip.wgsl"),
        };
        let entry = if kind == KIND_LABEL || kind == KIND_LABEL_BRICKS {
            "fs_label"
        } else {
            "fs"
        };
        let source = code.replace("TEXEL", wgsl_type);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("plane"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let buffer = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let read_only = wgpu::BufferBindingType::Storage { read_only: true };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("plane"),
            entries: &[
                buffer(0, wgpu::BufferBindingType::Uniform),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                buffer(2, read_only),
                buffer(3, read_only),
                buffer(4, read_only),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        // bricks: their value ranges too (binding 7)
        let layout = if kind == KIND_FLAT || kind == KIND_LABEL {
            layout
        } else {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bricks"),
                entries: &[
                    buffer(0, wgpu::BufferBindingType::Uniform),
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type,
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    buffer(2, read_only),
                    buffer(3, read_only),
                    buffer(4, read_only),
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    buffer(7, read_only),
                ],
            })
        };
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("plane"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        // layers add up: colours and coverage (alpha) both
        let add = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        // label images lie over what is beneath (premultiplied)
        let over = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        };
        let blend = if kind == KIND_LABEL || kind == KIND_LABEL_BRICKS {
            over
        } else {
            add
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("plane"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(entry),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState {
                        color: blend,
                        alpha: blend,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        (pipeline, layout)
    }

    /// Tile data (`w` x `h` texels, row-major, the layer's texel type) into atlas slot
    /// `slot` of layer `id`.
    pub fn upload_tile(&self, id: u16, slot: u32, data: &[u8], w: u32, h: u32) {
        let Some(p) = self.planes.get(&id) else {
            return;
        };
        let within = slot % SLOTS_PER_LAYER;
        let (sx, sy) = (
            (within % SLOTS_PER_ROW) * TILE,
            (within / SLOTS_PER_ROW) * TILE,
        );
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &p.atlas,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: sx,
                    y: sy,
                    z: slot / SLOTS_PER_LAYER,
                },
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * p.texel.bytes() as u32),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Write `data` into a storage buffer at element `offset`, making the buffer larger
    /// (and its contents lost: the caller writes everything then) when `len` elements do
    /// not fit. Returns whether it was made again.
    fn write_storage<T: Pod>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        p: &mut Plane,
        which: u8,
        len: usize,
        offset: usize,
        data: &[T],
    ) -> bool {
        let size = std::mem::size_of::<T>();
        let buf = match which {
            0 => &mut p.pages,
            1 => &mut p.levels,
            _ => &mut p.channels,
        };
        let need = (len.max(1) * size) as u64;
        let mut grown = false;
        if buf.size() < need {
            *buf = storage(device, "storage", need.next_power_of_two());
            p.bind = None;
            if let Some(b) = &mut p.bricks {
                b.bind = None;
            }
            grown = true;
        }
        if !data.is_empty() {
            queue.write_buffer(buf, (offset * size) as u64, bytemuck::cast_slice(data));
        }
        grown
    }

    /// Layer `id`'s page table: `len` entries in all, `data` written from entry
    /// `offset`. Returns whether the buffer was made again (then write every entry).
    pub fn write_pages(&mut self, id: u16, len: usize, offset: usize, data: &[u32]) -> bool {
        let Some(p) = self.planes.get_mut(&id) else {
            return false;
        };
        Self::write_storage(&self.device, &self.queue, p, 0, len, offset, data)
    }

    pub fn write_levels(&mut self, id: u16, data: &[LevelGpu]) {
        if let Some(p) = self.planes.get_mut(&id) {
            Self::write_storage(&self.device, &self.queue, p, 1, data.len(), 0, data);
        }
    }

    pub fn write_channels(&mut self, id: u16, data: &[ChannelGpu]) {
        if let Some(p) = self.planes.get_mut(&id) {
            Self::write_storage(&self.device, &self.queue, p, 2, data.len(), 0, data);
        }
    }

    /// Layer `id`'s colour maps: one row of LUT_SIZE RGBA8 entries per channel.
    pub fn write_luts(&mut self, id: u16, rows: &[[u8; 4]]) {
        let Some(p) = self.planes.get_mut(&id) else {
            return;
        };
        let n = (rows.len() as u32 / LUT_SIZE).max(1);
        if p.luts.height() != n {
            p.luts = lut_texture(&self.device, n);
            p.bind = None;
            if let Some(b) = &mut p.bricks {
                b.bind = None;
            }
        }
        if rows.is_empty() {
            return;
        }
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &p.luts,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(rows),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * LUT_SIZE),
                rows_per_image: Some(n),
            },
            wgpu::Extent3d {
                width: LUT_SIZE,
                height: n,
                depth_or_array_layers: 1,
            },
        );
    }

    fn bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        p: &mut Plane,
    ) -> wgpu::BindGroup {
        if p.bind.is_none() {
            let view = p.atlas.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            });
            let luts = p.luts.create_view(&Default::default());
            p.bind = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("plane"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: p.params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: p.pages.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: p.levels.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: p.channels.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&luts),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            }));
        }
        p.bind.clone().expect("made above")
    }

    fn brick_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        p: &mut Plane,
    ) -> Option<wgpu::BindGroup> {
        let b = p.bricks.as_mut()?;
        if b.bind.is_none() {
            let view = b.atlas.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            });
            let luts = p.luts.create_view(&Default::default());
            b.bind = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bricks"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: b.params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: b.pages.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: b.levels.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: p.channels.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&luts),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: b.ranges.as_entire_binding(),
                    },
                ],
            }));
        }
        b.bind.clone()
    }

    /// Start a frame: the target to draw into (None: the window cannot take a frame now;
    /// skip it).
    pub fn begin(&mut self) -> Result<Option<Frame>, ViewerError> {
        match &self.target {
            Target::Surface { surface, config } => match surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(t)
                | wgpu::CurrentSurfaceTexture::Suboptimal(t) => Ok(Some(Frame {
                    view: t.texture.create_view(&Default::default()),
                    surface: Some(t),
                })),
                // the window changed: configure again, draw at the next frame
                wgpu::CurrentSurfaceTexture::Outdated => {
                    surface.configure(&self.device, config);
                    Ok(None)
                }
                // hidden or busy: skip this frame
                wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                    Ok(None)
                }
                _ => Err(err("the window's surface was lost")),
            },
            Target::Offscreen { texture, .. } => Ok(Some(Frame {
                view: texture.create_view(&Default::default()),
                surface: None,
            })),
        }
    }

    /// One view of a frame: the layers in `frames` in order, then the background where
    /// none drew, within `rect` (x, y, width, height; None: the whole target, where one
    /// opaque layer draws the background itself), and the crosshair at `cross` (target
    /// pixels). The first view of a frame clears the target. Each view goes to the GPU on
    /// its own, so views may share a layer's buffers (written before each view).
    pub fn draw_view(
        &mut self,
        f: &Frame,
        frames: &[Draw],
        rect: Option<[u32; 4]>,
        first: bool,
        cross: Option<[f32; 2]>,
    ) {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("view"),
            });
        // one opaque layer draws the background itself (a projection may be translucent)
        let solo = rect.is_none()
            && frames.len() == 1
            && frames[0].params().opacity >= 1.0
            && matches!(frames[0].target().1, KIND_FLAT | KIND_BRICKS);
        let mut drawn = Vec::new();
        for d in frames {
            let mut fp = *d.params();
            if solo {
                fp.outside = BACKGROUND;
            }
            let (id, kind) = d.target();
            let Some(texel) = self.planes.get(&id).map(|p| p.texel) else {
                continue;
            };
            // made the first time a view needs it (a refusal shows as the GPU error)
            if !self.pipelines.contains_key(&(texel, kind))
                && self.ensure_pipeline(texel, kind).is_err()
            {
                continue;
            }
            let Some(p) = self.planes.get_mut(&id) else {
                continue;
            };
            let Some((pipeline, layout)) = self.pipelines.get(&(p.texel, kind)) else {
                continue;
            };
            if kind == KIND_FLAT || kind == KIND_LABEL {
                self.queue
                    .write_buffer(&p.params, 0, bytemuck::bytes_of(&fp));
                let bind = Self::bind_group(&self.device, layout, &self.sampler, p);
                drawn.push((pipeline.clone(), bind));
            } else {
                let Some(bind) = Self::brick_bind_group(&self.device, layout, &self.sampler, p)
                else {
                    continue;
                };
                if let Some(b) = &p.bricks {
                    self.queue
                        .write_buffer(&b.params, 0, bytemuck::bytes_of(&fp));
                }
                drawn.push((pipeline.clone(), bind));
            }
        }
        if let Some(c) = cross {
            let data = [c[0], c[1], 0.0, 0.0];
            self.queue
                .write_buffer(&self.cross_params, 0, bytemuck::cast_slice(&data));
        }
        let load = if first {
            let clear = if drawn.is_empty() && rect.is_none() {
                let [r, g, b, a] = BACKGROUND.map(f64::from);
                wgpu::Color { r, g, b, a }
            } else {
                wgpu::Color::TRANSPARENT
            };
            wgpu::LoadOp::Clear(clear)
        } else {
            wgpu::LoadOp::Load
        };
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("view"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &f.view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if let Some([x, y, w, h]) = rect {
                pass.set_viewport(x as f32, y as f32, w as f32, h as f32, 0.0, 1.0);
                pass.set_scissor_rect(x, y, w, h);
            }
            for (pipeline, bind) in &drawn {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bind, &[]);
                pass.draw(0..3, 0..1);
            }
            if (!drawn.is_empty() || rect.is_some()) && !solo {
                pass.set_pipeline(&self.background);
                pass.draw(0..3, 0..1);
            }
            if cross.is_some() {
                pass.set_pipeline(&self.cross);
                pass.set_bind_group(0, &self.cross_bind, &[]);
                pass.draw(0..3, 0..1);
            }
        }
        self.queue.submit([enc.finish()]);
    }

    /// A target of `w` x `h` pixels for a view drawn smaller than its place (then
    /// [`Gpu::blit`] scales it in).
    pub fn scratch_frame(&mut self, w: u32, h: u32) -> Frame {
        let fits = self
            .scratch
            .as_ref()
            .is_some_and(|(_, sw, sh)| *sw == w && *sh == h);
        if !fits {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("scratch"),
                size: wgpu::Extent3d {
                    width: w.max(1),
                    height: h.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.scratch = Some((texture, w, h));
        }
        let (texture, _, _) = self.scratch.as_ref().expect("made above");
        Frame {
            view: texture.create_view(&Default::default()),
            surface: None,
        }
    }

    /// Scale the scratch target into `rect` (x, y, width, height) of the frame, smoothly.
    pub fn blit(&mut self, f: &Frame, rect: [u32; 4], first: bool) {
        let Some((texture, _, _)) = &self.scratch else {
            return;
        };
        let view = texture.create_view(&Default::default());
        let [x, y, w, h] = rect;
        let data = [x as f32, y as f32, w as f32, h as f32];
        self.queue
            .write_buffer(&self.blit_params, 0, bytemuck::cast_slice(&data));
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit"),
            layout: &self.blit_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.blit_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("blit"),
            });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &f.view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: if first {
                            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                        } else {
                            wgpu::LoadOp::Load
                        },
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_viewport(x as f32, y as f32, w as f32, h as f32, 0.0, 1.0);
            pass.set_scissor_rect(x, y, w, h);
            pass.set_pipeline(&self.blit);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([enc.finish()]);
    }

    /// Fill what no view covered (the gaps between views) with the frame colour.
    pub fn fill_gaps(&mut self, f: &Frame) {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gaps"),
            });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gaps"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &f.view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.gaps);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([enc.finish()]);
    }

    /// End a frame: present it to the window, or read it back into `out` (RGBA8 rows,
    /// width x height x 4 bytes) through a readback buffer kept between frames.
    pub fn end(&mut self, f: Frame, out: Option<&mut [u8]>) -> Result<(), ViewerError> {
        if let Some(t) = f.surface {
            if out.is_some() {
                return Err(err("this viewer draws into a window"));
            }
            self.queue.present(t);
            return Ok(());
        }
        let Some(out) = out else {
            return Err(err("this viewer draws off-screen"));
        };
        let (texture, w, h) = match &self.target {
            Target::Offscreen {
                texture,
                width,
                height,
            } => (texture.clone(), *width, *height),
            Target::Surface { .. } => return Err(err("this viewer draws into a window")),
        };
        if out.len() != (4 * w * h) as usize {
            return Err(err(format!(
                "a frame of {w} x {h} needs {} bytes, not {}",
                4 * w * h,
                out.len()
            )));
        }
        let row = (4 * w).div_ceil(256) * 256;
        let size = (row * h) as u64;
        if self.readback.as_ref().is_none_or(|b| b.size() != size) {
            self.readback = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }));
        }
        let buf = self.readback.clone().expect("made above");
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback"),
            });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([enc.finish()]);
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(err)?;
        {
            let data = slice.get_mapped_range().map_err(err)?;
            let (row, line) = (row as usize, 4 * w as usize);
            if row == line {
                out.copy_from_slice(&data[..out.len()]);
            } else {
                for (y, dst) in out.chunks_exact_mut(line).enumerate() {
                    dst.copy_from_slice(&data[y * row..y * row + line]);
                }
            }
        }
        buf.unmap();
        Ok(())
    }

    /// One texel of layer `id`'s atlas slot `slot` at (`x`, `y`) within the tile, as
    /// stored (the layer's texel bytes), read back from the GPU.
    pub fn read_texel(&self, id: u16, slot: u32, x: u32, y: u32) -> Result<Vec<u8>, ViewerError> {
        let p = self.planes.get(&id).ok_or_else(|| err("no such layer"))?;
        let within = slot % SLOTS_PER_LAYER;
        self.read_at(
            &p.atlas,
            p.texel,
            (within % SLOTS_PER_ROW) * TILE + x,
            (within / SLOTS_PER_ROW) * TILE + y,
            slot / SLOTS_PER_LAYER,
        )
    }

    /// One voxel of layer `id`'s brick slot `slot` at (`x`, `y`, `z`) within the brick.
    pub fn read_brick_texel(
        &self,
        id: u16,
        slot: u32,
        x: u32,
        y: u32,
        z: u32,
    ) -> Result<Vec<u8>, ViewerError> {
        let p = self.planes.get(&id).ok_or_else(|| err("no such layer"))?;
        let b = p.bricks.as_ref().ok_or_else(|| err("no bricks"))?;
        let (layer, sx, sy) = bricks::place(slot);
        self.read_at(&b.atlas, p.texel, sx + z * bricks::BRICK + x, sy + y, layer)
    }

    fn read_at(
        &self,
        texture: &wgpu::Texture,
        texel: Texel,
        x: u32,
        y: u32,
        layer: u32,
    ) -> Result<Vec<u8>, ViewerError> {
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("texel"),
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("texel"),
            });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: layer },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([enc.finish()]);
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(err)?;
        let out = slice.get_mapped_range().map_err(err)?[..texel.bytes()].to_vec();
        buf.unmap();
        Ok(out)
    }

    /// Wait until the GPU has finished what was submitted (for timing).
    pub fn finish(&self) -> Result<(), ViewerError> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map(|_| ())
            .map_err(err)
    }
}

fn offscreen_texture(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn lut_texture(device: &wgpu::Device, rows: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("colour maps"),
        size: wgpu::Extent3d {
            width: LUT_SIZE,
            height: rows.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

/// The background where no layer drew: the colour weighted by what coverage is left
/// (1 - the target's alpha), alpha made 1.
fn background_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    colour: [f32; 4],
) -> wgpu::RenderPipeline {
    let [r, g, b, _] = colour;
    let source = format!(
        "@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {{
             let x = f32((i << 1u) & 2u) * 2.0 - 1.0;
             let y = f32(i & 2u) * 2.0 - 1.0;
             return vec4<f32>(x, y, 0.0, 1.0);
         }}
         @fragment fn fs() -> @location(0) vec4<f32> {{
             return vec4<f32>({r:?}, {g:?}, {b:?}, 1.0);
         }}"
    );
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("background"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let fill = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::OneMinusDstAlpha,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("background"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState {
                    color: fill,
                    alpha: fill,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// The crosshair of linked views: a line of one target pixel through the view's centre
/// across and down, laid over what is drawn.
fn cross_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> (wgpu::RenderPipeline, wgpu::Buffer, wgpu::BindGroup) {
    let source = "struct P { c: vec4<f32> }
         @group(0) @binding(0) var<uniform> p: P;
         @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
             let x = f32((i << 1u) & 2u) * 2.0 - 1.0;
             let y = f32(i & 2u) * 2.0 - 1.0;
             return vec4<f32>(x, y, 0.0, 1.0);
         }
         @fragment fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
             let d = abs(pos.xy - p.c.xy);
             if (min(d.x, d.y) > 0.5) {
                 discard;
             }
             return vec4<f32>(1.0, 0.85, 0.3, 0.5);
         }";
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("crosshair"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("crosshair"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let params = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("crosshair"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("crosshair"),
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: params.as_entire_binding(),
        }],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("crosshair"),
        bind_group_layouts: &[Some(&layout)],
        ..Default::default()
    });
    let over = wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::SrcAlpha,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Zero,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        },
    };
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("crosshair"),
        layout: Some(&pl),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(over),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multiview_mask: None,
        multisample: wgpu::MultisampleState::default(),
        cache: None,
    });
    (pipeline, params, bind)
}

/// Scales the scratch target into a view's place (linear filtering, written as it is).
fn blit_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout, wgpu::Buffer) {
    let source = "struct P { rect: vec4<f32> }
         @group(0) @binding(0) var<uniform> p: P;
         @group(0) @binding(1) var t: texture_2d<f32>;
         @group(0) @binding(2) var smp: sampler;
         @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
             let x = f32((i << 1u) & 2u) * 2.0 - 1.0;
             let y = f32(i & 2u) * 2.0 - 1.0;
             return vec4<f32>(x, y, 0.0, 1.0);
         }
         @fragment fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
             let uv = (pos.xy - p.rect.xy) / p.rect.zw;
             return textureSampleLevel(t, smp, uv, 0.0);
         }";
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("blit"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("blit"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let params = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("blit"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("blit"),
        bind_group_layouts: &[Some(&layout)],
        ..Default::default()
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("blit"),
        layout: Some(&pl),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multiview_mask: None,
        multisample: wgpu::MultisampleState::default(),
        cache: None,
    });
    (pipeline, layout, params)
}

fn request_device(adapter: &wgpu::Adapter) -> Result<(wgpu::Device, wgpu::Queue), ViewerError> {
    let mut limits = wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits());
    // the atlas is a texture array: as many layers as the adapter allows
    limits.max_texture_array_layers = adapter.limits().max_texture_array_layers;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("micro-glancer"),
        required_features: wgpu::Features::empty(),
        required_limits: limits,
        ..Default::default()
    }))
    .map_err(err)
}

fn adapter_label(adapter: &wgpu::Adapter) -> String {
    let i = adapter.get_info();
    format!("{} ({:?}, {:?})", i.name, i.backend, i.device_type)
}
