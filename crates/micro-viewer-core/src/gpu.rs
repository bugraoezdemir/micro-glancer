//! The GPU side: device, the target (a window's surface or an off-screen texture), and
//! per layer the tile atlas (a texture array in the image's own texel format), the
//! buffers the shader reads (page table, levels, channels) and the channels' colour maps.
//!
//! A frame clears to nothing, adds the layers up one by one (each drawn by the pipeline
//! of its texel type, colours and coverage added), then fills the background where no
//! layer drew.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

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
    pub _pad: [f32; 3],
}

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
}

pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    target: Target,
    format: wgpu::TextureFormat,
    /// Per layer id.
    planes: HashMap<u16, Plane>,
    /// Per texel type: the layer pipeline and its bind group layout.
    pipelines: HashMap<Texel, (wgpu::RenderPipeline, wgpu::BindGroupLayout)>,
    /// Fills the background where no layer drew.
    background: wgpu::RenderPipeline,
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

fn instance() -> wgpu::Instance {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::PRIMARY;
    wgpu::Instance::new(desc)
}

fn texel_format(t: Texel) -> (wgpu::TextureFormat, wgpu::TextureSampleType, &'static str) {
    use wgpu::{TextureFormat as F, TextureSampleType as S};
    match t {
        Texel::U8 => (F::R8Uint, S::Uint, "u32"),
        Texel::I8 => (F::R8Sint, S::Sint, "i32"),
        Texel::U16 => (F::R16Uint, S::Uint, "u32"),
        Texel::I16 => (F::R16Sint, S::Sint, "i32"),
        Texel::F32 => (F::R32Float, S::Float { filterable: false }, "f32"),
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
        let instance = instance();
        let surface = instance
            .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                raw_display_handle: Some(display),
                raw_window_handle: window,
            })
            .map_err(err)?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: options.software,
            compatible_surface: Some(&surface),
            apply_limit_buckets: false,
        }))
        .map_err(err)?;
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
        surface.configure(&device, &config);
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
        let instance = instance();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: options.software,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .map_err(err)?;
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
        let background = background_pipeline(&device, format);
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
            pipelines: HashMap::new(),
            background,
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
        if !self.pipelines.contains_key(&texel) {
            let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let made = self.pipeline(texel);
            if let Some(e) = pollster::block_on(scope.pop()) {
                return Err(err(format!("the GPU refused the drawing pipeline: {e}")));
            }
            self.pipelines.insert(texel, made);
        }
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
        let (params, pages, levels, channels, luts) = match old {
            Some(p) => (p.params, p.pages, p.levels, p.channels, p.luts),
            None => (
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("params"),
                    size: std::mem::size_of::<FrameParams>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                storage(&self.device, "pages", 4),
                storage(&self.device, "levels", 48),
                storage(&self.device, "channels", 32),
                lut_texture(&self.device, 1),
            ),
        };
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
            },
        );
        Ok(layers * SLOTS_PER_LAYER)
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

    fn pipeline(&self, texel: Texel) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
        let (_, sample_type, wgsl_type) = texel_format(texel);
        let source = include_str!("shader.wgsl").replace("TEXEL", wgsl_type);
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
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
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        let pl = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
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
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                    entry_point: Some("fs"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.format,
                        blend: Some(wgpu::BlendState {
                            color: add,
                            alpha: add,
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

    /// One frame into `view`: the layers in `frames` (id and parameters) in order, then
    /// the background where none drew (one opaque layer draws it itself).
    fn draw(&mut self, view: &wgpu::TextureView, frames: &[(u16, FrameParams)]) {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        let solo = frames.len() == 1 && frames[0].1.opacity >= 1.0;
        let mut drawn = Vec::new();
        for (id, fp) in frames {
            let Some(p) = self.planes.get_mut(id) else {
                continue;
            };
            let Some((pipeline, layout)) = self.pipelines.get(&p.texel) else {
                continue;
            };
            let mut fp = *fp;
            if solo {
                fp.outside = BACKGROUND;
            }
            self.queue
                .write_buffer(&p.params, 0, bytemuck::bytes_of(&fp));
            let bind = Self::bind_group(&self.device, layout, &self.sampler, p);
            drawn.push((pipeline.clone(), bind));
        }
        let clear = if drawn.is_empty() {
            let [r, g, b, a] = BACKGROUND.map(f64::from);
            wgpu::Color { r, g, b, a }
        } else {
            wgpu::Color::TRANSPARENT
        };
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("frame"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            for (pipeline, bind) in &drawn {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bind, &[]);
                pass.draw(0..3, 0..1);
            }
            if !drawn.is_empty() && !solo {
                pass.set_pipeline(&self.background);
                pass.draw(0..3, 0..1);
            }
        }
        self.queue.submit([enc.finish()]);
    }

    /// One frame to the window.
    pub fn present(&mut self, frames: &[(u16, FrameParams)]) -> Result<(), ViewerError> {
        let frame = match &self.target {
            Target::Surface { surface, config } => match surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(t)
                | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
                // the window changed: configure again, draw at the next frame
                wgpu::CurrentSurfaceTexture::Outdated => {
                    surface.configure(&self.device, config);
                    return Ok(());
                }
                // hidden or busy: skip this frame
                wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                    return Ok(())
                }
                _ => return Err(err("the window's surface was lost")),
            },
            Target::Offscreen { .. } => return Err(err("this viewer draws off-screen")),
        };
        let view = frame.texture.create_view(&Default::default());
        self.draw(&view, frames);
        self.queue.present(frame);
        Ok(())
    }

    /// One frame off-screen, read back into `out` (RGBA8 rows, width x height x 4
    /// bytes) through a readback buffer kept between frames.
    pub fn render_into(
        &mut self,
        frames: &[(u16, FrameParams)],
        out: &mut [u8],
    ) -> Result<(), ViewerError> {
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
        let view = texture.create_view(&Default::default());
        self.draw(&view, frames);
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
                texture: &p.atlas,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: (within % SLOTS_PER_ROW) * TILE + x,
                    y: (within / SLOTS_PER_ROW) * TILE + y,
                    z: slot / SLOTS_PER_LAYER,
                },
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
        let out = slice.get_mapped_range().map_err(err)?[..p.texel.bytes()].to_vec();
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
fn background_pipeline(device: &wgpu::Device, format: wgpu::TextureFormat) -> wgpu::RenderPipeline {
    let [r, g, b, _] = BACKGROUND;
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

fn request_device(adapter: &wgpu::Adapter) -> Result<(wgpu::Device, wgpu::Queue), ViewerError> {
    let mut limits = wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits());
    // the atlas is a texture array: as many layers as the adapter allows
    limits.max_texture_array_layers = adapter.limits().max_texture_array_layers;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("micro-viewer"),
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
