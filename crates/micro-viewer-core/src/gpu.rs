//! The GPU side: device, the target (a window's surface or an off-screen texture), the
//! pipeline that draws a plane from the tile atlas, and the atlas itself.

use bytemuck::{Pod, Zeroable};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use crate::tiles::{SLOTS_PER_ROW, TILE};
use crate::ViewerError;

/// The shader's parameters (std140-compatible, 48 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Params {
    pub origin: [f32; 2],
    pub scale: f32,
    pub lo: f32,
    pub hi: f32,
    pub width: f32,
    pub height: f32,
    pub grid_w: u32,
    pub tile: u32,
    pub slots_per_row: u32,
    pub _pad: [u32; 2],
}

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
}

pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    target: Target,
    format: wgpu::TextureFormat,
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    atlas: wgpu::Texture,
    pages: wgpu::Buffer,
    pages_len: usize,
    bind: Option<wgpu::BindGroup>,
    /// The off-screen frames' readback buffer, kept between frames.
    readback: Option<wgpu::Buffer>,
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
        // a plain (non-sRGB) format: the shader's grey values are written as they are
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
        Ok(Self::build(
            device,
            queue,
            Target::Surface { surface, config },
            format,
            info,
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
        Ok(Self::build(
            device,
            queue,
            Target::Offscreen {
                texture,
                width: width.max(1),
                height: height.max(1),
            },
            format,
            info,
        ))
    }

    fn build(
        device: wgpu::Device,
        queue: wgpu::Queue,
        target: Target,
        format: wgpu::TextureFormat,
        adapter_info: String,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("plane"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("plane"),
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
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("plane"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
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
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let side = TILE * SLOTS_PER_ROW;
        let atlas = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("atlas"),
            size: wgpu::Extent3d {
                width: side,
                height: side,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let pages = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pages"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            device,
            queue,
            target,
            format,
            pipeline,
            layout,
            params,
            atlas,
            pages,
            pages_len: 1,
            bind: None,
            readback: None,
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

    /// Tile data (`TILE` x `TILE` float32, row-major; a smaller edge tile padded by the
    /// caller) into atlas slot `slot`.
    pub fn upload_tile(&self, slot: u32, data: &[f32], w: u32, h: u32) {
        let (sx, sy) = ((slot % SLOTS_PER_ROW) * TILE, (slot / SLOTS_PER_ROW) * TILE);
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.atlas,
                mip_level: 0,
                origin: wgpu::Origin3d { x: sx, y: sy, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * w),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
    }

    /// The page table (slot + 1 per tile, 0 for none).
    pub fn upload_pages(&mut self, pages: &[u32]) {
        if pages.len() != self.pages_len {
            self.pages = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pages"),
                size: (4 * pages.len().max(1)) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.pages_len = pages.len();
            self.bind = None;
        }
        if !pages.is_empty() {
            self.queue
                .write_buffer(&self.pages, 0, bytemuck::cast_slice(pages));
        }
    }

    fn bind_group(&mut self) -> &wgpu::BindGroup {
        if self.bind.is_none() {
            let view = self.atlas.create_view(&Default::default());
            self.bind = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("plane"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.pages.as_entire_binding(),
                    },
                ],
            }));
        }
        self.bind.as_ref().expect("made above")
    }

    fn draw(&mut self, view: &wgpu::TextureView, params: &Params, plane: bool) {
        self.queue
            .write_buffer(&self.params, 0, bytemuck::bytes_of(params));
        let bind = plane.then(|| self.bind_group().clone());
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("frame"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.08,
                            g: 0.08,
                            b: 0.08,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if let Some(bind) = &bind {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, bind, &[]);
                pass.draw(0..3, 0..1);
            }
        }
        self.queue.submit([enc.finish()]);
    }

    /// One frame to the window (`plane`: whether a plane is open).
    pub fn present(&mut self, params: &Params, plane: bool) -> Result<(), ViewerError> {
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
        self.draw(&view, params, plane);
        self.queue.present(frame);
        Ok(())
    }

    /// One frame off-screen, read back as RGBA8 rows (width x height x 4 bytes).
    pub fn render_rgba(&mut self, params: &Params, plane: bool) -> Result<Vec<u8>, ViewerError> {
        let (w, h) = self.size();
        let mut out = vec![0u8; (4 * w * h) as usize];
        self.render_into(params, plane, &mut out)?;
        Ok(out)
    }

    /// One frame off-screen, read back into `out` (RGBA8 rows, width x height x 4
    /// bytes) through a readback buffer kept between frames.
    pub fn render_into(
        &mut self,
        params: &Params,
        plane: bool,
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
        self.draw(&view, params, plane);
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

fn request_device(adapter: &wgpu::Adapter) -> Result<(wgpu::Device, wgpu::Queue), ViewerError> {
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("micro-viewer"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits()),
        ..Default::default()
    }))
    .map_err(err)
}

fn adapter_label(adapter: &wgpu::Adapter) -> String {
    let i = adapter.get_info();
    format!("{} ({:?}, {:?})", i.name, i.backend, i.device_type)
}
