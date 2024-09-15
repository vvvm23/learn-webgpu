use std::borrow::Cow;
use winit::{
    event::{ElementState, Event, KeyEvent, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};
use std::time::{Instant, Duration};

mod camera;
mod constants;
mod texture;
mod instance;
mod time;
mod cube;


struct State<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,

    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,

    time: f32,
    time_buffer: wgpu::Buffer,
    time_bind_group: wgpu::BindGroup,

    instances: Vec<instance::Instance>,
    instance_buffer: wgpu::Buffer,

    #[allow(unused)]
    camera: camera::Camera,
    #[allow(unused)]
    camera_uniform: camera::CameraUniform,
    #[allow(unused)]
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    depth_texture: texture::Texture,

    window: &'a Window,
}

impl<'a> State<'a> {
    async fn new(window: &'a Window) -> State<'a> {
        let mut size = window.inner_size();
        size.width = size.width.max(1);
        size.height = size.height.max(1);

        let instance = wgpu::Instance::default();

        let surface = instance.create_surface(window).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                // Request an adapter which can render to our surface
                compatible_surface: Some(&surface),
            })
            .await
            .expect("Failed to find an appropriate adapter");

        // Create the logical device and command ueue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the swapchain.
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .expect("Failed to create device");

        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an sRGB surface texture. Using a different
        // one will result in all the colors coming out darker. If you want to support non
        // sRGB surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps.formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Immediate,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2
        };
        surface.configure(&device, &config);

        let (vertex_buffer, index_buffer, vertex_size, num_indices) = cube::create_buffers(&device);

        let (time_buffer, time_bind_group_layout, time_bind_group) = time::init_time_utils(1.0, &device);

        let aspect = config.width as f32 / config.height as f32;
        println!("aspect ratio {}", aspect);
        let camera = camera::Camera {
            // position the camera 1 unit up and 2 units back
            // +z is out of the screen
            eye: (0.0, 10.0, 0.1).into(),
            // have it look at the origin
            target: (0.0, 0.0, 0.0).into(),
            // which way is "up"
            up: cgmath::Vector3::unit_y(),
            aspect: config.width as f32 / config.height as f32,
            fovy: 90.0,
            znear: 0.1,
            zfar: 5.0,
        };
        let (camera_buffer, camera_uniform, camera_bind_group_layout, camera_bind_group) = camera::init_camera_utils(&camera, &device);
        let (instances, instance_buffer) = instance::create_instances(&device);

        let vertex_buffers = [cube::create_vertex_buffer_desc(vertex_size), instance::InstanceRaw::desc()];
        let depth_texture = texture::Texture::create_depth_texture(&device, &config, "depth_texture");

        // Load the shaders from disk
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline"),
            bind_group_layouts: &[
                &time_bind_group_layout,
                &camera_bind_group_layout
            ],
            push_constant_ranges: &[],
        });

        let swapchain_capabilities = surface.get_capabilities(&adapter);
        let swapchain_format = swapchain_capabilities.formats[0];

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &vertex_buffers,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                compilation_options: Default::default(),
                targets: &[Some(swapchain_format.into())],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default()
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self {
            surface,
            device,
            queue,
            config,
            size,

            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,

            time: 1.0,
            time_buffer,
            time_bind_group,

            instances,
            instance_buffer,

            camera,
            camera_uniform,
            camera_buffer,
            camera_bind_group,

            depth_texture,

            window,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture = texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
        }
    }

    fn input(&mut self, _event: &WindowEvent) -> bool {
        false
    }

    fn update(&mut self, delta_time: Duration) {
        const SPEED: f32 = 1.0;
        self.time += SPEED * (delta_time.as_micros() as f32) / 1_000_000.0;
        self.queue.write_buffer(&self.time_buffer, 0, bytemuck::cast_slice(&[time::TimeUniform { time: self.time }]));
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let frame = self
            .surface
            .get_current_texture()
            .expect("Failed to acquire next swap chain texture");
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(
                    wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_texture.view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }
                ),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(0, &self.time_bind_group, &[]);
            rpass.set_bind_group(1, &self.camera_bind_group, &[]);
            rpass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            rpass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            rpass.draw_indexed(0..self.num_indices, 0, 0..self.instances.len() as _);
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }
}

async fn run(event_loop: EventLoop<()>, window: Window) {
    let mut size = window.inner_size();
    size.width = size.width.max(1);
    size.height = size.height.max(1);

    let mut state = State::new(&window).await;
    let mut surface_ready = false;
    let mut last_render_time = Instant::now();

    event_loop
        .run(move |event, target| {
            match event {
                Event::WindowEvent {
                    window_id,
                    ref event,
                } if window_id == state.window().id() => {
                    if !state.input(event) {
                        match event {
                            WindowEvent::CloseRequested
                            | WindowEvent::KeyboardInput {
                                event:
                                    KeyEvent {
                                        state: ElementState::Pressed,
                                        physical_key:
                                            PhysicalKey::Code(KeyCode::Escape | KeyCode::KeyQ),
                                        ..
                                    },
                                ..
                            } => target.exit(),
                            WindowEvent::Resized(physical_size) => {
                                surface_ready = true;
                                state.resize(*physical_size);
                            }
                            WindowEvent::RedrawRequested => {
                                state.window().request_redraw();

                                if !surface_ready {
                                    return;
                                }

                                let now = Instant::now();
                                let delta_time = now - last_render_time;
                                state.update(delta_time);
                                last_render_time = now;

                                match state.render() {
                                    Ok(_) => {}
                                    // Reconfigure the surface if lost
                                    Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                                    // The system is out of memory, we should probably quit
                                    Err(wgpu::SurfaceError::OutOfMemory) => target.exit(),
                                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                                    Err(e) => eprintln!("{:?}", e),
                                }
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            };

        })
        .unwrap();
}

pub fn main() {
    let event_loop = EventLoop::new().unwrap();
    #[allow(unused_mut)]
    // let mut builder = winit::window::WindowBuilder::new().with_inner_size(winit::dpi::LogicalSize::new(600, 600));
    let mut builder = winit::window::WindowBuilder::new()
        .with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::JsCast;
        use winit::platform::web::WindowBuilderExtWebSys;
        let canvas = web_sys::window()
            .unwrap()
            .document()
            .unwrap()
            .get_element_by_id("canvas")
            .unwrap()
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .unwrap();
        builder = builder.with_canvas(Some(canvas));
    }
    let window = builder.build(&event_loop).unwrap();

    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        pollster::block_on(run(event_loop, window));
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        wasm_bindgen_futures::spawn_local(run(event_loop, window));
    }
}
